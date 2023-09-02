# Standard Libraries
import os

# Third-Party Libraries
import numpy as np
from einops import rearrange
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

# Local Modules
from src.utils.dist_utils import all_gather
from src.evals import (
    sdr_distance_metric, calc_dists_from_centers, unravel_index,
    PostProcessTouchdownSDR, PostProcessViLTGrounding, PostProcessMDETRGrounding,
    TouchdownSDREvaluator, GroundingHeatmapEvaluator, GroundingBoxEvaluator,
    F30kRefGameEvaluator, TangramGameEvaluator, DBVisualizer
)

# Loss Functions
kl_loss_fnc = nn.KLDivLoss(reduction='none')
cls_loss = nn.CrossEntropyLoss(reduction='none')


def compute_feature_map(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    bs, slen, dim = infer["image_feats"].shape
    image_feats = infer["image_feats"][:,1:,:] 

    _, _, h, w = batch["image"][0].shape
    h, w = int(h / 32), int(w / 32)
    feature_map = image_feats.view(bs, h, w, dim) 
    feature_map = feature_map.permute(0,3,1,2)  
    return infer, feature_map


def compute_vilt_grounding_loss(pl_module, batch, infer, feature_map):
    device = feature_map.device
    bbox_target_tensors_list = [msk for msk in batch["bbox_tensors"] if msk is not None] if "bbox_tensors" in batch else []
    if len(bbox_target_tensors_list) == 0:
        return {}
    bbox_target_tensors = torch.cat(bbox_target_tensors_list, 0)

    # stack image and text features
    stacked_imgfeats = []
    stacked_textfeats = []
    stacked_text_ids = []
    bbox_text_masks = []

    for i, msk in enumerate(batch["bbox_positive_maps"]):
        if msk is not None:
            stacked_imgfeats.append(feature_map[i, ...].unsqueeze(0).repeat(msk.shape[0], 1, 1, 1))
            stacked_textfeats.append(infer["text_feats"][i, ...].unsqueeze(0).repeat(msk.shape[0], 1, 1))
            stacked_text_ids.append(batch["text_ids"][i, ...].unsqueeze(0).repeat(msk.shape[0], 1))
            bbox_text_masks.append(msk)
        
    stacked_imgfeats = torch.cat(stacked_imgfeats, 0)
    stacked_textfeats = torch.cat(stacked_textfeats, 0)
    stacked_bbox_text_masks = torch.cat(bbox_text_masks, 0)
    text_mask = ~stacked_bbox_text_masks.bool()
    text_cts = (torch.sum(stacked_bbox_text_masks, 1) + 1e-9).unsqueeze(-1)
    stacked_textfeats[text_mask, ...] = 0
    pooled_bbox_text_rep = torch.sum(stacked_textfeats, 1)
    pooled_bbox_text_rep /= text_cts

    # model prediction
    output_size = tuple(bbox_target_tensors.shape[1:])
    
    if pl_module.hparams.config["model_cls"] == "vilt_aligner_probe":
        bbox_logit = pl_module.probe_head(stacked_imgfeats, pooled_bbox_text_rep, output_size=output_size)
    elif pl_module.split_detection_head:
        bbox_logit = pl_module.detection_head(stacked_imgfeats, pooled_bbox_text_rep, output_size=output_size)
    else:
        bbox_logit = pl_module.sdr_head(stacked_imgfeats, pooled_bbox_text_rep, output_size=output_size)

    batch_size, height, width = bbox_logit.shape
    bbox_log_pred = F.log_softmax(bbox_logit.view(batch_size, -1), 1).view(batch_size, height, width)
    bbox_losses_per_bboxes = torch.sum(kl_loss_fnc(bbox_log_pred, bbox_target_tensors), axis =[1,2])

    # accumulate and normalize loss
    bbox_losses = []
    ptr = 0
    for i, msk in enumerate(batch["bbox_positive_maps"]):
        if msk is None:
            # no bbox annotations for the example
            bbox_losses.append(torch.tensor(0).to(device))
        else:
            num_bboxes = msk.shape[0]
            bbox_losses.append(torch.mean(bbox_losses_per_bboxes[ptr:ptr+num_bboxes]))
            ptr += num_bboxes
    assert(len(bbox_losses_per_bboxes) == ptr)
    bbox_losses = torch.stack(bbox_losses)

    return {
        "grounding_log_pred": bbox_log_pred,
        "grounding_losses": bbox_losses
    }


def compute_mdetr_grounding_loss(pl_module, batch, infer):
    rets = {}
    out = {} 
    hidden_states = infer["hidden_states"]
    num_queries = pl_module.mdetr.query_embed.weight.shape[0]
    hidden_states = hidden_states[:, :, :num_queries]
    memory_cache = infer["memory_cache"]

    # predicting bounding boxes
    outputs_class = pl_module.mdetr.class_embed(hidden_states)
    outputs_coord = pl_module.mdetr.bbox_embed(hidden_states).sigmoid()
    out.update(
        {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
    )

    outputs_isfinal = None
    if pl_module.mdetr.isfinal_embed is not None:
        outputs_isfinal = pl_module.mdetr.isfinal_embed(hidden_states)
        out["pred_isfinal"] = outputs_isfinal[-1]

    # calculating auxuilary losses
    proj_queries, proj_tokens = None, None
    if pl_module.mdetr.contrastive_align_loss:
        proj_queries = F.normalize(pl_module.mdetr.contrastive_align_projection_image(hidden_states), p=2, dim=-1)
        proj_tokens = F.normalize(
            pl_module.mdetr.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
        )
        out.update(
            {
                "proj_queries": proj_queries[-1],
                "proj_tokens": proj_tokens,
                "tokenized": memory_cache["tokenized"],
            }
        )
    if pl_module.mdetr.aux_loss:
        if pl_module.mdetr.contrastive_align_loss:
            assert proj_tokens is not None and proj_queries is not None
            out["aux_outputs"] = [
                {
                    "pred_logits": a,
                    "pred_boxes": b,
                    "proj_queries": c,
                    "proj_tokens": proj_tokens,
                    "tokenized": memory_cache["tokenized"],
                }
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
            ]
        else:
            out["aux_outputs"] = [
                {
                    "pred_logits": a,
                    "pred_boxes": b,
                }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        if outputs_isfinal is not None:
            assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
            for i in range(len(outputs_isfinal[:-1])):
                out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]

    # calculating loss
    batch_size = len(batch["bbox_positive_maps"])
    positive_map = []
    for b in range(batch_size):
        for j in range(len(batch["bbox_coords"][b])):
            for _ in range(len(batch["bbox_coords"][b][j])):
                positive_map.append(batch["bbox_positive_maps"][b][j, :].unsqueeze(0))
    
    calc_fake_loss = False
    if len(positive_map) == 0:
        # handle the case of no grounding example in the batch, causing DDP to break.
        # ! cleaner way is assigning gradinets manually during backward pass
        calc_fake_loss = True
        positive_map = torch.zeros_like(batch["text_ids"]).float()
        all_tokens_positive = []
        for b in range(batch_size):
            all_tokens_positive.append([[[0,1]]])
        all_tokens_positive = all_tokens_positive
        device = pl_module.device
        bbox_labels = [[torch.tensor([1]).to(device)] for b in range(batch_size)]
        bbox_coords = [
            [torch.tensor([[0.5, 0.5, 0.01, 0.01]]).to(device)]
            for b in range(batch_size)
        ]
    else:
        positive_map = torch.cat(positive_map).float()
        all_tokens_positive = []
        for b in range(batch_size):
            tokens_positive = []
            for j in range(len(batch["bbox_coords"][b])):
                for _ in range(len(batch["bbox_coords"][b][j])):
                    tokens_positive.append(batch["bbox_token_postives"][b][j])
            all_tokens_positive.append(tokens_positive)
        bbox_labels = batch["bbox_labels"]
        bbox_coords = batch["bbox_coords"]

    targets = [
        {
            "labels": torch.cat(bbox_labels[b]),
            "boxes": torch.cat(bbox_coords[b]),
            "tokens_positive": all_tokens_positive[b]
        }
        for b in range(batch_size)
    ]
    losses = pl_module.criterion(out, targets, positive_map)
    if calc_fake_loss:
        for k in losses.keys():
            losses[k] = losses[k] * 0.
    else:
        rets.update(losses)

    # Update results for per phrase prediction 
    rets.update({
        "grounding_output": out,
        "grounding_losses": torch.stack([torch.tensor([0.]).to(pl_module.device) for _ in range(batch_size)])
    })
    
    return rets


def compute_grounding_stats(pl_module, batch, grounding_preds):
    rets = {}
    phase = "train" if pl_module.training else "val"  
    # calculate grounding statisitcs
    if phase == "train":  
        alias = f"{phase}_grounding"
        for k in grounding_preds.keys():
            if "loss" in k:
                np_losses = np.array(grounding_preds[k].tolist())  
                np_loss = np.mean(np_losses)
                rets.update({"wandb/{}/{}".format(alias, k): np_loss})
    else:
        results = {}
        # post-processing results
        for post_processor in pl_module.postprocessors:
            if isinstance(post_processor, (PostProcessViLTGrounding, PostProcessMDETRGrounding)):
                results.update(post_processor(grounding_preds, batch))

        # updating evaluation statisitcs
        for evaluator in pl_module.evaluators:
            if isinstance(evaluator, (GroundingHeatmapEvaluator, GroundingBoxEvaluator)):
                evaluator.update(results)

    return rets


def compute_vilt_touchdown_sdr_loss(pl_module, batch, infer, feature_map):
    batch_size = feature_map.shape[0]
    device = feature_map.device

    # get predictions   
    feature_map.get_device()
    sdr_losses = torch.zeros((batch_size)).to(device) 
    text_mask = batch["text_positive_map"].bool()
    text_cts = (torch.sum(batch["text_positive_map"], 1) + 1e-9).unsqueeze(-1)
    masked_rep = torch.zeros_like(infer["text_feats"])
    masked_rep[text_mask, ...] = infer["text_feats"][text_mask, ...]
    pooled_text_rep = torch.sum(masked_rep, 1)
    pooled_text_rep /= text_cts
    head_output = pl_module.sdr_head(feature_map, pooled_text_rep, output_size=tuple(batch["target"].shape[1:]))     # TODO: make sure this line does not break Touchdown training.

    # calculate losses and post prosess output
    batch_size, height, width = head_output.shape
    sdr_logpreds = F.log_softmax(head_output.view(batch_size, -1), 1).view(batch_size, height, width)
    sdr_losses = torch.sum(kl_loss_fnc(sdr_logpreds, batch["target"]), axis =[1,2])
    
    return {
        "sdr_logpreds": sdr_logpreds,
        "sdr_losses": sdr_losses
    }


def compute_mdetr_touchdown_sdr_loss(pl_module, batch, infer):
    loss = {}
    hidden_states = infer["hidden_states"]

    # Touchdowdn SDR coordinates prediction
    sdr_coords_gt = batch["target_coords"]
    sdr_embeds = hidden_states[0, :, -1]
    sdr_coords = pl_module.sdr_head(sdr_embeds).sigmoid()[:, :2] 

    # Touchdown SDR regression loss
    sdr_losses = F.l1_loss(sdr_coords, sdr_coords_gt, reduction="none").sum(axis=1)
    #print(sdr_coords, sdr_coords_gt)
    #print(sdr_losses)

    return {
        "sdr_coords": sdr_coords,
        "sdr_losses": sdr_losses
    }


def compute_touchdown_sdr_stats(pl_module, batch, sdr_preds):
    phase = "train" if pl_module.training else "val"  
    rets = {}
    sdr_losses = sdr_preds["sdr_losses"]

    # calculate Touchdown SDR statisitcs
    if phase == "train":
        np_sdr_losses = np.array(sdr_losses.tolist())  
        if "sdr_logpreds" in sdr_preds:
            sdr_logpreds = sdr_preds["sdr_logpreds"]
            sdr_dists = np.array(sdr_distance_metric(sdr_logpreds, batch["target"]))
        elif "sdr_coords" in sdr_preds:
            sdr_coords_gt = batch["target_coords"]
            sdr_coords = sdr_preds["sdr_coords"]
            if sdr_coords.dtype == torch.bfloat16:
                sdr_dists = calc_dists_from_centers(sdr_coords.float().cpu().detach().numpy(), sdr_coords_gt.cpu().detach().numpy())
            else:
                sdr_dists = calc_dists_from_centers(sdr_coords.cpu().detach().numpy(), sdr_coords_gt.cpu().detach().numpy()) # TODO: sdr_coords_gt.cpu().detach().numpy() seems to be the source of dsicrepahncy. 

        alias = f"{phase}_touchdown_sdr"
        sdr_loss, sdr_dist, sdr_acc = np.mean(np_sdr_losses), np.mean(sdr_dists), np.mean((sdr_dists < 80))
        rets.update(
            {
                "wandb/sdr/loss_touchdown_sdr": sdr_loss,
                "wandb/sdr/mean distance": sdr_dist,
                "wandb/sdr/accuracy@80px": sdr_acc
            }
        )
        _ = getattr(pl_module, f"{alias}_loss")(sdr_loss)
        _ = getattr(pl_module, f"{alias}_dist")(sdr_dist)
        _ = getattr(pl_module, f"{alias}_accuracy")(sdr_acc)
    else:
        output = {
            "sdr_losses": sdr_losses,
        }
        if "sdr_logpreds" in sdr_preds:
            output.update({"sdr_logpreds": sdr_preds["sdr_logpreds"]})
        elif "sdr_coords" in sdr_preds:
            output.update({"sdr_coords": sdr_preds["sdr_coords"]})

        results = {}
        for post_processor in pl_module.postprocessors:
            if isinstance(post_processor, (PostProcessTouchdownSDR)):
                results.update(post_processor(output, batch))
        for evaluator in pl_module.evaluators:
            if isinstance(evaluator, (TouchdownSDREvaluator)):
                evaluator.update(results)
            elif isinstance(evaluator, (DBVisualizer)):
                evaluator.update(batch, sdr_preds)

    return rets


def compute_vilt_touchdown_loss(pl_module, batch):
    ret = {}
    losses = []

    # calculate feature map
    infer, feature_map = compute_feature_map(pl_module, batch)

    # calculate Touchdown SDR losses
    sdr_rets = compute_vilt_touchdown_sdr_loss(pl_module, batch, infer, feature_map)
    sdr_stats = compute_touchdown_sdr_stats(pl_module, batch, sdr_rets)
    ret.update(sdr_stats)

    # skip touchdown loss during probing
    if pl_module.hparams.config["model_cls"] != "vilt_aligner_probe":
        losses.append(torch.mean(sdr_rets["sdr_losses"]))

    # calculate grounding losses
    if pl_module.hparams.config["use_grounding"]:
        grounding_rets = compute_vilt_grounding_loss(pl_module, batch, infer, feature_map)
        if len(grounding_rets) != 0:
            grounding_stats = compute_grounding_stats(pl_module, batch, grounding_rets)
            ret.update(grounding_stats)
            losses.append(torch.mean(grounding_rets["grounding_losses"]))
        
    # calculate a total loss
    if (pl_module.hparams.config["model_cls"] == "vilt_aligner_probe" and not pl_module.hparams.config["use_grounding"]) or len(losses) == 0:
        ret["kl_loss"] = torch.tensor([0]).to(pl_module.device)
    else:
        kl_loss = torch.mean(torch.stack(losses))
        ret["kl_loss"] = kl_loss

    return ret


def compute_mdetr_touchdown_loss(pl_module, batch):
    ret = {}
    losses = {}

    # calculate featues
    infer = pl_module.infer(batch)

    # calculate Touchdown SDR losses
    sdr_rets = compute_mdetr_touchdown_sdr_loss(pl_module, batch, infer)
    sdr_stats = compute_touchdown_sdr_stats(pl_module, batch, sdr_rets)
    ret.update(sdr_stats)
    losses["loss_touchdown_sdr"] = torch.mean(sdr_rets["sdr_losses"])

    # calculate grounding losses
    if pl_module.hparams.config["use_grounding"]:
        grounding_rets = compute_mdetr_grounding_loss(pl_module, batch, infer)
        for r in grounding_rets:
            if "loss" in r:
                losses[r] = grounding_rets[r]
        if len(grounding_rets) != 0:
            grounding_stats = compute_grounding_stats(pl_module, batch, grounding_rets)
            ret.update(grounding_stats)

    # calculate a total loss    
    ret["total_loss"] = sum(losses[k] * pl_module.weight_dict[k] for k in losses.keys() if k in pl_module.weight_dict)
    phase = "train" if pl_module.training else "val"  
    if phase == "train":
        ret["wandb/total_loss"] = ret["total_loss"] 
    return ret


def compute_vilt_cls_loss(pl_module, batch, infer):
    device = infer["image_feats"].device
    scores = pl_module.contrastive_head(infer["cls_feats"]) 
    answer = torch.tensor(batch["label_t"]).to(device).long() 
    cls_losses = cls_loss(scores, answer)

    return {
        'cls_losses': cls_losses,
        'cls_scores': scores
    }


def compute_mdetr_cls_loss(pl_module, batch, infer): 
    hidden_states = infer["hidden_states"]
    class_embeds = hidden_states[0, :, -1]
    device = class_embeds.device
    scores = pl_module.contrastive_head(class_embeds)
    answer = torch.tensor(batch["label_t"]).to(device).long() 
    cls_losses = cls_loss(scores, answer)
    return {
        'cls_losses': cls_losses,
        'cls_scores': scores
    }


def compute_cls_stats(pl_module, batch, refgame_preds):
    cls_losses = refgame_preds["cls_losses"]
    ret = {
        "wandb/cls_losses": torch.mean(cls_losses)
    }
    phase = "train" if pl_module.training else "val"  
    device = refgame_preds["cls_losses"].device
    batch_size = refgame_preds["cls_losses"].shape[0]
    answers = torch.tensor(batch["label_t"]).to(device).long() 

    if "cls_scores" in refgame_preds:
        cls_scores = refgame_preds["cls_scores"]
        preds = torch.argmax(cls_scores, dim=1)
    elif "cls_coords" in refgame_preds:
        num_images = torch.tensor([len(i) for i in batch["label_i_orders"]]).to(device)
        rge = 1 / num_images 
        y_preds = refgame_preds["cls_coords"][:, 1]
        preds = torch.floor(y_preds / rge)
    elif "cls_logpreds" in refgame_preds:
        num_images = torch.tensor([len(i) for i in batch["label_i_orders"]]).to(device)
        rge = 1 / num_images 
        max_inds = refgame_preds["cls_logpreds"].flatten(-2).argmax(-1)
        preds = unravel_index(max_inds, refgame_preds["cls_logpreds"].shape)
        y_preds = torch.stack(preds).transpose(0,1)[:,1]
        y_preds_rel = y_preds / torch.tensor(refgame_preds["cls_logpreds"].shape[1]).to(device)
        preds = torch.floor(y_preds_rel / rge)

    if phase == "train":
        batch_correct_t = int(torch.sum((preds == answers)))
        accuracy = batch_correct_t / batch_size
        ret.update(
            {
                'wandb/accuracy': accuracy, 
            }
        )
    elif phase == "val":
        batch_correct = (preds == answers).int().tolist()
        results = {
            "identifier": batch["identifier"],
            "correct_t": batch_correct,
            "loss": [cls_losses[b].item() for b in range(batch_size)],
            'split': batch["split"]
        }

        if "cls_scores" in refgame_preds:
            with torch.no_grad():
                probs = F.softmax(cls_scores, 1)
                batch_tensor = torch.tensor([i for i in range(batch_size)])
                results["score"] =  probs[batch_tensor, answers].tolist()
        elif "cls_coords" in refgame_preds:
            # compute L1 distance
            score = torch.diagonal(torch.cdist(batch["target_coords"], refgame_preds["cls_coords"], p=1))
            results["score"] = score.tolist()
        elif "cls_logpreds" in refgame_preds:
            cls_preds = torch.exp(refgame_preds["cls_logpreds"]).flatten(-2)
            max_inds = batch["target"].flatten(-2).argmax(-1)
            batch_inds = torch.tensor([i for i in range(batch_size)])
            score = cls_preds[batch_inds, max_inds]
            results["score"] = score.tolist()

        for evaluator in pl_module.evaluators:
            if isinstance(evaluator, (TangramGameEvaluator)):
                results["tangram_id"] = batch["tangram_id"]
                evaluator.update(results)
            elif isinstance(evaluator, (F30kRefGameEvaluator)):
                evaluator.update(results)
            elif isinstance(evaluator, (DBVisualizer)):
                evaluator.update(batch, refgame_preds)

    return ret


def compute_vilt_refgame_loss(pl_module, batch):
    ret = {}
    losses = []
    # calculate feature map
    infer, feature_map = compute_feature_map(pl_module, batch)

    # calculate refernc game losses
    if pl_module.hparams.config["use_touchdown_loss_for_cls"]:
        sdr_rets = compute_vilt_touchdown_sdr_loss(pl_module, batch, infer, feature_map)
        refgame_rets = {
            "cls_logpreds": sdr_rets["sdr_logpreds"],
            "cls_losses": sdr_rets["sdr_losses"],
        }
    else:
        refgame_rets = compute_vilt_cls_loss(pl_module, batch, infer)
    refgame_stats = compute_cls_stats(pl_module, batch, refgame_rets)
    ret.update(refgame_stats)
    refgame_rets["cls_losses"] *= pl_module.hparams.config["refgame_options"]["cls_loss_alpha"]  # scale contrastive losse
    losses.append(torch.mean(refgame_rets["cls_losses"]))

    # calculate grounding losses
    if pl_module.hparams.config["use_grounding"]:
        grounding_rets = compute_vilt_grounding_loss(pl_module, batch, infer, feature_map)
        if len(grounding_rets) != 0:
            grounding_stats = compute_grounding_stats(pl_module, batch, grounding_rets)
            ret.update(grounding_stats)
            losses.append(torch.mean(grounding_rets["grounding_losses"]))

    # scale loss
    total_loss = torch.mean(torch.stack(losses))
    ret["total_loss"] = total_loss

    return ret


def compute_mdetr_refgame_loss(pl_module, batch):
    ret = {}
    losses = {}

    # calculate feature map
    infer = pl_module.infer(batch)
 
    # calculate reference game losses
    if pl_module.hparams.config["use_touchdown_loss_for_cls"]:
        sdr_rets = compute_mdetr_touchdown_sdr_loss(pl_module, batch, infer)
        refgame_rets = {
            "cls_coords": sdr_rets["sdr_coords"],
            "cls_losses": sdr_rets["sdr_losses"],
        }
    else:
        refgame_rets = compute_mdetr_cls_loss(pl_module, batch, infer)

    refgame_stats = compute_cls_stats(pl_module, batch, refgame_rets)
    ret.update(refgame_stats)
    losses["loss_cls"] = torch.mean(refgame_rets["cls_losses"])

    # calculate grounding losses
    if pl_module.hparams.config["use_grounding"]:
        grounding_rets = compute_mdetr_grounding_loss(pl_module, batch, infer)
        for r in grounding_rets:
            if "loss" in r:
                losses[r] = grounding_rets[r]

        if len(grounding_rets) != 0:
            grounding_stats = compute_grounding_stats(pl_module, batch, grounding_rets)
            ret.update(grounding_stats)

    # calculate a total loss
    all_losses = [losses[k] * pl_module.weight_dict[k] for k in losses.keys() if k in pl_module.weight_dict]
    ret["total_loss"] = sum(all_losses)
    phase = "train" if pl_module.training else "val"  
    if phase == "train":
        ret["wandb/total_loss"] = ret["total_loss"] 
    return ret
