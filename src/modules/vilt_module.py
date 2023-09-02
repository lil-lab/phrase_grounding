# source: https://github.com/dandelin/ViLT
# modified by Noriyuki Kojima
# Standard Libraries
import os
import time
import copy
import pickle
import uuid
import pprint

# Third-party Libraries
import psutil
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from IPython import embed
from typing import Optional, Dict
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from psutil._common import bytes2human

# Local Modules
from src.utils import objectives, dist_utils
from src.modules import heads, metric_utils, vilt_utils, vision_transformer as vit
from src.evals import (
    TouchdownSDREvaluator,
    F30kRefGameEvaluator,
    TangramGameEvaluator,
    GroundingHeatmapEvaluator,
    DBVisualizer
)
import src.utils.dist_utils as dist


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # intialize ViLT
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(vilt_utils.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(vilt_utils.init_weights)
        
        pretrained = True if self.hparams.config["load_path"] == "" else False
        self.transformer = getattr(vit, self.hparams.config["vit"])(
            pretrained=pretrained, config=self.hparams.config
        )
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(vilt_utils.init_weights)

        self.sdr_head = heads.SDRConcatDeconvHead() 
        self.prepare_sdr = True
        self.prepare_refgame = (self.hparams.config["loss_names"]["refgame"] > 0) and (not self.hparams.config["use_touchdown_loss_for_cls"])
        if self.prepare_refgame:
            self.contrastive_head = heads.ContrastiveHead(hidden_size=config["hidden_size"], num_distractors=config["refgame_options"]["num_distractors"])
            self.contrastive_head.apply(vilt_utils.init_weights)


        self.split_detection_head = self.hparams.config["vilt_split_detection_head"]
        if self.split_detection_head:
            self.detection_head = heads.SDRConcatDeconvHead() 

        # initialize metrics
        metric_utils.set_metrics(self)
        self.wandb_logger = None
        self.use_wandb = True if not("disable_wandb" in self.hparams.config and self.hparams.config["disable_wandb"]) else False
        self.current_tasks = list()
        self.process = psutil.Process(os.getpid())

        # load a full or partial model
        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            if config["load_from_ema"]:
                if "callbacks" in ckpt:
                    keys = list(ckpt["callbacks"].keys())
                    if "EMA" in keys:
                        state_dict = ckpt["callbacks"]["EMA"]["ema_state_dict"]
                    else:
                        raise ValueError("EMA checkpoints not found.")
            else:
                state_dict = ckpt["state_dict"]

            if self.hparams.config["test_only"]:
                # loading checkpoint from ema
                self.load_state_dict(state_dict, strict=False)
            else:
                state_dict = vilt_utils.initialize_vilt_state_dict(self, state_dict)
                self.load_state_dict(state_dict, strict=False)
                if self.split_detection_head:
                    self.detection_head = copy.deepcopy(self.sdr_head)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
        text_ids = batch[f"text_ids"]
        text_labels = batch[f"text_labels"]
        text_masks = batch["text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)


        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Touchdown SDR 
        if "touchdown_sdr" in self.current_tasks:
            ret.update(objectives.compute_vilt_touchdown_loss(self, batch))

        # Flickr 30k Entities reference game
        if "refgame" in self.current_tasks:
            ret.update(objectives.compute_vilt_refgame_loss(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        metric_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if ("loss" in k and "wandb" not in k)])

        if dist.is_main_process():
            if self.wandb_logger is not None:
                train_stats = {}
                train_stats["total_loss"] = total_loss.item()
                for k in output:
                    if "wandb" in k:
                        train_stats[k.replace("wandb/", "")] = output[k]
                memory_stats = metric_utils.get_memory_stats(self)
                lr_stats = {f"lr_group{i}": list(self.optimizers().param_groups)[i]["lr"] for i in range(len(self.optimizers().param_groups))}

                self.wandb_logger.log(results=train_stats, split="train", step=self.global_step)
                self.wandb_logger.log(results=memory_stats, split="memory", step=self.global_step)
                self.wandb_logger.log(results=lr_stats, split="lr", step=self.global_step)

        return total_loss

    def training_epoch_end(self, outs):
        metric_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        metric_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_start(self):
        self._start_time = time.time()

    def on_train_start(self): 
        # source: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
        if (self.hparams.config["resume_from"] != "") and (self.hparams.config["resume_from"] is not None):
            # ! dealing with different PyTorch versions
            try:
                self.lr_schedulers().optimizer = self.trainer.strategy._lightning_optimizers[0] 
            except:
                # FIXME: maybe this line is not necessary
                self.lr_schedulers().optimizer = self.trainer.optimizers[0]
                
            self.lr_schedulers().last_epoch -= 1
            self.lr_schedulers().step()
            # reset wandb
            if dist.is_main_process():
                if self.use_wandb and (self.wandb_logger is None):
                    from src.utils import WandbLogger
                    from argparse import Namespace
                    self.wandb_logger = WandbLogger(Namespace(**self.hparams.config))
                    nn_modules_to_monitor = []
                    if self.prepare_sdr:
                        nn_modules_to_monitor.append(self.sdr_head)
                    elif self.prepare_refgame:
                        nn_modules_to_monitor.append(self.contrastive_head)
                    wandb.watch(nn_modules_to_monitor, log='all', log_freq=500)

    def validation_epoch_end(self, outs):
        summaries = {}
        self.checkpoint_metric = 0.

        if self.hparams.config["loss_names"]["touchdown_sdr"] > 0 or self.hparams.config["loss_names"]["refgame"] > 0:
            for evaluator in self.evaluators:
                evaluator.synchronize_between_processes()
                if dist.is_main_process():
                    summary = evaluator.summarize()
                    summaries.update(summary)
                evaluator.reset()
                if "debug" in summaries and "accuracy@80px" in summaries["debug"]:
                    self.checkpoint_metric += summaries["debug"]["accuracy@80px"]
            print(summaries)

        # logging to wandb
        if dist.is_main_process():
            # initializing wandb logger right after the first debugging steps
            if self.use_wandb and (self.wandb_logger is None):
                from src.utils import WandbLogger
                from argparse import Namespace
                self.wandb_logger = WandbLogger(Namespace(**self.hparams.config))
                if self.prepare_refgame:
                    nn_modules_to_monitor = [self.contrastive_head]
                else:
                    nn_modules_to_monitor = [self.sdr_head]
                wandb.watch(nn_modules_to_monitor, log='all', log_freq=500)
            elif self.wandb_logger is not None:
                for sp in summaries.keys():
                    self.wandb_logger.log(results=summaries[sp], split=sp, step=self.global_step, commit=False)

        metric_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        metric_utils.set_task(self)
        output = self(batch)
        ret = dict()
        
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        summaries = {}
        self.checkpoint_metric = 0.

        # summarize evaluation results
        for evaluator in self.evaluators:
            evaluator.synchronize_between_processes()
            if dist.is_main_process():
                if isinstance(evaluator, (TouchdownSDREvaluator, F30kRefGameEvaluator, TangramGameEvaluator)):
                    task_evaluator = evaluator
                    summary, task_scores = task_evaluator.summarize(eval_correlation=True)
                elif isinstance(evaluator, GroundingHeatmapEvaluator):
                    grounding_evaluator = evaluator
                    summary = grounding_evaluator.summarize(task_scores)
                elif isinstance(evaluator, DBVisualizer):
                    complementary_evaluator = evaluator

                summaries.update(summary)
            
        # print out result summary
        for sp in summaries:
            print("="*10 + f" {sp} " + "="*10)
            pprint.pprint(summaries[sp])
        outdir = "eval_outputs/{}".format(self.hparams.config["exp_name"])
        os.makedirs(outdir, exist_ok=True)
        summary_save_path = os.path.join(outdir, "sumamry.pkl")
        pickle.dump(summaries, open(summary_save_path, "wb"))

        # save evaluation results to database
        if dist.is_main_process():
            if self.hparams.config["test_update_db"]:
                from src.evals import update_db
                task_type = self.hparams.config["datasets"][0]
                exp_name = self.hparams.config["exp_name"]
                update_db(task_evaluator, grounding_evaluator, complementary_evaluator, task_type, exp_name)

        # wrap up epoch
        if "debug" in summaries and "accuracy@80px" in summaries["debug"]:
            self.checkpoint_metric += summaries["debug"]["accuracy@80px"]
        metric_utils.epoch_wrapup(self)


    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)

def print_cuda_memory():
    print("Total GPU Memory (in GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    print("Allocated GPU Memory (in GB):", torch.cuda.memory_allocated(0) / 1e9)
    print("Reserved GPU Memory (in GB):", torch.cuda.memory_reserved(0) / 1e9)

class ViLTransformerProbeSS(ViLTransformerSS):
    def __init__(self, config):
        super().__init__(config)

        # freeze existing model parameters
        self.freeze_params = []
        for n, p in self.named_parameters():
            p.requires_grad = False
            self.freeze_params.append(p)

        # construct probe networks
        out_dims = 0
        ctr = 0
        for m in self.sdr_head._head:
            if type(m) is nn.ConvTranspose2d:
                out_dims += m.out_channels
                ctr += 1
        self.probe_layer = nn.Linear(out_dims,1)
        self._interp = nn.functional.interpolate

        if config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            if config["load_from_ema"]:
                if "callbacks" in ckpt:
                    keys = list(ckpt["callbacks"].keys())
                    if "EMA" in keys:
                        state_dict = ckpt["callbacks"]["EMA"]["ema_state_dict"]
                    else:
                        raise ValueError("EMA checkpoints not found.")
            else:
                state_dict = ckpt["state_dict"]
            
            self.probe_layer.weight.data = state_dict["probe_layer.weight"]
            self.probe_layer.bias.data = state_dict["probe_layer.bias"]

        # generate random identifier (for caching)
        self._use_cache = True if config["loss_names"]["touchdown_sdr"] == 1  else False
        self._use_cache = False
        if self._use_cache:
            self._cache_path = "activation_cache/{}".format(str(uuid.uuid4())) 
            os.makedirs(self._cache_path, exist_ok = True)


    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
        ):  
            use_cache = self._use_cache
            # Set cache usage condition based on training state
            batch_size_key = "per_gpu_batchsize" if self.training else "per_gpu_batchsize_eval"
            use_cache &= self.hparams.config[batch_size_key] == 1

            if use_cache:
                identifier = batch["identifier"][0]
                cache_file = os.path.join(self._cache_path, f"{identifier}.pt")

                if os.path.exists(cache_file):
                    print("Loading from cache...")
                    ret = torch.load(cache_file, map_location=self.device)


            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            text_ids = batch[f"text_ids"]
            text_labels = batch[f"text_labels"]
            text_masks = batch["text_masks"]
            text_embeds = self.text_embeddings(text_ids)

            if image_embeds is None and image_masks is None:
                img = batch[imgkey][0]
                (
                    image_embeds,
                    image_masks,
                    patch_index,
                    image_labels,
                ) = self.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image,
                )
            else:
                patch_index, image_labels = (
                    None,
                    None,
                )
            text_embeds, image_embeds = (
                text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
                image_embeds
                + self.token_type_embeddings(
                    torch.full_like(image_masks, image_token_type_idx)
                ),
            )
            
            co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
            co_masks = torch.cat([text_masks, image_masks], dim=1)

            x = co_embeds

            for i, blk in enumerate(self.transformer.blocks):
                x, _attn = blk(x, mask=co_masks)


            x = self.transformer.norm(x)
            text_feats, image_feats = (
                x[:, : text_embeds.shape[1]],
                x[:, text_embeds.shape[1] :],
            )
            cls_feats = self.pooler(x)

            ret = {
                "text_feats": text_feats,
                "image_feats": image_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_labels": image_labels,
                "image_masks": image_masks,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "patch_index": patch_index,
            }

            if use_cache:
                torch.save(ret, cache_file)

            return ret

    def probe_head(self, image, text, output_size=(100, 464)):
        """Apply probe head operations on image and text tensors.

        Parameters:
        image (torch.Tensor): The image tensor.
        text (torch.Tensor): The text tensor.
        output_size (tuple): The desired output size of Segmentation logit map.

        Returns:
        torch.Tensor: Segmentation logit map
        """
        # Get the height and width of the image
        h, w = image.shape[2:]

        # Repeat the text tensor to match the image tensor dimensions
        tiled_text = text.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

        # Concatenate the image and tiled_text tensors
        feat = torch.cat([image, tiled_text], 1)

        # Determine if processing should be done phrase-by-phrase to save memory
        num_phrases = feat.shape[0]
        memory_save_mode = False
        slices = [[i] for i in range(num_phrases)] if memory_save_mode else [[i for i in range(num_phrases)]]

        # Initialize list to hold the processed tensors
        slice_out_logits = []

        # Process each slice
        for inds in slices:
            slice_feat = feat[inds, ...]

            # get feature maps from SDR intermediate activations
            slice_feats = []
            for m in self.sdr_head._head:
                slice_feat = m(slice_feat)
                if type(m) is nn.ConvTranspose2d:
                    slice_feats.append(slice_feat)
        
            # Tile and stack
            max_size = tuple(slice_feats[-1].shape[2:])
            slice_tiled_feat = [self._interp(sf, size=max_size, mode="bilinear", align_corners=False) for sf in slice_feats]
            cat_slice_feat = torch.cat(slice_tiled_feat, 1)

            # Apply probe layer and append to the list
            slice_out_logit = self.probe_layer(cat_slice_feat.permute(0, 2, 3, 1))
            slice_out_logits.append(slice_out_logit)

        # Concatenate all processed tensors
        out_logits = torch.cat(slice_out_logits)

        # Rescale the processed tensor to the original resolution
        # This if statement will be called both in Touchdown and Flickr30k Entities
        if output_size != max_size:
            out_logits = self._interp(out_logits.permute(0, 3, 1, 2), size=output_size, mode="bilinear", align_corners=False)
            out_logits = out_logits.squeeze(1).unsqueeze(-1)

        # Return the processed tensor
        return out_logits.squeeze(-1)

