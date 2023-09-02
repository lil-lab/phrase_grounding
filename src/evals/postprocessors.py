import os, sys
import numpy as np
from typing import Dict

from torch import nn
import torch
import torch.nn.functional as F

from .utils.eval_utils import convert_heatmap_to_center, convert_heatmap_to_center_gpu

# ! code duplicate with mdetr_utils.py
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class PostProcessTouchdownSDR(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, batch):
        # convert heatmap to relative coordinates
        if "sdr_logpreds" in outputs:
            # ! this is likely the line where memory leak is happening?
            # ! potentially workaround is find the best coordinate in GPU space.
            sdr_centers_pred_list = [list(convert_heatmap_to_center_gpu(outputs["sdr_logpreds"][i, ...])) for i in range(outputs["sdr_logpreds"].shape[0])]
        elif "sdr_coords" in outputs:
            sdr_centers_pred_list = outputs["sdr_coords"].cpu().tolist()

        if "sdr_coords" in outputs:
            # coordinates 
            sdr_centers_gt_list = batch["target_coords"].cpu().tolist()
        else:
            # heatmap 
            sdr_centers_gt_list = [list(convert_heatmap_to_center_gpu(batch["target"][i, ...])) for i in range(batch["target"].shape[0])]

        batch_size = len(sdr_centers_gt_list)
        identifier_list = [batch["identifier"][i] for i in range(batch_size)]
        captionid_list = [batch["caption_id"][i] for i in range(batch_size)]
        propagated_list = [batch["is_propagated"][i] for i in range(batch_size)]
        sdrloss_list = [outputs["sdr_losses"][i].item() for i in range(batch_size)]
        processed_results = {
            "prediction": sdr_centers_pred_list,
            "groundtruth": sdr_centers_gt_list,
            "captionid": captionid_list,
            "identifier": identifier_list,
            "propoagated": propagated_list,
            "sdrloss": sdrloss_list,
            "split": batch["split"]
        }
        return processed_results


class PostProcessGrounding(nn.Module):
    def __init__(self, model_cls: str, eval_bbox_metric: bool = True, use_segmentation: bool = False):
        super().__init__()
        self.model_cls = model_cls
        self.eval_bbox_metric = eval_bbox_metric
        self.use_segmentation = use_segmentation
        
    @torch.no_grad()
    def forward(self, outputs, batch):        
        processed_results = {}
        batch_size = len(batch["identifier"])

        # identifiers  
        identifier_list = [batch["identifier"][i] for i in range(batch_size)]
        processed_results["identifier"] = identifier_list

        # batch split
        processed_results["split"] = batch["split"]

        # losses
        grounding_loss_list = [outputs["grounding_losses"][i].item()  for i in range(batch_size)]
        processed_results["grounding_loss"] = grounding_loss_list

        # captions
        caption_list = [batch["text"][i] for i in range(batch_size)]
        processed_results["caption"] = caption_list

        if self.eval_bbox_metric:
            # phrases
            tokens_positive_list = []
            for i in range(batch_size):  
                tokens_positive = []
                if batch["bbox_token_postives"][i] is not None:
                    for tkp in batch["bbox_token_postives"][i]:
                        tokens_positive.append(tkp)
                tokens_positive_list.append(tokens_positive)
            processed_results["tokens_positive"] = tokens_positive_list

            phrases_list = []
            for i in range(batch_size):
                text = caption_list[i]
                phrases = []
                for tkps in tokens_positive_list[i]:
                    tokens = []
                    for tkp in tkps:
                        tokens.append(text[tkp[0]:tkp[1]])
                    phrases.append(" ".join(tokens))
                phrases_list.append(phrases)
            processed_results["phrases"] = phrases_list

            # ground-truth bounding-boxes and phrase ids
            phrase_bboxes, phrase_ids = [], []
            if self.use_segmentation:
                phrase_polygons = []
                
            for i in range(batch_size):
                if len(batch["bbox_coords"][i]) != 0 and len(batch["bbox_coords"][i][0]) != 0:
                    if self.model_cls == "vilt_aligner":
                        # convert each bbox from (xleft, ytop, w, h)  to (x1, y1, x2, y2) format
                        bboxes = [b.tolist() for b in batch["bbox_coords"][i]] 
                        for j in range(len(bboxes)):
                            for k in range(len(bboxes[j])):
                                    # for ViLT_aligner image transformation
                                    bboxes[j][k][2] += bboxes[j][k][0]
                                    bboxes[j][k][3] += bboxes[j][k][1]
                    elif self.model_cls == "mdetr":
                        # for MDETR image transformation
                        bboxes = [torch.clip(box_cxcywh_to_xyxy(b), 0, 1).tolist() for b in batch["bbox_coords"][i]] 
                    phrase_bboxes.append(bboxes)
                    phrase_id = batch["phrase_ids"][i]
                    identifier_phrase_id = ["{}_{}".format(identifier_list[i], pid) for pid in phrase_id]
                    phrase_ids.append(identifier_phrase_id)
                    if self.use_segmentation:
                        if self.model_cls == "vilt_aligner":
                            polygons = [p for p in batch["polygon_coords"][i]] 
                            phrase_polygons.append(polygons)
                        elif self.model_cls == "mdetr":
                            raise NotImplementedError
                else:
                    phrase_bboxes.append([])
                    phrase_ids.append([])
                    if self.use_segmentation:
                        phrase_polygons.append([])

            processed_results["groundtruth_bboxes"] = phrase_bboxes
            if self.use_segmentation:
                processed_results["groundtruth_polygons"] = phrase_polygons

            # concatenate phrase_ids to identifiers
            # this is necessary to gurantee the uniquness for Tangram evaluation
            processed_results["phrase_ids"] = phrase_ids

            # original image sizes
            image_original_sizes = []
            for i in range(batch_size):
                image_original_sizes.append([
                      batch["picture_orig_size"][i] if "picture_orig_size" in batch else (800, 3712) for j in range(len(phrases_list[i])) ]) 
            processed_results["image_original_sizes"] = image_original_sizes
        else:
            for k in ["tokens_positive", "phrases", "image_original_sizes", "phrase_ids"]:
                processed_results[k] = [[] for _ in range(batch_size)]

        return processed_results


class PostProcessViLTGrounding(PostProcessGrounding):
    def __init__(self, eval_bbox_metric: bool = True, use_segmentation: bool = False):
        super().__init__(model_cls="vilt_aligner", eval_bbox_metric=eval_bbox_metric, use_segmentation=use_segmentation)
        
    @torch.no_grad()
    def forward(self, outputs, batch):
        processed_results = {}
        processed_results.update(super().forward(outputs, batch))

        # phrase and alignments
        if self.eval_bbox_metric and (outputs["grounding_log_pred"] is not None):     
            batch_size = len(batch["identifier"])
            phrases_list = processed_results["phrase_ids"]

            # phrase alignment heatmaps
            phrase_prediction = torch.exp(outputs["grounding_log_pred"])
            phrase_prediction = phrase_prediction.cpu().numpy()

            offset = 0
            phrase_heatmaps = []
            for i in range(batch_size):
                heatmaps = []
                if "bbox_tensors_orig_sizes" in batch and batch["bbox_tensors_orig_sizes"][i] is not None:
                    h_end, w_end = batch["bbox_tensors_orig_sizes"][i]
                for j in range(len(phrases_list[i])):
                    if "bbox_tensors_orig_sizes" in batch and batch["bbox_tensors_orig_sizes"][i] is not None:
                        heatmap = phrase_prediction[offset+j, :h_end, :w_end]
                    else:
                        heatmap = phrase_prediction[offset+j, ...]
                    heatmaps.append(heatmap)
                phrase_heatmaps.append(heatmaps)
                offset += len(phrases_list[i])
            processed_results["grounding_heatmaps"] = phrase_heatmaps
        
            # assert
            for k in processed_results.keys():
                assert(len(processed_results[k]) == batch_size)
        else:
            batch_size = len(batch["identifier"])
            processed_results["groundtruth_bboxes"] = [[] for _ in range(batch_size)]
            processed_results["groundtruth_polygons"] = [[] for _ in range(batch_size)]
            processed_results["grounding_heatmaps"] = [[] for _ in range(batch_size)]

        return processed_results


class PostProcessMDETRGrounding(PostProcessGrounding):
    """This module converts the model's output for Flickr30k entities evaluation.

    This processor is intended for recall@k evaluation with respect to each phrase in the sentence.
    It requires a description of each phrase (as a binary mask), and returns a sorted list of boxes for each phrase.
    """
    def __init__(self, eval_bbox_metric: bool = True):
        super().__init__(model_cls="mdetr", eval_bbox_metric=eval_bbox_metric)

    @torch.no_grad()
    def forward(self, outputs, batch):
        """Perform the computation.
        Args:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            positive_map: tensor [total_nbr_phrases x max_seq_len] for each phrase in the batch, contains a binary
                          mask of the tokens that correspond to that sentence. Note that this is a "collapsed" batch,
                          meaning that all the phrases of all the batch elements are stored sequentially.
            items_per_batch_element: list[int] number of phrases corresponding to each batch element.
        """
        processed_results = {}
        processed_results.update(super().forward(outputs, batch))

        out_logits, out_bbox = outputs["grounding_output"]["pred_logits"], outputs["grounding_output"]["pred_boxes"]
        target_sizes = torch.tensor(batch["picture_orig_size"]).to(out_logits.device)
        positive_map =  [m for m in batch["bbox_positive_maps"] if m is not None]
        batch_size = target_sizes.shape[0]

        if len(positive_map) == 0:
            print("validation no positive map.")
            processed_results["predicted_bboxes"] = [[] for _ in range(batch_size)]
            return processed_results

        positive_map = torch.cat(positive_map)
        items_per_batch_element = [len(m) if m is not None else 0 for m in batch["bbox_positive_maps"]]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2


        prob = F.softmax(out_logits, -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.clip(boxes, 0,1) # we make sure boxes fit within images for Touchdown SDR evaluation

        cum_sum = np.cumsum(items_per_batch_element)

        curr_batch_index = 0
        # binarize the map if not already binary
        pos = positive_map > 1e-6

        predicted_boxes = [[] for _ in range(batch_size)]

        # The collapsed batch dimension must match the number of items
        assert len(pos) == cum_sum[-1]

        if len(pos) == 0:
            return predicted_boxes

        # if the first batch elements don't contain elements, skip them.
        while cum_sum[curr_batch_index] == 0:
            curr_batch_index += 1

        for i in range(len(pos)):
            # scores are computed by taking the max over the scores assigned to the positive tokens
            scores, _ = torch.max(pos[i].unsqueeze(0) * prob[curr_batch_index, :, :], dim=-1)
            _, indices = torch.sort(scores, descending=True)

            assert items_per_batch_element[curr_batch_index] > 0
            predicted_boxes[curr_batch_index].append(boxes[curr_batch_index][indices].to("cpu").tolist())
            if i == len(pos) - 1:
                break

            # check if we need to move to the next batch element
            while i >= cum_sum[curr_batch_index] - 1:
                curr_batch_index += 1
                assert curr_batch_index < len(cum_sum)

        processed_results["predicted_bboxes"] = predicted_boxes

        return processed_results


