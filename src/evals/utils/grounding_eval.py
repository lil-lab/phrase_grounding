# functions to evaluate phrase alginment prediction in Touchdown SDR task
import copy
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Union, Tuple, Sequence

import torch
from torch import nn
from pycocotools import mask as coco_mask



TOUCHDOWN_IMG_HEIGHT, TOUCHDOWN_IMG_WIDTH = 800, 3712
BBOX_THRESHHOLDS = [0.] + [np.round(0.01 * th_index, 3) for th_index in range(1,100)]

device = "cuda" if torch.cuda.is_available() else "cpu"
interpolation = nn.functional.interpolate
pooler = nn.functional.adaptive_avg_pool2d


# implementation from https://github.com/ashkamath/mdetr/blob/main/datasets/flickr_eval.py
class RecallTracker:
    """ Utility class to track recall@k for various k, split by categories"""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[int, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.
        report[k][cat] is the recall@k for the given category
        """
        report: Dict[int, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[k] = {
                cat: self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat] for cat in self.total_byk_bycat[k]
            }
        return report



def round_bboxes(boxes: np.array):
    boxes = np.round(boxes).astype(int)
    return boxes

def discretize_boxes(boxes: np.array, img_size: Tuple): 
    assert(len(img_size) == 2)
    target_h, target_w = img_size

    boxes[:, [1,3]] *= target_h
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, target_h)
    boxes[:, [0,2]] *= target_w
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, target_w)
    discretized_boxes = round_bboxes(boxes)

    return discretized_boxes

# implementation from https://github.com/ashkamath/mdetr/blob/main/datasets/flickr_eval.py
#### Bounding box utilities imported from torchvision and converted to numpy
def box_area(boxes: np.array) -> np.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


# implementation from https://github.com/ashkamath/mdetr/blob/main/datasets/flickr_eval.py
def box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


# implementation from https://github.com/ashkamath/mdetr/blob/main/datasets/flickr_eval.py
def _merge_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """
    Return the boxes corresponding to the smallest enclosing box containing all the provided boxes
    The boxes are expected in [x1, y1, x2, y2] format
    """
    if len(boxes) == 1:
        return boxes

    np_boxes = np.asarray(boxes)

    return [[np_boxes[:, 0].min(), np_boxes[:, 1].min(), np_boxes[:, 2].max(), np_boxes[:, 3].max()]]


def polygons_rel_to_abs(polygons: List, img_size: Tuple):
    polygons = copy.deepcopy(polygons)
    h, w = img_size
    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            if j % 2 == 0:
                polygons[i][j] = polygons[i][j]  * float(w)
            else:
                polygons[i][j]  = polygons[i][j]  * float(h)
    return polygons

# create a dictionary mapping from phrase ids to ground-truth bounding boxes of each phrase. 
def _process_target_bboxes(targets, merged = False, verbose: bool = True) -> Dict:
    processed_targets = {} # phraseid to bboxes coords 
    bbox_ct = 0

    for k in targets.keys():
        bbox_ct += len(targets[k])

    if merged:
        for k in targets.keys():
            processed_targets[k] = _merge_boxes(targets[k])
    else:
        processed_targets = targets
    
    if verbose:
        print("{} phrases with {} bounding boxes (before merging) evaluated.".format(len(processed_targets.keys()), bbox_ct))

    return processed_targets 

def bbox_to_heatmap(boxes: np.array, img_size: Tuple = (TOUCHDOWN_IMG_HEIGHT, TOUCHDOWN_IMG_WIDTH)) -> np.array:
    # detect relative coordinates system [0,1]
    # ! investigate why the rel coordianetes go anove 1.0 for the first place (sigmoid?)
    target_h, target_w = img_size
    float_type = np.issubdtype(boxes.dtype, np.floating) or (boxes.dtype == float)
    is_relative_coords = float_type and (np.max(boxes) <= (1. + 1e-6))
    if is_relative_coords:
        boxes = copy.deepcopy(boxes)
        for i, b in enumerate(boxes):
            boxes[i, [0,2]] *= target_w
            boxes[i, [1,3]] *= target_h
    
    # round coordinates to image coordinates system
    boxes = round_bboxes(boxes)
    assert(len(boxes.shape) == 2)
    heatmap = np.zeros((target_h, target_w))
    for b in boxes:
        x1, y1, x2, y2 = b
        heatmap[y1:y2, x1:x2] = 1

    return heatmap

def polygon_to_heatmap(polygons: np.array, img_size: Tuple = (TOUCHDOWN_IMG_HEIGHT, TOUCHDOWN_IMG_WIDTH)) -> np.array:
    polygons = polygons_rel_to_abs(polygons, img_size)
    rles = coco_mask.frPyObjects(list(polygons), img_size[0], img_size[1])
    heatmap = coco_mask.decode(rles)
    if len(heatmap.shape) < 3:
        heatmap = heatmap[..., None]
    heatmap = np.array(heatmap)
    heatmap = np.sum(heatmap, axis=2)     

    """
    import matplotlib.pyplot as plt
    plt.imshow(heatmap.numpy())
    plt.savefig("temp.png")
    """       

    return heatmap


def heatmap_iou(heatmap: np.array, boxes: np.array, heatmap_thresh: float, protocol: str, img_size : Tuple = (TOUCHDOWN_IMG_HEIGHT, TOUCHDOWN_IMG_WIDTH),use_segmentation_target: bool = False) -> np.array:
    """
    Return intersection-over-union (Jaccard index) between a heatmap and bboxes.

    A heatmap has a size of [100 464]

    Boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        heatmap (Matrix[N, 100 464])
        boxes (Matrix[M, 4])

    Returns:
        iou  (Matrix[NUM_THRESH, NUM_PREDS, NUM_TARGETS])
         - NUM_THRESH = 100 if heatmap_iou is None else 1
         - NUM_PREDS = N 
         - NUM_TARGETS = 1 if protocol is all else M
    """
    target_h, target_w = img_size

    # create binary target masks from boxes
    if protocol == "all":
        num_targets = 1
        if use_segmentation_target:
            bbox_masks = polygon_to_heatmap(boxes, img_size)
        else:
            bbox_masks = bbox_to_heatmap(boxes, img_size)
        bbox_masks = bbox_masks[np.newaxis, ...]
    else:
        if use_segmentation_target:
            raise ValueError
        num_targets = boxes.shape[0]
        bbox_masks = np.zeros((num_targets, target_h, target_w))
        for i, b in enumerate(boxes):
            bbox_masks[i, ...] = bbox_to_heatmap(boxes[i][np.newaxis,:], img_size)
    
    try:
        binary_targets = bbox_masks > 0.5
        assert(np.sum(binary_targets) != 0)
    except:
        print("polygon missing {}.".format(boxes))
    
    # calculate ious
    if device == "cuda":
        # GPU Implementation
        binary_targets = torch.tensor(binary_targets).to(device)
        binary_targets = binary_targets.unsqueeze(0)

        # process heatmap prediction
        #   resize heatmap
        #   normalize heatmap so that values ranging from 0 to 1
        heatmap = torch.tensor(heatmap).to(device)
        num_preds, h, w = heatmap.shape 

        if not(h == target_h and w == target_w):
            # use interpolation for upsampling, otherwise use pooler.
            if target_h >= h and target_w >= w:
                heatmap = interpolation(heatmap.unsqueeze(1), size=img_size, mode="bilinear", align_corners=False).squeeze(1)
            else:
                heatmap = pooler(heatmap.unsqueeze(1), output_size=img_size).squeeze(1)

        heatmap /= torch.amax(heatmap, dim=(1,2)).unsqueeze(-1).unsqueeze(-1)
        heatmap = heatmap.unsqueeze(1)

        assert(torch.sum(heatmap > 1) == 0 and torch.sum(heatmap < 0) == 0)

        # calculate iou matrix
        if heatmap_thresh is not None:
            binary_heatmap = heatmap > heatmap_thresh
            ious = torch.sum((binary_heatmap * binary_targets), (2,3)) / torch.sum((binary_heatmap + binary_targets), (2,3)) # [NUM_PREDS x NUM_TARGETS]
            iou_matrix = ious.unsqueeze(0)
            iou_matrix = iou_matrix.cpu().numpy()
        else:
            num_thresh = len(BBOX_THRESHHOLDS)
            iou_matrix = torch.zeros((num_thresh, num_preds, num_targets)).to(device)
            heatmap = torch.tensor(heatmap).to(device)
            for i, th in enumerate(BBOX_THRESHHOLDS):
                binary_heatmap = heatmap > th
                ious = torch.sum((binary_heatmap * binary_targets), (2,3)) / torch.sum((binary_heatmap + binary_targets), (2,3)) # [NUM_PREDS x NUM_TARGETS]
                iou_matrix[i, ...] = ious
            iou_matrix = iou_matrix.cpu().numpy()
    else:
        # CPU Implementation
        raise NotImplementedError("CPU version of the evaluation is not implemented.")

    return iou_matrix


# ! consider direcly loading the evaluataion data from the processed files not fed as an input
# evaluate sdr alignment prediction 
def eval_grounding(preds: Dict[str, List[any]], targets: Dict[str, List[List]], image_original_sizes: Dict[str, List[any]], pred_type: str, iou_thresh: float = 0.5, top_k = (1, 5, 10, -1), heatmap_threshes: Dict = {}, heatmap_thresh_criteria: str = 'recall', protocols = ["merged", "any", "all"], discretize_box: bool = True, use_heatmap: bool = False, verbose: bool = True, return_ious: bool = False, use_segmentation_target: bool = False) -> Dict:
    """
    Evaluate the alignment 

    A heatmap has a size of [100 464]

    Boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        preds: mapping from phrase ids to prediction 
            - bbox: Dict[str, Array[100, 4]]
            - heatmap: Dict[str, Matrix[100, 464]]
        targets: mapping from phrase ids to ground-truth bboxes 
            - bbox: Dict[str, Array[100, 4]]
        pred_type: "bbox" or "heatmap"
        iou_thresh
        top_k
        heatmap_threshes
        heatmap_thresh_criteria: what to optimize threshohold of heatmaps on "recall" or "iou"
        discretize_box: discreteize bboxes from continuous coordinates to pixels in the image. Make the box coordinates based evaluation function(box_iou) and the heatmap based evaluation function (heatmap_iou)  produce equivalent results.
        use_heatmap: force the evaluation to use the heatmap based evaluation function (heatmap_iou) 
        verbose
        return_ious

    Returns: 
        summaries: the summary (Dict) of recall and IOU scores
        heatmap_threshes: optimal heatmap threshhold for each evaluation protocol
        max_iou_results: max_iou of each example

    """
    summaries = {}
    top_k = (1, ) if pred_type == "heatmap" else top_k
    if use_segmentation_target:
        assert(len(protocols) == 1 and protocols[0] == "all")
    max_iou_results = {p: {k: {} for k in top_k} for p in protocols}

    for p in protocols:
        # process targets
        merged = True if p == "merged" else False
        recall_tracker = RecallTracker(top_k)

        # process targets (e.g., merging bounding boxes)
        processed_targets = _process_target_bboxes(targets, merged) 

        # process predictions and evaluate 
        if pred_type == "coords":
            for pk in tqdm(processed_targets.keys()):
                eval_heatmap = ((p == "all") and (len(processed_targets[pk]) != 1)) or use_heatmap 
                # evaluate iou
                img_orig_size = image_original_sizes[pk] if image_original_sizes is not None else (TOUCHDOWN_IMG_HEIGHT, TOUCHDOWN_IMG_WIDTH)
                if eval_heatmap:
                    largest_k = max(top_k)
                    heatmap = np.zeros((largest_k, img_orig_size[0], img_orig_size[1])) 
                    for idx in range(largest_k):
                        bbox = np.array(preds[pk][idx])[np.newaxis,:]
                        heatmap[idx, ...] = bbox_to_heatmap(bbox, img_size = img_orig_size) 
                    ious = heatmap_iou(heatmap, np.array(processed_targets[pk]), 0.5, p, img_size=img_orig_size, use_segmentation_target=use_segmentation_target)
                    if p == "all": 
                        ious = ious.squeeze(0).squeeze(-1)
                    else:
                        ious = ious.squeeze(0)
                else:
                    if discretize_box:
                        img_size = img_orig_size
                        disc_preds = discretize_boxes(np.array(preds[pk]), img_size)
                        disc_targets = discretize_boxes(np.array(processed_targets[pk]), img_size)
                        ious = box_iou(disc_preds, disc_targets)
                    else:
                        ious = box_iou(np.array(preds[pk]), np.array(processed_targets[pk]))

                for k in top_k:
                    # skipping when k == -1 (100) for heatmap evaluation (GPU memory overflows)
                    if (p == "all" or use_heatmap) and k == -1:
                        continue

                    maxi = 0
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= iou_thresh:
                        recall_tracker.add_positive(k, "total")
                    else:
                        recall_tracker.add_negative(k, "total")
                    max_iou_results[p][k][pk] = maxi

        elif pred_type == "heatmap":
            # heatmap threshhold. if not provides as an input we perform a grid search
            heatmap_thresh  = heatmap_threshes[p] if p in heatmap_threshes else None
            print("Protocol: {}, threshhold: {}".format(p, heatmap_thresh))

            max_iou_matrix = None # keeps track of max ious for each heatmap threshhold Matrix[NUM_THRESH, NUM_PHRASES]

            # evaluate 
            assert(len(preds) == len(processed_targets.keys()))
            for i, pk in enumerate(tqdm(processed_targets.keys())):
                img_orig_size = image_original_sizes[pk] if image_original_sizes is not None else (TOUCHDOWN_IMG_HEIGHT, TOUCHDOWN_IMG_WIDTH)
                ious = heatmap_iou(preds[pk][np.newaxis, ...], np.array(processed_targets[pk]), heatmap_thresh, p, img_size=img_orig_size, use_segmentation_target=use_segmentation_target)
                ious = ious.squeeze(1)
                maxi = ious.max(1)
                maxi = maxi[:, np.newaxis]
                max_iou_matrix = maxi if max_iou_matrix is None else np.concatenate([max_iou_matrix, maxi], axis=1)
            
            # get positive predictions and pick the optimal threshhold
            print("Choosing heatmap threshhold to optimize on {}.".format(heatmap_thresh_criteria))
            
            if heatmap_thresh_criteria == "recall":
                positive_matrix = (max_iou_matrix >= iou_thresh)  
                num_positives = np.sum(positive_matrix, 1)
                if heatmap_thresh is None:
                    best_th_index = num_positives.argmax()
                    best_th = BBOX_THRESHHOLDS[best_th_index]
                    heatmap_threshes[p] = best_th
                    max_ious = max_iou_matrix[best_th_index, :]
                else:
                    max_ious = max_iou_matrix[0, :]
            elif heatmap_thresh_criteria == "iou":
                mean_iou_matrix = np.mean(max_iou_matrix, 1)
                if heatmap_thresh is None:
                    best_th_index = mean_iou_matrix.argmax()
                    best_th = BBOX_THRESHHOLDS[best_th_index]
                    heatmap_threshes[p] = best_th
                    max_ious = max_iou_matrix[best_th_index, :]
                else:
                    max_ious = max_iou_matrix[0, :]
            else: 
                raise ValueError("Heatmap threshhold criteria {} is not supported.".format(heatmap_thresh_criteria))

            # update recall and keeps track of instance-level results
            for i, pk in enumerate(processed_targets.keys()):
                maxi = max_ious[i]
                if maxi >= iou_thresh:
                    recall_tracker.add_positive(1, "total")
                else:
                    recall_tracker.add_negative(1, "total")
                max_iou_results[p][1][pk] = maxi
        else:
            raise NotImplementedError("prediction type of {} is not supported".format(pred_type))

        # summarize recall / iou stats
        recall_summary = recall_tracker.report()
        for thk in recall_summary.keys():
            if (thk == -1 and (p == "all" or use_heatmap) and pred_type == "coords"):
                continue

            # update recall score
            metric = f"recall({p})@{thk}" if thk != -1 else f"recall({p})@100" 
            summaries[metric] = recall_summary[thk]["total"] 

            # update mean IOU score
            metric = f"mean IOU({p})@{thk}" if thk != -1 else f"mean IOU({p})@100"
            summaries[metric] = np.mean(list(max_iou_results[p][thk].values()))

    if verbose:
        message = ""
        # print out evaluation set-up
        message += f"pred_type: {pred_type}, discretize_box: {discretize_box}, use_heatmap: {use_heatmap} \n"

        # print out threshhold information
        if pred_type == "heatmap":
            message += "Threshholds  "
            for k in heatmap_threshes.keys():
                message += " {}:{} ".format(k, np.round(heatmap_threshes[k], 3))
            message += "\n"
        print(message)

    # return results
    if pred_type == "coords":
        return summaries if not return_ious else (summaries, max_iou_results)
    elif pred_type == "heatmap":
        return (summaries, heatmap_threshes) if not return_ious else (summaries, heatmap_threshes, max_iou_results)

