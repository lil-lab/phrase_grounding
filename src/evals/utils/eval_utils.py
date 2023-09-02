import numpy as np
from typing import Dict, Tuple, List

import torch

BBOX_THRESHHOLDS = [0.] + [np.round(0.01 * th_index, 3) for th_index in range(1,100)]

def sdr_distance_metric(preds, targets, scale_factor: int = 8):
    """ SDR task
    Calculate distances between model predictions and targets within a batch.
    """
    scale_factor: int = 8
    distances = []
    for i in range(preds.shape[0]):
        pred, target = preds[i,...], targets[i,...]
        pred_coord = np.unravel_index(pred.argmax().item(), pred.size())
        target_coord = np.unravel_index(target.argmax().item(), target.size())
        dist = np.sqrt((target_coord[0] - pred_coord[0]) ** 2 + (target_coord[1] - pred_coord[1]) ** 2)
        distances.append(dist * scale_factor)
    return distances

def bounding_box_metrics(preds, targets):
    """
    Calculate bounding box ious
    """
    stats = {}

    # normalization (max value become 1)
    preds /= torch.max(preds)
    pos_targets = (targets != 0)
    
    # threshholding
    import time
    for th in BBOX_THRESHHOLDS:
        pos_preds = preds > th

        # calulate IOUs
        iou = torch.sum(pos_preds & pos_targets) / torch.sum(pos_preds | pos_targets)
        stats.update({
            "iou@{}".format(th): iou.item()
        })

        # calulate precisions
        prec = torch.sum(pos_preds & pos_targets) / torch.sum(pos_preds)
        stats.update({
            "prec@{}".format(th): prec.item()
        })

        # calculate recall
        recall = torch.sum(pos_preds & pos_targets) / torch.sum(pos_targets)
        stats.update({
            "recall@{}".format(th): recall.item()
        })
    return stats

def process_bounding_box_coords(orig_bbox: Dict, orig_img_height: int, orig_img_width: int, target_img_height: int, target_img_width: int, target_height_coverage = float) -> Tuple[Dict, bool]:
    """
    convert bounding box annotations coords between different image resolutions and coverages
    - the assumption is that the same FoV holds and the center of the camera is on the same direction

    orig_bbox = {
        "left": int[0,orig_img_width), 
        "top": int[0,orig_img_height)
        "width": int[0,orig_img_width)
        "height": int[0, orig_img_height)
    }
    """
    x, y = orig_bbox["left"], orig_bbox["top"]
    w, h = orig_bbox["width"], orig_bbox["height"]

    width_scale = target_img_width / orig_img_width
    height_scale = (target_img_height / target_height_coverage) / orig_img_height

    x2 = x * width_scale
    w2 = w * width_scale
    assert(x2 >= 0 and x2 <= (target_img_width - 1))
    assert(w > 0 and w <= target_img_width)
    slack = (orig_img_height * target_height_coverage) / 2
    y2 = (y - slack) * height_scale
    y2 = min(max(y2, 0), target_img_height-1)
    h_ = h * height_scale
    y_ =  (y - slack) * height_scale + h_  
    y_ = min(max(y_, 0), target_img_height-1)
    h2 = y_ - y2

    target_bbox = {
        "left": int(x2),
        "top": int(y2),
        "width": int(w2),
        "height": int(h2),
    }

    out_of_image = (int(w2) * int(h2) == 0) # if bounding box is completely out of the target image

    return target_bbox, out_of_image

def process_target_centers(orig_center: Dict, orig_img_height: int, orig_img_width: int, target_img_height: int, target_img_width: int, target_height_coverage = float) -> Tuple[Dict, bool]:
    """
    convert target centers between different image resolutions and coverages
    - the assumption is that the same FoV holds and the center of the camera is on the same direction

    orig_center = {
        "x": float[0,1), 
        "y": float[0,1)
    }
    """
    x, y = orig_center["x"], orig_center["y"]
    x *= orig_img_width
    y *= orig_img_height

    width_scale = target_img_width / orig_img_width
    height_scale = (target_img_height / target_height_coverage) / orig_img_height

    x2 = x * width_scale
    slack = (orig_img_height * target_height_coverage) / 2
    y2 = (y - slack) * height_scale
    y_ = min(max(y2, 0), target_img_height-1)
    out_of_image = (y2 != y_) # if bounding box is completely out of the target image

    x2 /= target_img_width
    y2 /= target_img_height
    new_center = {
        "x": x2,
        "y": y2,
    }

    return new_center, out_of_image

def unravel_index(index, shape):
    '''
    source: https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    '''
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def convert_heatmap_to_center(heatmap: np.array) -> List:
    '''
    convert heatmap representation to relative coordinateas of the largest mass in a heatmap
    heatmap ==> [x, y] or [col, row]
    '''
    coord = np.array(np.unravel_index(heatmap.argmax().item(), heatmap.shape))
    return np.array([coord[1] / heatmap.shape[1], coord[0] / heatmap.shape[0]])

@torch.no_grad()
def convert_heatmap_to_center_gpu(heatmap: torch.tensor) -> List:
    '''
    convert heatmap representation to relative coordinateas of the largest mass in a heatmap
    heatmap ==> [x, y] or [col, row]
    '''
    coord = np.array(unravel_index(heatmap.argmax().item(), heatmap.shape))
    return np.array([coord[1] / heatmap.shape[1], coord[0] / heatmap.shape[0]])
