# functions to evaluate SDR prediction in Touchdown SDR task
import numpy as np
from typing import List, Dict, Union, Tuple

DEBUG = True
if DEBUG:
    import os
    from IPython import embed

GROUNDTRUTH_IMG_WDITH, GROUNDTRUTH_IMG_HIEGHT = 3712, 800

# calculate distances of two absolute coordinates
def calc_distance(pred_coord: np.array, target_coord: np.array) -> np.array:
    dist = np.sqrt((target_coord[0] - pred_coord[0]) ** 2 + (target_coord[1] - pred_coord[1]) ** 2)
    return dist
    
# calculate distances between model predictions and targets from relative coordinates
def calc_dists_from_centers(pred_centers: np.array, target_centers: np.array):
    distances = []
    assert(pred_centers.shape == target_centers.shape)
    for i in range(pred_centers.shape[0]):
        pred_coord = [pred_centers[i,0] * GROUNDTRUTH_IMG_WDITH, pred_centers[i,1] * GROUNDTRUTH_IMG_HIEGHT]
        target_coord = [target_centers[i,0] * GROUNDTRUTH_IMG_WDITH, target_centers[i,1] * GROUNDTRUTH_IMG_HIEGHT]
        dist = calc_distance(pred_coord, target_coord)
        distances.append(dist)
    return np.array(distances)

# calculate distances between model predictions and targets from heatmaps of the same shape
def calc_dists_from_heatmaps(preds: np.array, targets: np.array, scale_factor: int = 8) -> np.array:
    assert(preds.shape == targets.shape)
    _, h, w = preds.shape
    height_scale_factor = GROUNDTRUTH_IMG_HIEGHT / h
    width_scale_factor = GROUNDTRUTH_IMG_WDITH / w
    assert(height_scale_factor == width_scale_factor)
    print(height_scale_factor, width_scale_factor)
    scale_factor = height_scale_factor

    distances = []
    for i in range(preds.shape[0]):
        pred, target = preds[i,...], targets[i,...]
        pred_coord = np.array(np.unravel_index(pred.argmax().item(), pred.shape))
        target_coord = np.array(np.unravel_index(target.argmax().item(), target.shape))
        dist = calc_distance(pred_coord, target_coord)
        distances.append(dist * scale_factor)
    return np.array(distances)

# calculate accuracy, consistency and mean distance
def summarize_results(distances: np.array, captionids: np.array, propagationinfo: np.array) -> Dict:
    results = {}
    # accuracies
    for th in [40, 80, 120]:
        met = f"accuracy@{th}px"
        results[met] = np.sum(distances < th) / len(distances)
        met = f"accuracy@{th}px (main)"
        results[met] = np.sum(distances[propagationinfo == False] < th) / len(distances[propagationinfo == False])
        met = f"accuracy@{th}px (propagated)"
        results[met] = np.sum(distances[propagationinfo == True] < th) / len(distances[propagationinfo == True])

    # consistencies
    for th in [40, 80, 120]:
        success = []
        for c in set(captionids):
            cap_dists = distances[captionids == c]
            success.append(np.all(cap_dists < th))
        met = f"consistency@{th}px"
        results[met] = np.sum(success) / len(set(captionids))

    # mean distances
    results["mean distance"] = np.mean(distances)
    results["mean distance (main)"] = np.mean(distances[propagationinfo == False] )
    results["mean distance (propagated)"] = np.mean(distances[propagationinfo == True])

    return results

# evaluate sdr prediction 
def eval_sdr(preds, targets, captionids: Union[np.array, List], propagationinfo: Union[np.array, List], verbose: bool = True, return_distances: bool = False) -> Union[Dict, Tuple[Dict, List]]:
    if type(propagationinfo) != np.array:
        propagationinfo = np.array(propagationinfo)
    if type(captionids) != np.array:
        captionids = np.array(captionids)

    if len(preds.shape) == 2:
        distances = calc_dists_from_centers(preds, targets)
    elif len(preds.shape) == 3:
        distances = calc_dists_from_heatmaps(preds, targets)
    summary = summarize_results(distances, captionids, propagationinfo)

    # record number of examples for the sanity check 
    num_examples, num_main_examples, num_propagated_examples = len(distances), np.sum(propagationinfo == False), np.sum(propagationinfo == True)
    summary["num_examples"] = num_examples
    summary["num_examples (main)"] = num_main_examples
    summary["num_examples (propagated)"] = num_propagated_examples
    if verbose:
        print(f"Evalauted {num_examples} examples ({num_main_examples} main and {num_propagated_examples} propagted examples)")

    return summary if not return_distances else (summary, distances)