# functions analyze model behaivors in Touchdown SDR task
import os
import pickle
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict

# correlation analysis and visualization of SDR prediction and phrase prediction
def eval_correlation(task_scores: Dict, max_ious: Dict, phraseid_to_imgid: Dict, outdir: str = ".debug", visualize: bool = True, task_thresh: int = 80, iou_thresh: float = 0.5, eps=1):
    os.makedirs(outdir, exist_ok=True)

    # save results in pkls
    pickle.dump(
        {
            "task_scores": task_scores, 
            "max_ious": max_ious, 
            "phraseid_to_imgid": phraseid_to_imgid, 
        },
    open(os.path.join(outdir, "results.pkl"), "wb"))

    if (task_scores is None) or  (max_ious is None):
        print("Either SDR task_scores or max_ious are missing. Skipping correlation analysis.")
        return

    # process iou / recall alignment scores
    x_dicts, x_labels = [], []
    for prt in max_ious.keys():
        for topk in max_ious[prt].keys():
            img_to_recalls = {}
            img_to_ious = {}
            if len(max_ious[prt][topk]) == 0:
                continue
            else:
                for p_id in max_ious[prt][topk]:
                    iou = max_ious[prt][topk][p_id]
                    img_id = phraseid_to_imgid[p_id]
                    img_to_ious.setdefault(img_id, {})
                    img_to_ious[img_id][p_id] = iou
                    img_to_recalls.setdefault(img_id, {})
                    img_to_recalls[img_id][p_id] = (iou >= iou_thresh)

                x_label = f"IOU@{topk} ({prt})" if topk != -1 else f"IOU@all ({prt})"
                x_dicts.append(img_to_ious)
                x_labels.append(x_label)
                x_label = f"recall@{topk} ({prt})" if topk != -1 else f"recall@all ({prt})"
                x_dicts.append(img_to_recalls)
                x_labels.append(x_label)

    # 1. scattered plot (alignment scores vs log dist)
    y_dicts, y_labels = [], []

    img_to_logdits = {}
    for k in task_scores.keys():
        img_to_logdits[k] = np.log(task_scores[k] + eps)
    y_dicts.append(img_to_logdits)
    y_labels.append("log-dist (eps={})".format(eps))
    
    scattered_outpath = os.path.join(outdir, "scattered.png")
    draw_plots(x_dicts, y_dicts, x_labels, y_labels, "scattered", scattered_outpath)


    # 2. bar plot (binned alignment scores vs success@X)
    y_dicts, y_labels = [], []

    img_to_success = {}
    for k in task_scores.keys():
        img_to_success[k] = int(task_scores[k] < task_thresh)
    y_dicts.append(img_to_success)
    y_labels.append(f"success@{task_thresh}px")

    bar_outpath = os.path.join(outdir, "bar.png")
    draw_plots(x_dicts, y_dicts, x_labels, y_labels, "bar", bar_outpath)


# draw graphs
def draw_plots(x_dicts: List[Dict], y_dicts: List[Dict], x_labels: List, y_labels: List, mode: str = "scattered", outpath: str = "", verbose: bool = True):
    assert(len(x_dicts) == len(y_dicts) or len(y_dicts) == 1)
    assert(len(x_labels) == len(y_labels) or len(y_labels) == 1)

    num_plots = len(x_dicts)
    num_rows = 1 if num_plots == 1 else int(np.ceil(num_plots / 2))
    num_cols = 1 if num_plots == 1 else 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows), dpi=160)
    for i in range(num_plots): 
        if num_rows == 1:
            ax = axs[i]
        else:
            ax = axs[int(np.floor(i / 2)), i % 2]
        y_ind = i if len(x_dicts) == len(y_dicts) else 0
        if mode == "scattered":
            draw_scattered_plot(x_dicts[i], y_dicts[y_ind], x_labels[i], y_labels[y_ind], ax)
        elif mode == "bar":
            draw_bar_plot(x_dicts[i], y_dicts[y_ind], x_labels[i], y_labels[y_ind], ax)
        else:
            raise NotImplementedError(f"Graph mode {mode} is not supproted.")


    plt.subplots_adjust(hspace=0.5)
    plt.savefig(outpath)
    plt.close()

    if verbose:
        print(f"{mode} plot is saved under {outpath}")


# draw a scattered plot
def draw_scattered_plot(x_dict: Dict, y_dict: Dict, x_label: str, y_label: str, ax):
    x, y = [], []
    for k in x_dict.keys():
        if len(x_dict[k].values()) == 0:
            continue
        x.append(np.mean(list(x_dict[k].values())))
        y.append(y_dict[k])

    ax.scatter(x, y, s=2, marker='x')
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_xlabel(x_label, fontsize=8)
    pearson = scipy.stats.pearsonr(x, y)[0] # Pearson's r
    ax.set_title(r'$Pearson=%.3f$' % (pearson), fontsize=8)
    #ax.axhline(y=80, c="r")


# draw a bar plot
def draw_bar_plot(x_dict: Dict, y_dict: Dict, x_label: str, y_label: str, ax, num_bins=40):
    x, y = [], []
    for k in x_dict.keys():
        if len(x_dict[k].values()) == 0:
            continue
        x.append(np.mean(list(x_dict[k].values())))
        y.append(y_dict[k])
    x, y = np.array(x), np.array(y)

    step = (np.max(x) + 1e-6) / num_bins
    bins = []
    for i in range(num_bins):
        bins.append(0 + (i + 1) * step)
    #bins = [0 + (i + 1) * step for i in range(num_bins)]
    inds = np.digitize(x, bins)
    names, values = [], []
    for ind, b in enumerate(bins):
        mask = (inds == ind)
        val = 0 if np.sum(mask) == 0 else np.mean(y[mask]) 
        name = "{} - {}".format(0, np.round(b, 3)) if ind == 0 else "{} - {}".format(np.round(bins[ind-1], 3), np.round(b, 3))
        names.append(name)
        values.append(val)

    # reference: https://stackoverflow.com/questions/47838680/matplotlib-xticks-values-in-bar-chart/47893553
    idxs = np.asarray([i for i in range(len(names))])
    ax.bar(idxs, values)
    ax.set_xticks(idxs)
    ax.set_xticklabels(names, rotation = 60, fontsize=5)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
