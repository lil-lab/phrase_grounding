import os
import sqlite3
import torch
import numpy as np 
from datetime import datetime
from typing import List, Dict

import src.utils.dist_utils as dist
from src.visualization import DB_PATHES

def unravel_index(index, shape):
    '''
    source: https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/3
    '''
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

class DBVisualizer(object):
    def __init__(self, exp_name: str):
        super().__init__()
        self.reset(exp_name)

    def update(self, batch, raw_results):
        self.results["image_path"] += batch["picture_path"]
        self.results["task_groundtruth"] += batch["target_coords"].tolist()
        if "cls_coords" in raw_results or "sdr_coords" in raw_results:
            name = "cls_coords" if "cls_coords" in raw_results else "sdr_coords" 
            self.results["task_prediction"] += raw_results[name].tolist() 
        elif "cls_logpreds" in raw_results or "sdr_logpreds" in raw_results:
            name = "cls_logpreds" if "cls_logpreds" in raw_results else "sdr_logpreds" 
            device = raw_results[name].device
            max_inds = raw_results[name].flatten(-2).argmax(-1)
            max_inds = unravel_index(max_inds, raw_results[name].shape)
            max_inds = torch.stack(max_inds).transpose(0,1)
            max_inds = max_inds[:,1:]
            relative_max_inds = max_inds / torch.tensor(raw_results[name].shape[1:]).to(device)
            self.results["task_prediction"] += relative_max_inds[:,[1,0]].tolist()
            
            # save task prediction heatmap
            preds = torch.exp(raw_results[name])
            for i in range(len(batch["identifier"])):
                identifier = batch["identifier"][i]
                save_path = "{}/{}.npy".format(self.out_dir, identifier) 
                task_heatmap = preds[i, ...].cpu().numpy()
                np.save(open(save_path, "wb"), task_heatmap)
                self.results["task_heatmap_pathes"].append(save_path)

    def synchronize_between_processes(self):
        all_results = dist.all_gather(self.results)
        merged_results = {}
        for r in all_results:
            for k in r.keys():
                merged_results.setdefault(k, [])
                merged_results[k] += r[k]
        self.results = merged_results

    def reset(self, exp_name: str):
        cwd = os.getcwd()
        out_dir = "{}/src/visualization/heatmaps/{}".format(cwd, exp_name)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.results = {}
        self.fields = [
            "image_path",
            "task_prediction",
            "task_groundtruth",
            "task_heatmap_pathes"
        ]

        for f in self.fields:
            self.results.setdefault(f, [])


def update_db(task_evaluator, grounding_evaluator, complementary_evaluator, task_type: str, exp_name: str):
    # convert results to database format
    db_entries = summarize_results(task_evaluator, grounding_evaluator, complementary_evaluator, exp_name)

    # connect and write to  database
    db_path = DB_PATHES[task_type]
    db = sqlite3.connect(db_path)
    c = db.cursor()
    write_db_results(c, exp_name, db_entries)
    db.commit()


def summarize_results(task_evaluator, grounding_evaluator, complementary_evaluator, exp_name: str):
    task_results = task_evaluator.results
    task_accs = task_evaluator.game_accs
    task_scores = task_evaluator.game_scores
    grounding_results = grounding_evaluator.results
    grounding_metrics = grounding_evaluator.max_ious_dict
    complementary_results  = complementary_evaluator.results 
    identifiers = grounding_results["identifier"]
    
    db_entries = []
    is_heatmap = "grounding_heatmaps" in grounding_results
    if is_heatmap:
        cwd = os.getcwd()
        heatmap_outdir = "{}/src/visualization/heatmaps/{}".format(cwd, exp_name)

    # FIXME: skipping examples without grounding annotations
    fix_i = 0 

    for i in range(len(identifiers)):
        db_entry = {}

        # FIXME: skipping examples without grounding annotations
        while identifiers[i] != task_results["identifier"][fix_i]:
            fix_i += 1
        from ast import literal_eval
        # batch informations
        db_entry["identifier"] = grounding_results["identifier"][i]
        db_entry["caption"] = grounding_results["caption"][i]
        db_entry["split"] = grounding_results["split"][i]
        db_entry["image_path"] = str(list(complementary_results["image_path"][fix_i])) 

        # grounding 
        phrase_dict = {}
        phrase_ids, phrases = grounding_results["phrase_ids"][i], grounding_results["phrases"][i]
        for j, pid in enumerate(phrase_ids):
            phrase_dict[pid] = phrases[j]
        groundtruth_bboxes_dict = convert_to_dict_format([grounding_results["groundtruth_bboxes"][i]], [phrase_ids])
        db_entry["phrases"] = str(phrase_dict) 
        db_entry["grounding_bboxes_groundtruth"] = str(groundtruth_bboxes_dict)
        if is_heatmap:
            grounding_heatmap_pathes = {}
            grounding_heatmaps = convert_to_dict_format([grounding_results["grounding_heatmaps"][i]], [phrase_ids])
            for k in grounding_heatmaps.keys(): 
                save_path = "{}/{}.npy".format(heatmap_outdir, k) 
                np.save(open(save_path, "wb"), grounding_heatmaps[k])
                grounding_heatmap_pathes[k] = save_path
            db_entry["grounding_heatmap_pathes"] = str(grounding_heatmap_pathes)
        else:
            predicted_bboxes_dict = {}
            predicted_bboxes = convert_to_dict_format([grounding_results["predicted_bboxes"][i]], [phrase_ids])
            for pid in phrase_ids:
                predicted_bboxes_dict[pid] = predicted_bboxes[pid][0]
            db_entry["grounding_bboxes_prediction"] = str(predicted_bboxes_dict)

        # task 
        db_entry["task_prediction"] = str(complementary_results["task_prediction"][fix_i]) 
        db_entry["task_groundtruth"] = str(complementary_results["task_groundtruth"][fix_i]) 
        if is_heatmap:
            db_entry["task_heatmap_pathes"] = complementary_results["task_heatmap_pathes"][fix_i]

        # grounding metrics
        split = db_entry["split"] 
        met_keys = ["any", "merged", "all"]
        for met_key in met_keys:
            ious = {}
            for pid in phrase_ids:
                if met_key in grounding_metrics[split]:
                    ious[pid] = grounding_metrics[split][met_key][1][pid]
                else:
                    ious[pid] = -1
            db_entry["iou_{}".format(met_key)] = str(ious)

        # task metrics 
        identifier = db_entry["identifier"]
        db_entry["accuracy"] = str(task_accs[split][identifier])
        db_entry["score"] = str(task_scores[split][identifier])

        db_entries.append(db_entry)

    return db_entries


def convert_to_dict_format(output: List[List[any]], phraseids: List[List[str]]) -> Dict:
    phrase_to_output = {}
    for i in range(len(phraseids)):
        for j, pk in enumerate(phraseids[i]):
            phrase_to_output[pk] = output[i][j]
    return phrase_to_output

def get_current_time():
    # Stores the current time and date (including microseconds)
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


def write_db_modelList(c, exp_name: str):
    # initalize model list table if necessary
    q = (
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    c.execute(q)
    results = c.fetchall()
    if ("modelList", ) not in results:
        q = (
            "CREATE TABLE modelList (modelName STRING, timeStamp STRING)"
        )
        c.execute(q)

    # avoid duplicate
    q = 'SELECT * FROM modelList WHERE modelName=?'
    t = (exp_name, )
    c.execute(q, t)
    results = c.fetchall()

    if len(results) == 0:
        q = "INSERT INTO modelList (modelName, timeStamp) VALUES (?, ?)"
        t = (exp_name, get_current_time())
        c.execute(q, t)


def write_db_results(c, exp_name: str, results_list):
    exp_name = exp_name.replace("-", "_").replace("=", "_")

    # write the model name to the list keeping track of model names
    write_db_modelList(c, exp_name)

    # check if the table corresponding to the model name already exists
    # if no, make a new table for the model name
    q = (
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    c.execute(q)
    results = c.fetchall()
    if (exp_name, ) not in results:
        entires = "("
        keys = results_list[0].keys()
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                entires += "{} STRING)".format(k)
            else:
                entires += "{} STRING, ".format(k)
        q = (
            "CREATE TABLE {} {}".format(exp_name, entires)
        )
        c.execute(q)
    
    # updating the table corresponding to the model name
    entire_keys = "("
    for k in results_list[0].keys():
        entire_keys += "{}, ".format(k)
    entire_keys = entire_keys[:-2] 
    entire_keys += ")"
    queries = "("
    for k in results_list[0].keys():
        queries += "?, " 
    queries = queries[:-2] 
    queries += ")"

    for result in results_list:
        q = "INSERT INTO {} {} VALUES {}".format(exp_name, entire_keys, queries)
        t = []
        for k in result.keys():
            t.append(result[k])
        t = tuple(t)
        c.execute(q, t)

