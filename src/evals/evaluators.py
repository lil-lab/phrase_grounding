import os, sys, pickle
import numpy as np
from ast import literal_eval
from collections import defaultdict
from typing import Dict, List

from .utils.sdr_eval import eval_sdr
from .utils.grounding_eval import eval_grounding
from .utils.correlation_eval import eval_correlation
import src.utils.dist_utils as dist

def convert_to_dict_format(output: List[List[any]], phraseids: List[List[str]]) -> Dict:
    phrase_to_output = {}
    for i in range(len(phraseids)):
        if phraseids[i] is not None:
            for j, pk in enumerate(phraseids[i]):
                phrase_to_output[pk] = output[i][j]
    return phrase_to_output


class TouchdownSDREvaluator(object):
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        if self.model_name != "":
            self.outdir = f"eval_outputs/{self.model_name}"
        os.makedirs(self.outdir, exist_ok=True)
        self.reset()

    def update(self, results):
        for k in self.fields:
            self.results[k] += results[k]

    def synchronize_between_processes(self):
        all_results = dist.all_gather(self.results)
        merged_results = {}
        for r in all_results:
            for k in r.keys():
                merged_results.setdefault(k, [])
                merged_results[k] += r[k]
        for k in merged_results.keys():
            merged_results[k] = np.array(merged_results[k])
        self.results = merged_results

    def summarize(self, eval_correlation: bool = False, sdr_thresh: int = 40):
        #! With multi-gpu, the dataloader duplicates examples if the number of examples is not divisible by the batch size.
        if dist.is_main_process():
            summaries = {}
            game_scores = {}
            game_accs = {}
            for sp in set(self.results["split"]):
                mask = (self.results["split"] == sp)
                summary = {"loss": np.mean(self.results["sdrloss"][mask])}
                # get sdr evaluation
                summary, distances_list = eval_sdr(self.results["prediction"][mask], self.results["groundtruth"][mask], self.results["captionid"][mask], self.results["propoagated"][mask], return_distances=True)
                distances_dict = {k: d for k, d in zip(self.results["identifier"][mask], distances_list)}
                summary.update(summary)
                summaries[sp] = summary
                game_scores[sp] = distances_dict
                acc_dict = {k: int(distances_dict[k] < sdr_thresh) for k in distances_dict.keys()}
                game_accs[sp] = acc_dict     

            self.game_scores = game_scores
            self.game_accs = game_accs
            pickle.dump(game_scores, open(os.path.join(self.outdir, "game_scores.pkl"), "wb"))
            pickle.dump(game_accs, open(os.path.join(self.outdir, "game_accs.pkl"), "wb"))

            return (summaries, game_scores) if eval_correlation else summaries
            
        return None

    def reset(self):        
        self.results = {}
        self.fields = [
            "prediction",
            "groundtruth",
            "captionid",
            "propoagated",
            "sdrloss",
            "identifier",
            "split"
        ]

        for f in self.fields:
            self.results.setdefault(f, [])


class F30kRefGameEvaluator(object):
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        if self.model_name != "":
            self.outdir = f"eval_outputs/{self.model_name}"
        os.makedirs(self.outdir, exist_ok=True)
        self.reset()

    def update(self, results, verbose: bool = False):
        for k in self.fields:
            self.results[k] += results[k]

    def synchronize_between_processes(self):
        all_results = dist.all_gather(self.results)
        merged_results = {}
        for r in all_results:
            for k in r.keys():
                merged_results.setdefault(k, [])
                merged_results[k] += r[k]
        for k in merged_results.keys():
            merged_results[k] = np.array(merged_results[k])
        self.results = merged_results

    def summarize(self, eval_correlation: bool = False):
        #! With multi-gpu, the dataloader duplicates examples if the number of examples is not divisible by the batch size.
        if dist.is_main_process():
            summaries = {}
            game_scores = {}
            game_accs = {}

            for sp in set(self.results["split"]):
                mask = (self.results["split"] == sp)
                summary = {
                    "num_examples": np.sum(mask),
                    "accuracy": np.mean(self.results["correct_t"][mask]),
                    "avg_loss": np.mean(self.results["loss"][mask]),
                }
                summary.update(summary)
                summaries[sp] = summary
                scores_dict = {k: d for k, d in zip(self.results["identifier"][mask], self.results["score"][mask])}
                game_scores[sp] = scores_dict
                accs = {k: d for k, d in zip(self.results["identifier"][mask], self.results["correct_t"][mask])}
                game_accs[sp] =  accs

            # save task metrics
            self.game_scores = game_scores
            self.game_accs = game_accs
            pickle.dump(game_scores, open(os.path.join(self.outdir, "game_scores.pkl"), "wb"))
            pickle.dump(game_accs, open(os.path.join(self.outdir, "game_accs.pkl"), "wb"))

            return (summaries, game_scores) if eval_correlation else summaries
            
        return None

    def reset(self):
        self.results = {}
        self.fields = [
            "identifier",
            "correct_t",
            "loss",
            "score",
            "split"
        ]
        for f in self.fields:
            self.results.setdefault(f, [])


class TangramGameEvaluator(F30kRefGameEvaluator):
    def __init__(self, model_name: str = ""):
        super().__init__(model_name=model_name)

    def summarize(self, eval_correlation: bool = False):
        if dist.is_main_process():
            summaries = {}
            game_scores = {}
            game_accs = {}
            for sp in set(self.results["split"]):
                mask = (self.results["split"] == sp)
                accs = {}
                for i, cor in enumerate(self.results["correct_t"][mask]):
                    tangram_id = self.results["tangram_id"][mask][i]
                    try:
                        tangram, annt_id = tangram_id.split("_")
                    except:
                        print("Warining: dataset format is incorrect.")
                        tangram, annt_id = "", ""
                    accs.setdefault(tangram, {})
                    accs[tangram].setdefault(annt_id, [])
                    accs[tangram][annt_id].append(cor)
                
                final_accs = []
                # 1. iterate thorugh tangrams
                for tg in accs.keys():
                    annt_acc = []
                    # 2. iterate thorugh annotations
                    for annt_id in accs[tg].keys():
                        # 3. iterate thorugh distractor sets
                        annt_acc.append(np.mean(accs[tg][annt_id]))
                    final_accs.append(np.mean(annt_acc)) 
                final_acc = np.mean(final_accs)
                summary = {
                    "num_examples": np.sum(mask),
                    "accuracy": final_acc,
                    "avg_loss": np.mean(self.results["loss"][mask]),
                }
                summary.update(summary)
                summaries[sp] = summary
                scores_dict = {k: d for k, d in zip(self.results["identifier"][mask], self.results["score"][mask])}
                game_scores[sp] = scores_dict
                accs_dict = {k: d for k, d in zip(self.results["identifier"][mask], self.results["correct_t"][mask])}
                game_accs[sp] =  accs_dict

            # save task metrics
            self.game_scores = game_scores
            self.game_accs = game_accs
            pickle.dump(game_scores, open(os.path.join(self.outdir, "game_scores.pkl"), "wb"))
            pickle.dump(game_accs, open(os.path.join(self.outdir, "game_accs.pkl"), "wb"))

            return (summaries, game_scores) if eval_correlation else summaries
            
        return None

    def reset(self):
        self.results = {}
        self.fields = [
            "identifier",
            "tangram_id",
            "correct_t",
            "loss",
            "score",
            "split"
        ]
        for f in self.fields:
            self.results.setdefault(f, [])


# FIXME: merge most of code with bbox evaluator
class GroundingHeatmapEvaluator(object):
    def __init__(self, model_name: str, dataset_name:str, eval_bbox_metric: bool = True, use_segmentation: bool = False):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.outdir = f"eval_outputs/{self.model_name}"
        os.makedirs(self.outdir, exist_ok=True)
        self.eval_bbox_metric = eval_bbox_metric
        self.use_segmentation = use_segmentation
        self.reset()

        # memory profiler
        import psutil 
        self._process = psutil.Process(os.getpid())

    def update(self, results, verbose: bool = False):
        for k in self.fields:
            self.results[k] += results[k]

        if verbose:
            util_stats = {}
            ram_usage = self._process.memory_info().rss / 1024 ** 2 
            vms_usage = self._process.memory_info().vms / 1024 ** 2
            util_stats["ram_usage"] = f"{ram_usage} MB"
            util_stats["vms_usage"] = f"{vms_usage} MB"
            print(util_stats)

    def synchronize_between_processes(self):
        all_results = dist.all_gather(self.results)
        merged_results = {}
        for r in all_results:
            for k in r.keys():
                merged_results.setdefault(k, [])
                merged_results[k] += r[k]
        for k in merged_results.keys():
            merged_results[k] = np.array(merged_results[k])
        self.results = merged_results

    def summarize(self, task_metric: Dict = None):
        if dist.is_main_process():
            summaries = {}
            max_ious_dict = {}    
            phraseid_to_imgid_dict = {}       
            heatmap_threshes = {}

            # order dataset split to decide which dataset split to use for picking up heatmap threshhold
            split_options = set(self.results["split"])
            if self.eval_bbox_metric:
                if self.dataset_name == "touchdown_sdr":
                    self.is_touchdown_sdr = True
                    if "tune" in split_options:
                        split_options = ["tune"] + list(split_options - set(["tune"]))
                elif self.dataset_name == "f30k_refgame":
                    self.is_touchdown_sdr = False
                    if "flickr_dev" in split_options:
                        split_options = ["flickr_dev"] + list(split_options - set(["flickr_dev"]))
                elif self.dataset_name == "tangram_refgame":
                    self.is_touchdown_sdr = False
                    if "tangram_val" in split_options:
                        split_options = ["tangram_val"] + list(split_options - set(["tangram_val"]))

            # evaluation loop
            for sp in split_options:
                mask = (self.results["split"] == sp)
                # evaluate phrase alignment prediction
                summaries.setdefault(sp + "_alignment", 
                    {
                        "grounding_loss": np.mean(self.results["grounding_loss"][mask]),
                        "num_examples (captions)": np.sum(mask)
                    }
                ) 

                if self.eval_bbox_metric:
                    # format results
                    formatted_heatmaps = convert_to_dict_format(self.results["grounding_heatmaps"][mask], self.results["phrase_ids"][mask])
                    formatted_groundtruth_bboxes = convert_to_dict_format(self.results["groundtruth_bboxes"][mask], self.results["phrase_ids"][mask])
                    formatted_image_original_sizes = None if self.is_touchdown_sdr else convert_to_dict_format(self.results["image_original_sizes"][mask], self.results["phrase_ids"][mask]) # evaluation heatmap resolution, Touchdown: (800, 3712), others: original image resolution

                    if self.use_segmentation:   
                        formatted_groundtruth_polygons = convert_to_dict_format(self.results["groundtruth_polygons"][mask], self.results["phrase_ids"][mask])
                        alignment_summary, heatmap_threshes, max_ious = eval_grounding(formatted_heatmaps, formatted_groundtruth_polygons, formatted_image_original_sizes, pred_type = "heatmap", protocols=["all"], heatmap_threshes=heatmap_threshes, heatmap_thresh_criteria="recall", return_ious=True, use_segmentation_target=self.use_segmentation)
                    else:
                        alignment_summary, heatmap_threshes, max_ious = eval_grounding(formatted_heatmaps, formatted_groundtruth_bboxes, formatted_image_original_sizes, pred_type = "heatmap", heatmap_threshes=heatmap_threshes, heatmap_thresh_criteria="recall", return_ious=True)
                    summaries[sp + "_alignment"].update(alignment_summary)

                    # analysis of correlation between SDR and alignment prediction
                    phraseid_to_imgid = {}
                    for i in range(len(self.results["phrase_ids"][mask])):
                        if self.results["phrase_ids"][mask][i] is not None:
                            for pk in self.results["phrase_ids"][mask][i]:
                                phraseid_to_imgid[pk] = self.results["identifier"][mask][i]
                    os.makedirs(os.path.join(self.outdir, sp), exist_ok=True)
                    eval_correlation(task_metric[sp] if task_metric is not None else None, max_ious, phraseid_to_imgid, os.path.join(self.outdir, sp))
                    max_ious_dict[sp] = max_ious
                    phraseid_to_imgid_dict[sp] = phraseid_to_imgid
            
            # save task metrics
            self.max_ious_dict = max_ious_dict
            self.phraseid_to_imgid_dict = phraseid_to_imgid_dict

            pickle.dump(max_ious_dict, open(os.path.join(self.outdir, "max_ious.pkl"), "wb"))
            pickle.dump(phraseid_to_imgid_dict, open(os.path.join(self.outdir, "phraseid_to_imgid.pkl"), "wb"))

            return summaries
        return None

    def reset(self):
        self.results = {}
        self.fields = [
            "caption",
            "grounding_loss",
            "identifier",
            "split"]
        
        if self.eval_bbox_metric:
            self.fields += [
                "phrases",
                "phrase_ids",
                "tokens_positive",
                "image_original_sizes",
                "groundtruth_bboxes",
                "grounding_heatmaps"
            ]

        if self.use_segmentation:    
            self.fields += [
                "groundtruth_polygons"
            ]

        for f in self.fields:
            self.results.setdefault(f, [])


class GroundingBoxEvaluator(object):
    def __init__(self, model_name: str, dataset_name:str, eval_bbox_metric: bool = True):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.outdir = f"eval_outputs/{self.model_name}"
        os.makedirs(self.outdir, exist_ok=True)
        self.eval_bbox_metric = eval_bbox_metric
        self.reset()

        # memory profiler
        import psutil 
        self._process = psutil.Process(os.getpid())

    def update(self, results, verbose: bool = False):
        for k in self.fields:
            self.results[k] += results[k]

        if verbose:
            util_stats = {}
            ram_usage = self._process.memory_info().rss / 1024 ** 2 
            vms_usage = self._process.memory_info().vms / 1024 ** 2
            util_stats["ram_usage"] = f"{ram_usage} MB"
            util_stats["vms_usage"] = f"{vms_usage} MB"
            print(util_stats)

    def synchronize_between_processes(self):
        all_results = dist.all_gather(self.results)
        merged_results = {}
        for r in all_results:
            for k in r.keys():
                merged_results.setdefault(k, [])
                merged_results[k] += r[k]
        for k in merged_results.keys():
            merged_results[k] = np.array(merged_results[k])
        self.results = merged_results

    def summarize(self, task_metric: Dict = None):
        if dist.is_main_process():
            summaries = {}
            max_ious_dict = {}
            phraseid_to_imgid_dict = {}
            
            # order dataset split to decide which dataset split to use for picking up heatmap threshhold
            split_options = set(self.results["split"])
            if self.eval_bbox_metric:
                if self.dataset_name == "touchdown_sdr":
                    self.is_touchdown_sdr = True
                    if "tune" in split_options:
                        split_options = ["tune"] + list(split_options - set(["tune"]))
                elif self.dataset_name == "f30k_refgame":
                    self.is_touchdown_sdr = False
                    if "flickr_dev" in split_options:
                        split_options = ["flickr_dev"] + list(split_options - set(["flickr_dev"]))
                elif self.dataset_name == "tangram_refgame":
                    self.is_touchdown_sdr = False
                    if "tangram_val" in split_options:
                        split_options = ["tangram_val"] + list(split_options - set(["tangram_val"]))

            # evaluation loop
            for sp in split_options:
                mask = (self.results["split"] == sp)
                # evaluate phrase alignment prediction
                summaries.setdefault(sp + "_alignment", 
                    {
                        "grounding_loss": np.mean(self.results["grounding_loss"][mask]),
                        "num_examples (captions)": np.sum(mask)
                    }
                ) 

                if self.eval_bbox_metric:
                    # format results
                    formatted_bbox_predictions = convert_to_dict_format(self.results["predicted_bboxes"][mask], self.results["phrase_ids"][mask])
                    formatted_bbox_groundtruthes = convert_to_dict_format(self.results["groundtruth_bboxes"][mask], self.results["phrase_ids"][mask])
                    formatted_image_original_sizes = None if self.is_touchdown_sdr else convert_to_dict_format(self.results["image_original_sizes"][mask], self.results["phrase_ids"][mask]) # evaluation heatmap resolution, Touchdown: (800, 3712), others: original image resolution
                    protocols = ["merged", "any", "all"] 
                    alignment_output = eval_grounding(formatted_bbox_predictions, formatted_bbox_groundtruthes, formatted_image_original_sizes, pred_type = "coords", protocols = protocols, return_ious = True)
                    if type(alignment_output) == tuple:
                        alignment_summary, max_ious = alignment_output
                    else:
                        alignment_summary = alignment_output
                    summaries[sp + "_alignment"].update(alignment_summary)

                    # analysis of correlation between SDR and alignment prediction
                    phraseid_to_imgid = {}
                    for i in range(len(self.results["phrase_ids"][mask])):
                        if self.results["phrase_ids"][mask][i] is not None:
                            for pk in self.results["phrase_ids"][mask][i]:
                                phraseid_to_imgid[pk] = self.results["identifier"][mask][i]
                    os.makedirs(os.path.join(self.outdir, sp), exist_ok=True)
                    eval_correlation(task_metric[sp] if task_metric is not None else None, max_ious, phraseid_to_imgid, os.path.join(self.outdir, sp))
                    max_ious_dict[sp] = max_ious
                    phraseid_to_imgid_dict[sp] = phraseid_to_imgid
        
            # save task metrics
            self.max_ious_dict = max_ious_dict
            self.phraseid_to_imgid_dict = phraseid_to_imgid_dict
            pickle.dump(max_ious_dict, open(os.path.join(self.outdir, "max_ious.pkl"), "wb"))
            pickle.dump(phraseid_to_imgid_dict, open(os.path.join(self.outdir, "phraseid_to_imgid.pkl"), "wb"))

            return summaries
        return None

    def reset(self):
        self.results = {}
        self.fields = [
            "caption",
            "grounding_loss",
            "identifier",
            "split"]
        
        if self.eval_bbox_metric:
            self.fields += [
                "phrases",
                "phrase_ids",
                "tokens_positive",
                "predicted_bboxes",
                "image_original_sizes",
                "groundtruth_bboxes"
            ]
        
        for f in self.fields:
            self.results.setdefault(f, [])
        

