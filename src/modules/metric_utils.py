# Third-party Libraries
import torch

# Local Modules
from src.evals import (
    Accuracy,
    Scalar,
    PostProcessTouchdownSDR,
    PostProcessViLTGrounding,
    PostProcessMDETRGrounding,
    TouchdownSDREvaluator,
    F30kRefGameEvaluator,
    TangramGameEvaluator,
    GroundingHeatmapEvaluator,
    GroundingBoxEvaluator,
    DBVisualizer
)
import src.utils.dist_utils as dist


def get_memory_stats(pl_module):
    memory_stats = {}
    memory_stats["ram_usage"] = pl_module.process.memory_info().rss / 1024 ** 2
    memory_stats["vms_usage"] = pl_module.process.memory_info().vms / 1024 ** 2
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    memory_stats["gpu:0_reserved"] = r / 1024 ** 2
    memory_stats["gpu:0_used"] = a / 1024 ** 2
    memory_stats["gpu:0_free"] = (r-a) / 1024 ** 2
    return memory_stats

def set_metrics(pl_module):
    pl_module.all_metircs = []
    pl_module.postprocessors = []
    pl_module.evaluators = []
    exp_name = pl_module.hparams.config["exp_name"]
    test_only = pl_module.hparams.config["test_only"]

    for split in ["train", "tune", "val"]:
        for loss_name, v in pl_module.hparams.config["loss_names"].items():
            if v > 0:
                metric_name = f"{split}_{loss_name}_loss"
                setattr(pl_module, metric_name, Scalar())
                pl_module.all_metircs.append(metric_name)
                metric_name = f"{split}_{loss_name}_accuracy"
                setattr(pl_module, metric_name, Scalar())
                pl_module.all_metircs.append(metric_name)

    # Touchdown SDR
    if pl_module.hparams.config["datasets"][0] in ["touchdown_sdr"]:
        pl_module.postprocessors.append(PostProcessTouchdownSDR())
        pl_module.evaluators.append(TouchdownSDREvaluator(exp_name))
        for split in ["train", "tune", "val"]:
            setattr(pl_module, f"{split}_touchdown_sdr_dist", Scalar())
            pl_module.all_metircs.append(metric_name)

    # reference games
    if pl_module.hparams.config["datasets"][0] == "f30k_refgame":
        pl_module.evaluators.append(F30kRefGameEvaluator(exp_name))

    if pl_module.hparams.config["datasets"][0] == "tangram_refgame":
        pl_module.evaluators.append(TangramGameEvaluator(exp_name))

    # grounding
    if pl_module.hparams.config["use_grounding"]:
        if  pl_module.hparams.config["model_cls"] == "vilt_aligner" or pl_module.hparams.config["model_cls"] == "vilt_aligner_probe":
            pl_module.postprocessors.append(PostProcessViLTGrounding( eval_bbox_metric=test_only, use_segmentation=pl_module.hparams.config["tangram_options"]["use_segmentation_mask"]))
            pl_module.evaluators.append(GroundingHeatmapEvaluator(exp_name, dataset_name=pl_module.hparams.config["datasets"][0], eval_bbox_metric=test_only, use_segmentation=pl_module.hparams.config["tangram_options"]["use_segmentation_mask"]))
        elif pl_module.hparams.config["model_cls"] == "mdetr":
            pl_module.postprocessors.append(PostProcessMDETRGrounding( eval_bbox_metric=test_only)) 
            pl_module.evaluators.append(GroundingBoxEvaluator(exp_name, dataset_name=pl_module.hparams.config["datasets"][0],  eval_bbox_metric=test_only))  
    
    if pl_module.hparams.config["test_update_db"]:
        pl_module.evaluators.append(DBVisualizer(exp_name))  

def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0
    wandb_stats = {}    
    
    for m in pl_module.all_metircs:
        value = getattr(pl_module, m).compute()
        getattr(pl_module, m).reset()
        phase, met_name = m.split("_")[0], "_".join(m.split("_")[1:])
        pl_module.log(f"{met_name}/{phase}", value)
        wandb_stats[f"{phase}/{met_name}"] = value
        if "_loss" in m:
            the_metric += value

    value = pl_module.checkpoint_metric if phase == "vale" else value
    pl_module.log(f"{phase}/the_metric", the_metric)

def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return
