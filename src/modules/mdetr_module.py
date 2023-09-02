# Standard Libraries
import os
import copy
import time
import pickle
import pprint

# Third-party Libraries
import psutil
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Local Modules
from src.utils import objectives, dist_utils
from src.modules import heads, metric_utils, mdetr_utils
from src.evals import (
    TouchdownSDREvaluator,
    F30kRefGameEvaluator,
    TangramGameEvaluator,
    GroundingBoxEvaluator,
    DBVisualizer
)
import src.utils.dist_utils as dist

class MDETR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # intialize MDETR
        assert(self.hparams.config["contrastive_align_loss"])
        self.mdetr, self.postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_resnet101', pretrained=True, return_postprocessor=True)

        self.mdetr.aux_loss = self.hparams.config["aux_loss"]
        self.mdetr.contrastive_align_loss = self.hparams.config["contrastive_align_loss"]
        if self.mdetr.contrastive_align_loss:
            hidden_dim = self.mdetr.transformer.d_model
            contrastive_hdim = 64
            self.mdetr.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.mdetr.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

        # intialize downstream heads
        self.prepare_sdr = (self.hparams.config["loss_names"]["touchdown_sdr"] > 0) or self.hparams.config["use_touchdown_loss_for_cls"]
        if self.prepare_sdr:
            nb_heads, hidden_dim = 1 , self.mdetr.bbox_embed.layers[0].in_features
            self.sdr_embed = nn.Embedding(nb_heads, hidden_dim)
            if self.hparams.config["task_head_setup"]["use_shared_head"]: 
                self.sdr_head = self.mdetr.bbox_embed
            else:
                self.sdr_head = copy.deepcopy(self.mdetr.bbox_embed)
                if not self.hparams.config["mdetr_weights_init"]["initialize_sdr_head_from_bbox_embed"]: 
                    self.sdr_head.apply(mdetr_utils.init_weights)

        self.prepare_refgame = (self.hparams.config["loss_names"]["refgame"] > 0) and (not self.hparams.config["use_touchdown_loss_for_cls"])
        if self.prepare_refgame:
            nb_heads, hidden_dim = 1 , self.mdetr.bbox_embed.layers[0].in_features
            self.class_embed = nn.Embedding(nb_heads, hidden_dim)
            self.contrastive_head = heads.ContrastiveHead(hidden_size=hidden_dim, num_distractors=config["refgame_options"]["num_distractors"])
            self.contrastive_head.apply(mdetr_utils.init_weights)

        # intializing loss 
        losses = ["labels", "boxes"]
        if self.hparams.config["contrastive_align_loss"]:
            losses += ["contrastive_align"]
        self.criterion = mdetr_utils.SetCriterion(
            num_classes=255,
            matcher=mdetr_utils.HungarianMatcher(
                cost_class = self.hparams.config["cost_class"], 
                cost_bbox = self.hparams.config["cost_bbox"], 
                cost_giou = self.hparams.config["cost_giou"],
            ),
            eos_coef=self.hparams.config["eos_coef"], 
            losses=losses, 
            temperature=self.hparams.config["temperature_NCE"]
        )
        mdetr_utils.setup_weight_dict(self)

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
                keys = list(ckpt["callbacks"].keys())
                if "EMA" in keys:
                    state_dict = ckpt["callbacks"]["EMA"]["ema_state_dict"]
                else:
                    raise ValueError("EMA checkpoints not found.")
            else:
                state_dict = ckpt["state_dict"]

            if state_dict is not None:
                self.load_state_dict(state_dict, strict=False)
            else:
                raise NotImplementedError


    def infer(self, batch, mask_text=False, mask_image=False):
        samples = batch["image"][0]
        captions = batch["text"]

        if not isinstance(samples, mdetr_utils.NestedTensor):
            samples = mdetr_utils.NestedTensor.from_tensor_list(samples)

        features, pos = self.mdetr.backbone(samples)
        src, mask = features[-1].decompose()
        query_embed = self.mdetr.query_embed.weight
        if self.prepare_sdr:
            query_embed = torch.cat([query_embed, self.sdr_embed.weight], 0)
        elif self.prepare_refgame:
            query_embed = torch.cat([query_embed, self.class_embed.weight], 0)

        memory_cache = self.mdetr.transformer(
            self.mdetr.input_proj(src), 
            mask,
            query_embed,
            pos[-1],
            captions,
            encode_and_save=True,
            text_memory=None,
            img_memory=None,
            text_attention_mask=None,
        )

        hidden_states = self.mdetr.transformer(
            mask=memory_cache["mask"],
            query_embed=memory_cache["query_embed"],
            pos_embed=memory_cache["pos_embed"],
            encode_and_save=False,
            text_memory=memory_cache["text_memory_resized"],
            img_memory=memory_cache["img_memory"],
            text_attention_mask=memory_cache["text_attention_mask"],
        )

        return {
            "hidden_states": hidden_states,
            "memory_cache": memory_cache,
        }

    def forward(self, batch):
        ret = dict()
        
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Touchdown SDR 
        if "touchdown_sdr" in self.current_tasks:
            ret.update(objectives.compute_mdetr_touchdown_loss(self, batch))

        # Flickr 30k Entities reference game
        if "refgame" in self.current_tasks:
            ret.update(objectives.compute_mdetr_refgame_loss(self, batch))
        return ret

    # def training_step(self, batch, batch_idx, optimizer_idx):
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
            self.lr_schedulers().optimizer = self.trainer.strategy._lightning_optimizers[0] 
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
                nn_modules_to_monitor = []
                if self.prepare_sdr:
                    nn_modules_to_monitor.append(self.sdr_head)
                elif self.prepare_refgame:
                    nn_modules_to_monitor.append(self.contrastive_head)
                wandb.watch(nn_modules_to_monitor, log='all', log_freq=500)

            if self.wandb_logger is not None:
                for sp in summaries.keys():
                    self.wandb_logger.log(results=summaries[sp], split=sp, step=self.global_step, commit=False)

        metric_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        metric_utils.set_task(self)
        output = self(batch)
        ret = dict()
        
        return ret

    def test_epoch_end(self, outs):
        summaries = {}
        self.checkpoint_metric = 0.
        
        # summarize evaluation results
        for evaluator in self.evaluators:
            evaluator.synchronize_between_processes()
            if dist.is_main_process():
                if isinstance(evaluator, (TouchdownSDREvaluator, F30kRefGameEvaluator, TangramGameEvaluator)):
                    task_evaluator = evaluator
                    summary, task_scores = task_evaluator.summarize(eval_correlation=True)
                elif isinstance(evaluator, GroundingBoxEvaluator):
                    grounding_evaluator = evaluator
                    summary = grounding_evaluator.summarize(task_scores)
                elif isinstance(evaluator, DBVisualizer):
                    complementary_evaluator = evaluator

                summaries.update(summary)
            
        # save evaluation results to database
        if dist.is_main_process():
            if self.hparams.config["test_update_db"]:
                from src.evals import update_db
                task_type = self.hparams.config["datasets"][0]
                exp_name = self.hparams.config["exp_name"]
                update_db(task_evaluator, grounding_evaluator, complementary_evaluator, task_type, exp_name)

        # print out result summary
        for sp in summaries:
            print("="*10 + f" {sp} " + "="*10)
            pprint.pprint(summaries[sp])
        outdir = "eval_outputs/{}".format(self.hparams.config["exp_name"])
        os.makedirs(outdir, exist_ok=True)
        summary_save_path = os.path.join(outdir, "sumamry.pkl")
        pickle.dump(summaries, open(summary_save_path, "wb"))

        # wrap up epoch
        if "debug" in summaries and "accuracy@80px" in summaries["debug"]:
            self.checkpoint_metric += summaries["debug"]["accuracy@80px"]
        metric_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return mdetr_utils.set_schedule(self)

