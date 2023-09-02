# defines wandb logger
import os
import wandb 
from datetime import datetime
from typing import Dict
from argparse import Namespace

class WandbLogger(object):
    def __init__(self, args: Namespace):
        super().__init__()
        self._step = 0
        self._debug = args.debug
        now = datetime.now()
        exp_name = args.exp_name + "-" + str(now).replace(" ", "-")
        wandb_input = {
            "entity": "lil",
            "name": exp_name,
            "config": args
        }
        wandb_input["project"] = "p-interactive-touchdown-sdr-vilt-debug" if args.debug else "p-interactive-touchdown-sdr-vilt"
        if args.wandb_run_id is not None:
            wandb_input["id"] = args.wandb_run_id
            wandb_input["resume"] = "must"
        wandb.init(**wandb_input)

        self._sdr_train_loss_keys = []

    def log(self, results, split: str, step: int = None, commit=False):
        formated_results = {}
        if step is not None:
            self._step = step

        for k, v in results.items():
            formated_results["{}/{}".format(split, k)] = v
        wandb.log(formated_results, step=self._step, commit=commit)

    def get_step(self):
        return self._step


    
