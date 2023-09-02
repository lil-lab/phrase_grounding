import torch
import random
import wandb
import numpy as np


from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import torch.nn as nn
import torch.nn.functional as F

def initialize_vilt_state_dict(pl_module, state_dict):
    # extend positional embeddings if necessary
    posemb_weight = state_dict['text_embeddings.position_embeddings.weight'].permute(1,0)
    pretrain_max_text_len = posemb_weight.shape[0]
    max_text_len = pl_module.hparams.config["max_text_len"] 
    interp_posembs = (max_text_len != pretrain_max_text_len)
    if interp_posembs:
        state_dict['text_embeddings.position_ids'] = torch.tensor([i for i in range(max_text_len)]).unsqueeze(0)
        pos_embed = F.interpolate(
            posemb_weight.unsqueeze(0), size=(max_text_len), mode="linear", align_corners=True,
        )
        pos_embed = pos_embed.permute(0,2,1)
        state_dict['text_embeddings.position_embeddings.weight'] = pos_embed.squeeze(0)
    
    # initalize new token embeddings if necessary
    vocab_size = pl_module.hparams.config["vocab_size"]
    pretrained_vocab_size, worddim = state_dict["text_embeddings.word_embeddings.weight"].shape
    extend_vocab = (vocab_size > pretrained_vocab_size)
    if extend_vocab:
        wordembs = torch.zeros((vocab_size , worddim))
        wordembs.data.normal_(mean=0.0, std=0.02)
        wordembs[:pretrained_vocab_size ,...] = state_dict["text_embeddings.word_embeddings.weight"]
        state_dict["text_embeddings.word_embeddings.weight"] = wordembs

    return state_dict


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and p.requires_grad
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and p.requires_grad

            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    
    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )
    
    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

