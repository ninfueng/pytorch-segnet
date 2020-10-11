#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from collections import Counter
from tqdm import tqdm
import torch
import pytorch_lightning as pl


def lr_finder_plot(
        model: pl.LightningModule,
        trainer: pl.Trainer,
        train_dataloader = None,
        val_dataloader = None) -> None:
    r"""Plotting and suggest a initial learning to train model with.
    Get for using usable batch and learning rate finding.
    """
    if not hasattr(trainer, 'lr_find'):
        # This indicates pytorch-lightning version >= 1.0.
        trainer = trainer.tuner
    if train_dataloader is None or val_dataloader is None:
        # This indicates trainer contains methods related to dataloader.
        lr_finder = trainer.lr_find(model)
    else:
        lr_finder = trainer.lr_find(
            model, train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader)
    recommended_lr = lr_finder.suggestion()
    print(f'Recommended learning rate: {recommended_lr}')
    lr_finder.plot(suggest=True, show=True)


def cal_class_weights(train_dataloader, num_cls: int)-> torch.Tensor:
    r"""From: https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514
    The more labels the less amount of weights.
    """
    counter = Counter()
    for batch in tqdm(train_dataloader):
        mask = batch['label']
        for i in range(num_cls):
            count = torch.where(mask == i, torch.ones_like(mask), torch.zeros_like(mask)).sum()
            counter.update({i: count})
            
    counts = torch.tensor([counter[i] for i in range(num_cls)]).float()
    nor_counts = counts/counts.sum()
    weights = 1.0/nor_counts
    nor_weights = (weights/weights.sum()).float()
    return nor_weights
