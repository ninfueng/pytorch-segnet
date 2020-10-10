"""
"""
import os
import argparse
from dataset import VOCSegmentation
from model import SegNet
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
from collections import Counter


def lr_finder_plot(
        model: pl.LightningModule,
        trainer: pl.Trainer,
        train_dataloader=None,
        val_dataloader=None) -> None:
    r"""Plotting and suggest a initial learning to train model with.
    Get for using usable batch and learning rate finding.
    """
    if train_dataloader is None or val_dataloader is None:
        lr_finder = trainer.lr_find(model)
    else:
        lr_finder = trainer.lr_find(
            model, train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader)
    recommended_lr = lr_finder.suggestion()
    print(f'Recommended learning rate: {recommended_lr}')
    lr_finder.plot(suggest=True, show=True)


def cal_class_weights(train_dataloader, num_cls: int)-> torch.Tensor:
    """From: https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training and validation SegNet model on PASCAL VOC 2012')
    parser.add_argument('--data_root', type=str, default='./data/VOCdevkit/VOC2012/')
    parser.add_argument('--train_path', type=str, default='ImageSets/Segmentation/train.txt')
    parser.add_argument('--val_path', type=str, default='ImageSets/Segmentation/val.txt')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--checkpoint', type=str, default='save/best.pth')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()

    NUM_CLASSES = 21
    pl.seed_everything(args.seed)
    train_path = os.path.join(args.data_root, args.train_path)
    val_path = os.path.join(args.data_root, args.val_path)

    # train_dataset = PascalVOCDataset(
    #     root=args.data_root,
    #     txt_path='ImageSets/Segmentation/train.txt',
    #     resize_size=(224, 224),
    #     crop_size=(224, 224)
    # )
    
    
    train_dataset = VOCSegmentation(
        base_size=224,
        crop_size=224,
        split='train'
    )
    
    val_dataset = VOCSegmentation(
        base_size=224,
        crop_size=224,
        split='val'
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)
    
    class_weights = cal_class_weights(train_dataloader, NUM_CLASSES)
    class_weights = class_weights.cuda()
    
    
    model = SegNet(
        input_channels=3,
        output_channels=NUM_CLASSES,
        class_weights=class_weights)
    model.load_vgg16_weight()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, gpus=args.gpus, precision=args.precision)
    #lr_finder_plot(model, trainer, train_dataloader, val_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)
