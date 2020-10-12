import os
import argparse
from dataset import VOCSegDataset
from model import SegNet
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import lr_finder_plot, cal_class_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training and validation SegNet model on PASCAL VOC 2012')
    parser.add_argument('--data_root', type=str, default='./data/VOCdevkit/VOC2012/')
    parser.add_argument('--train_path', type=str, default='ImageSets/Segmentation/train.txt')
    parser.add_argument('--val_path', type=str, default='ImageSets/Segmentation/val.txt')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_epochs', type=int, default=2_000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()

    NUM_CLASSES = 21
    MILESTONES = [500, 1000, 1500]
    RESIZE_SIZE = (256, 256)
    CROP_SIZE = (224, 224)
    IN_CHANNEL = 3
    SAVE_PATH = os.path.join(
        args.save_dir, '{epoch:03d}-{val_loss:.4f}')
    LOAD = False
    LOAD_PATH = f'epoch-val_loss.pth'
    
    pl.seed_everything(args.seed)
    train_path = os.path.join(args.data_root, args.train_path)
    val_path = os.path.join(args.data_root, args.val_path)    
    train_dataset = VOCSegDataset(
        resize_size=RESIZE_SIZE, crop_size=CROP_SIZE, train=True)
    val_dataset = VOCSegDataset(
        resize_size=RESIZE_SIZE, crop_size=CROP_SIZE, train=False)

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
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_miou',
        filepath=SAVE_PATH,
        save_top_k=1,
        mode='min')

    #class_weights = cal_class_weights(train_dataloader, NUM_CLASSES)
    class_weights = torch.tensor(
        [0.0006, 0.0630, 0.1572, 0.0539, 0.0757,
         0.0770, 0.0263, 0.0326, 0.0173, 0.0405, 
         0.0552, 0.0356, 0.0275, 0.0500, 0.0409, 
         0.0097, 0.0717, 0.0524, 0.0321, 0.0292, 0.0515])
    class_weights = class_weights.cuda()
    
    model = SegNet(
        milestones=MILESTONES,
        lr=args.lr,
        input_channels=IN_CHANNEL,
        output_channels=NUM_CLASSES,
        class_weights=class_weights,
        weight_decay=args.weight_decay)
    
    if not LOAD:
        model.init_weights()
        model.load_pretrained_vgg16()
    else:
        model.load_from_checkpoint(LOAD_PATH)
        
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, gpus=args.gpus,
        precision=args.precision, 
        checkpoint_callback=checkpoint_callback)
    
    tuner = pl.tuner.tuning.Tuner(trainer)
    #lr_finder_plot(model, trainer, train_dataloader, val_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)
