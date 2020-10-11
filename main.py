import os
import argparse
from dataset import VOCSegmentation
from model import SegNet
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import lr_finder_plot, cal_class_weights


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
    parser.add_argument('--max_epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()

    NUM_CLASSES = 21
    milestones = [200, 400]
    pl.seed_everything(args.seed)
    train_path = os.path.join(args.data_root, args.train_path)
    val_path = os.path.join(args.data_root, args.val_path)    
    
    train_dataset = VOCSegmentation(
        base_size=224,
        crop_size=224,
        split='train')
    
    val_dataset = VOCSegmentation(
        base_size=224,
        crop_size=224,
        split='val')

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
        milestones=milestones,
        lr=args.lr,
        input_channels=3,
        output_channels=NUM_CLASSES,
        class_weights=class_weights)
    model.init_weights()
    model.load_pretrained_vgg16()


    trainer = pl.Trainer(
        max_epochs=args.max_epochs, gpus=args.gpus, precision=args.precision)
    #lr_finder_plot(model, trainer, train_dataloader, val_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)
