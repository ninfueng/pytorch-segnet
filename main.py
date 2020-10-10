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


def calculate_weigths_labels(dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        #_, y = sample
        #print(y)
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    #classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    #np.save(classes_weights_path, ret)
    return ret


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
    
    #class_weights = 1.0/calculate_weigths_labels(train_dataloader, NUM_CLASSES)
    class_weights = torch.tensor([
        0.5220, 0.0300, 0.0230, 0.0304, 
        0.0273, 0.0279, 0.0444, 0.0395, 
        0.0556, 0.0308, 0.0298, 0.0339, 
        0.0415, 0.0318, 0.0352, 0.0748, 
        0.0274, 0.0314, 0.0335, 0.0409,
        0.0311])
    
    class_weights = class_weights.float().cuda()
    model = SegNet(
        input_channels=3,
        output_channels=NUM_CLASSES,
        class_weights=class_weights)
    model.load_vgg16_weight()

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, precision=16)
    #lr_finder_plot(model, trainer, train_dataloader, val_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)
