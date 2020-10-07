"""
"""
import os
import time
import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
from model import SegNet
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def train():
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in tqdm(range(NUM_EPOCHS)):
        loss_f = 0
        t_start = time.time()

        for batch in train_dataloader:
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            predicted_tensor, softmaxed_tensor = model(input_tensor)


            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))
        print(f'Epoch #{epoch + 1}\tLoss: {loss_f:.8f}\t Time: {delta:2f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a SegNet model')
    parser.add_argument('--data_root', type=str, default='./data/VOCdevkit/VOC2012/')
    parser.add_argument('--train_path', type=str, default='ImageSets/Segmentation/train.txt')
    parser.add_argument('--img_dir', type=str, default='JPEGImages')
    parser.add_argument('--mask_dir', type=str, default='SegmentationClass')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--checkpoint', type=str, default='save/best.pth')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    NUM_INPUT_CHANNELS = 3
    NUM_OUTPUT_CHANNELS = NUM_CLASSES
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    BATCH_SIZE = 8

    data_root = args.data_root
    train_path = os.path.join(data_root, args.train_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    train_dataset = PascalVOCDataset(list_file=train_path,
                                     img_dir=img_dir,
                                     mask_dir=mask_dir)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)

    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)

        class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        class_weights = 1.0/train_dataset.get_class_probability()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train()
