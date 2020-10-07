# Pytorch SegNet#

This repo is a re-implementation of SegNet.

[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)


## To use:
```

python train.py --data_root ./data/VOCdevkit/VOC2012/ --train_path ImageSets/Segmentation/train.txt --img_dir JPEGImages --mask_dir SegmentationClass --save_dir ./save
```
