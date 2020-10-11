from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOCSegDataset(Dataset):    
    def __init__(
            self,
            resize_size: list = (256, 256),
            crop_size: list = (224, 224),
            base_dir='./data/VOCdevkit/VOC2012/',
            train: bool = True) -> None:
        """Resize first then crop image.
        """
        super().__init__()
        assert isinstance(base_dir, str)
        assert isinstance(train, bool)
        assert len(resize_size) == 2
        assert len(crop_size) == 2
        image_dir = os.path.join(base_dir, 'JPEGImages')
        seg_dir = os.path.join(base_dir, 'SegmentationClass')
        txt_dir = os.path.join(base_dir, 'ImageSets', 'Segmentation')

        if train:
            txt_dir = os.path.join(txt_dir, 'train') + '.txt'
        else:
            txt_dir = os.path.join(txt_dir, 'val') + '.txt'
            
        data_path = np.loadtxt(txt_dir, delimiter='\n', dtype=str).tolist()
        img_list = [os.path.join(image_dir, i) + '.jpg'  for i in data_path]
        seg_list = [os.path.join(seg_dir, i) + '.png'  for i in data_path]
        self.img_list = img_list
        self.seg_list = seg_list
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.seg_list[idx])
        img, mask = np.array(img), np.array(mask)
        
        if self.train:
            img, mask = self.transform_train(img, mask)
        else:
            img, mask = self.transform_val(img, mask)
        return img, mask

    def transform_train(self, img, mask):
        """Refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        aug = A.Compose([
            A.RandomResizedCrop(*self.crop_size),
            A.HorizontalFlip(),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
            ])
        augmented = aug(image=img, mask=mask)
        return augmented['image'], augmented['mask'].long()

    def transform_val(self, img, mask):        
        aug = A.Compose([ 
            A.Resize(*self.resize_size),
            A.CenterCrop(*self.crop_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
            ])
        augmented = aug(image=img, mask=mask)
        return augmented['image'], augmented['mask'].long()


if __name__ == '__main__':
    dataset = VOCSegDataset()
    train_img, train_mask = next(iter(dataset))
    
    # Comment out ToTensorV2 first.
    import matplotlib.pyplot as plt
    plt.imshow(train_img)
    plt.show()
    plt.imshow(train_mask)
    plt.show()
    
    dataset = VOCSegDataset(train=False)
    test_img, test_mask = next(iter(dataset))
    plt.imshow(test_img)
    plt.show()
    plt.imshow(test_mask)
    plt.show()
