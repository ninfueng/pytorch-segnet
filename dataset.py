from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import custom_transforms as tr


class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return './data/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 base_size: int,
                 crop_size: int,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.base_size = base_size
        self.crop_size = crop_size
        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomHorizontalFlip(),
            #tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size),
            ##tr.RandomGaussianBlur(),
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'
    


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(513, 513, split='train')
    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()            
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)



# """Pascal VOC Dataset Segmentation Dataloader"""
#
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# #from albumentations import
# from torch.utils.data import Dataset
# from PIL import Image
#
#
# class PascalVOCDataset(Dataset):
#     """Pascal VOC label: 0-21 (include background).
#     With label as .png, which indicates
#     """
#     def __init__(
#             self, root: str,
#             txt_path: str,
#             resize_size: tuple,
#             crop_size: tuple,
#             transforms=None) -> None:
#
#         assert isinstance(root, str)
#         assert isinstance(txt_path, str)
#         assert len(resize_size) == 2
#         assert len(crop_size) == 2
#
#         self.VOC_LABELS = (
#             'background', 'aeroplane', 'bicycle',
#             'bird', 'boat', 'bottle', 'bus',
#             'car', 'cat', 'chair', 'cow',
#             'diningtable', 'dog', 'horse',
#             'motorbike', 'person', 'pottedplant',
#             'sheep', 'sofa', 'train', 'tvmonitor')
#         self.NUM_CLASSES = len(self.VOC_LABELS)
#
#         txt_path = os.path.join(root, txt_path)
#         self.images = np.loadtxt(txt_path, dtype=str, delimiter='\n')
#         self.img_dir = os.path.join(root, 'JPEGImages')
#         self.mask_dir = os.path.join(root, 'SegmentationClass')
#         self.transforms = transforms
#         self.resize_size = resize_size
#         self.crop_size = crop_size
#         self.weights = None
#
#     def __len__(self) -> int:
#         return len(self.images)
#
#     def __getitem__(self, index):
#         name = self.images[index]
#         image_path = os.path.join(self.img_dir, name + '.jpg')
#         mask_path = os.path.join(self.mask_dir, name + '.png')
#         image = self.load_image(path=image_path)
#         mask = self.load_mask(path=mask_path)
#         return torch.FloatTensor(image), torch.LongTensor(mask)
#
#     def compute_class_weights(self):
#         r"""Due to imbalance dataset. Computing to find all weight per each class.
#         """
#         counter = dict((i, 0) for i in range(self.NUM_CLASSES))
#         for name in self.images:
#             mask_path = os.path.join(self.mask_dir, name + '.png')
#             mask = Image.open(mask_path)
#             mask = mask.resize(self.resize_size)
#             mask = np.array(mask)
#             mask[mask == 255] = len(self.VOC_LABELS)
#             for i in range(self.NUM_CLASSES):
#                 counter[i] += np.sum(mask == i)
#
#         weights = np.array(list(counter.values()))
#         fract_weights = weights/weights.sum()
#         return torch.Tensor(fract_weights)
#
#     def load_image(self, path: str = None):
#         image = Image.open(path)
#         image = np.transpose(image.resize(self.resize_size), (2, 1, 0))
#         if self.transforms is None:
#             pass
#         image = np.array(image, dtype=np.float32)/255.0
#         return image
#
#     def load_mask(self, path: str = None):
#         mask = Image.open(path)
#         mask = mask.resize(self.resize_size)
#         mask = np.array(mask)
#         mask[mask == 255] = len(self.VOC_LABELS)
#         return mask
#
#     def get_label_definition(self):
#         r"""Return definition of each segmentation pixels.
#         """
#         return np.asarray(
#             [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
#              [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
#              [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
#              [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
#              [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
#              [0, 64, 128]])
#
#
# if __name__ == "__main__":
#     data_root = os.path.join("data", 'VOCdevkit', 'VOC2012')
#     list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
#     img_dir = os.path.join(data_root, "JPEGImages")
#     mask_dir = os.path.join(data_root, "SegmentationObject")
#
#
#     objects_dataset = PascalVOCDataset(
#         root=data_root,
#         txt_path=os.path.join('ImageSets/Segmentation/train.txt'),
#         resize_size=(224, 224),
#         crop_size=(224, 224)
#     )
#     #print(objects_dataset.get_class_probability())
#     print(objects_dataset.compute_class_weights())
#
#
#     sample = objects_dataset[2]
#     image, mask = sample
#     print(np.unique(mask))
#
#     image.transpose_(0, 2)
#
#     fig = plt.figure()
#     a = fig.add_subplot(1,2,1)
#     plt.imshow(image)
#
#     a = fig.add_subplot(1,2,2)
#     plt.imshow(mask)
#
#     plt.show()
#
