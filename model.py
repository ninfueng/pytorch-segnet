"""
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision.models import vgg16, vgg16_bn
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.metrics.functional.classification import iou
from utils import IoU

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=10,
   verbose=True,
   mode='min')


class SegNet(pl.LightningModule):
    def __init__(
            self, 
            class_weights = None, 
            input_channels: int = 3, 
            output_channels: int = 21,
            ignore_idx: int = 255,
            lr: float = 1e-3,
            weight_decay: float = 1e-6,
            batch_size: int = 32,
            milestones: list = None):

        super().__init__()
        self.input_channels = input_channels
        self.num_channels = input_channels
        self.output_channels = output_channels
        self.save_hyperparameters(
            'lr', 'weight_decay', 'batch_size')
        self.ignore_idx = ignore_idx
        self.milestones = milestones
        self.INPLACE = True
        self.iou = IoU(21, ignore_index=255)

        
        if class_weights is None:
            self.class_weights = [
                0.0008, 0.0594, 0.1569, 0.0501, 0.0718, 
                0.0798, 0.0258, 0.0318, 0.0165, 0.0452, 
                0.0550, 0.0389, 0.0263, 0.0478, 0.0398,
                0.0099, 0.0757, 0.0508, 0.0367, 0.0293,
                0.0517]
        else:
            self.class_weights = class_weights
        
        # Model definitation.
        self.conv_00 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_01 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=self.INPLACE))
        self.conv_10 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_20 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_21 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_22 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_30 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_31 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_32 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))        

        self.conv_40 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_41 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.conv_42 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        # Decoder layers
        self.deconv_42 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_41 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_40 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_32 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_31 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_30 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_22 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_21 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_20 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_11 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_10 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_01 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=self.INPLACE))

        self.deconv_00 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=self.output_channels,
                kernel_size=3,
                padding=1))

    def forward(self, input_img):
        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = self.conv_00(input_img)
        x_01 = self.conv_01(x_00)
        x_0, indices_0 = F.max_pool2d(
            x_01, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = self.conv_10(x_0)
        x_11 = self.conv_11(x_10)
        x_1, indices_1 = F.max_pool2d(
            x_11, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = self.conv_20(x_1)
        x_21 = self.conv_21(x_20)
        x_22 = self.conv_22(x_21)
        x_2, indices_2 = F.max_pool2d(
            x_22, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = self.conv_30(x_2)
        x_31 = self.conv_31(x_30)
        x_32 = self.conv_32(x_31)
        x_3, indices_3 = F.max_pool2d(
            x_32, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = self.conv_40(x_3)
        x_41 = self.conv_41(x_40)
        x_42 = self.conv_42(x_41)
        x_4, indices_4 = F.max_pool2d(
            x_42, kernel_size=2, stride=2, return_indices=True)

        # Decoder
        dim_d = x_4.size()

        # Decoder Stage - 5
        x_4d = F.max_unpool2d(
            x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = self.deconv_42(x_4d)
        x_41d = self.deconv_41(x_42d)
        x_40d = self.deconv_40(x_41d)
        dim_4d = x_40d.size()

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(
            x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = self.deconv_32(x_3d)
        x_31d = self.deconv_31(x_32d)
        x_30d = self.deconv_30(x_31d)
        dim_3d = x_30d.size()

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(
            x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = self.deconv_22(x_2d)
        x_21d = self.deconv_21(x_22d)
        x_20d = self.deconv_20(x_21d)
        dim_2d = x_20d.size()

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(
            x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = self.deconv_11(x_1d)
        x_10d = self.deconv_10(x_11d)
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(
            x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = self.deconv_01(x_0d)
        x_00d = self.deconv_00(x_01d)
        dim_0d = x_00d.size()
        #x_softmax = F.softmax(x_00d, dim=1)

        return x_00d#, x_softmax

    def configure_optimizers(self):
        
        lr = self.hparams.lr
        # decoder = self.get_10x_lr_children()
        # encoder = self.get_1x_lr_children()
        # list_optims = [
        #     {'params': decoder, 'lr': lr}, 
        #     {'params': encoder, 'lr': lr*10}]

        optimizer = optim.AdamW(
            self.parameters(),
            weight_decay=self.hparams.weight_decay)
        if self.milestones is None:
            scheduler = MultiStepLR(
                optimizer, milestones=self.milestones, gamma=0.1)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self.forward(img)
        loss = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=255)(logits, mask)
        
        logits = F.softmax(logits, dim=1)
        
        miou = iou(logits.argmax(dim=1), mask, ignore_index=self.ignore_idx)
        #_, miou = self.iou.add(logits.argmax(dim=1).detach(), mask.detach())
        
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_miou', miou, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self.forward(img)
        loss = nn.CrossEntropyLoss(
            weight=self.class_weights, 
            ignore_index=self.ignore_idx)(logits, mask)
        
        logits = F.softmax(logits, dim=1)
        miou = iou(logits.argmax(dim=1), mask, ignore_index=self.ignore_idx)

        #miou = iou.add(logits.detach(), mask.detach())
        #_, miou = self.iou.add(logits.argmax(dim=1).detach(), mask.detach())
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_miou', miou, on_epoch=True)
        
        return {'val_loss': loss, 'val_miou': miou}
        
    
    def validation_epoch_end(self, outputs):
        r"""
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['val_miou'] for x in outputs]).mean()
        return {'val_avg_loss': avg_loss, 'val_avg_iou': avg_iou}

    
    def load_pretrained_vgg16(self) -> None:
        r"""Load pretrained VGG16 weight to SegNet.
        """
        pretrain = vgg16_bn(pretrained=True)
        conv_bn = [i for i in pretrain.features.children() if isinstance(i, nn.Conv2d) or isinstance(i, nn.BatchNorm2d)]
        seg_conv = [
            self.conv_00[0], self.conv_00[1], 
            self.conv_01[0], self.conv_01[1],
            self.conv_10[0], self.conv_10[1],
            self.conv_11[0], self.conv_11[1],
            self.conv_20[0], self.conv_20[1], 
            self.conv_21[0], self.conv_21[1],
            self.conv_22[0], self.conv_22[1],
            self.conv_30[0], self.conv_30[1],
            self.conv_31[0], self.conv_31[1],
            self.conv_32[0], self.conv_32[1],
            self.conv_40[0], self.conv_40[1], 
            self.conv_41[0], self.conv_41[1], 
            self.conv_42[0], self.conv_42[1]]
        
        assert len(conv_bn) == len(seg_conv)
        for c, s in zip(conv_bn, seg_conv):
             assert c.weight.size() == s.weight.size()
             assert c.bias.size() == s.bias.size()
             s.weight.data = c.weight.data
             s.bias.data = c.bias.data
        print('Finished load VGG16 weights.')
    
    def init_weights(self):
        """Initialize weights and biases of models. 
        From: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        """
        for ch in self.children():
            for c in ch:
                if isinstance(c, nn.Conv2d):
                    nn.init.kaiming_uniform_(c.weight, nonlinearity='relu')
                    if c.bias is not None:
                        nn.init.zeros_(c.bias)
                elif isinstance(c, nn.BatchNorm2d):
                    torch.nn.init.normal_(c.weight, 1.0, 0.02)
                    torch.nn.init.zeros_(c.bias)
                elif isinstance(c, nn.Linear):
                    nn.init.kaiming_uniform_(c.weight, nonlinearity='relu')
                    if c.bias is not None:
                        nn.init.zeros_(c.bias)
                elif isinstance(c, nn.ConvTranspose2d):
                    nn.init.kaiming_uniform_(c.weight, nonlinearity='relu')
                    if c.bias is not None:
                        nn.init.zeros_(c.bias)
                else:
                    print(f'{c.__class__.__name__} is not initialized.')
    
    def get_1x_lr_children(self) -> list:
        """Find children from name. Designed for finding 1x learning `conv` and
        finding 10x learning `deconv`.
        """
        params = []
        filter_name: str = 'conv'
        filter_name2: str = 'deconv'
        for name, child in self.named_children():
            if name.find(filter_name) != -1:
                if name.find(filter_name2) != -1:
                    continue
                params += list(child.parameters()) 
        return params

    def get_10x_lr_children(self) -> list:
        params = []
        filter_name: str = 'deconv'
        for name, child in self.named_children():
            if name.find(filter_name) != -1:
                params += list(child.parameters()) 
        return params


if __name__ == '__main__':
    model = SegNet(
        input_channels=3,
        output_channels=21)
    print(model)

    model.load_pretrained_vgg16() 
    
    
