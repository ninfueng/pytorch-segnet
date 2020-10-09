"""
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.metrics.functional.classification import iou


early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=5,
   verbose=True,
   mode='min')


class SegNet(pl.LightningModule):    
    def __init__(
            self, 
            class_weights, 
            input_channels: int, 
            output_channels: int, 
            lr: float=1e-3, 
            weight_decay: float=1e-4):
        
        super().__init__()
        self.input_channels = input_channels
        self.num_channels = input_channels
        self.output_channels = output_channels
        self.class_weights = class_weights
        self.save_hyperparameters('lr', 'weight_decay')
        self.IGNORE_IDX = 255

        
        self.encoder_conv_00 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_01 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv_10 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.encoder_conv_20 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_21 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_22 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder_conv_30 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_31 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_32 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.encoder_conv_40 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_41 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv_42 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder layers
        self.decoder_convtr_42 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_41 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_40 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_32 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_31 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_30 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_22 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_21 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_20 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_11 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_10 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_01 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder_convtr_00 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=self.output_channels,
                kernel_size=3,
                padding=1)
        )

    def forward(self, input_img):
        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = self.encoder_conv_00(input_img)
        x_01 = self.encoder_conv_01(x_00)
        x_0, indices_0 = F.max_pool2d(
            x_01, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = self.encoder_conv_10(x_0)
        x_11 = self.encoder_conv_11(x_10)
        x_1, indices_1 = F.max_pool2d(
            x_11, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = self.encoder_conv_20(x_1)
        x_21 = self.encoder_conv_21(x_20)
        x_22 = self.encoder_conv_22(x_21)
        x_2, indices_2 = F.max_pool2d(
            x_22, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = self.encoder_conv_30(x_2)
        x_31 = self.encoder_conv_31(x_30)
        x_32 = self.encoder_conv_32(x_31)
        x_3, indices_3 = F.max_pool2d(
            x_32, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = self.encoder_conv_40(x_3)
        x_41 = self.encoder_conv_41(x_40)
        x_42 = self.encoder_conv_42(x_41)
        x_4, indices_4 = F.max_pool2d(
            x_42, kernel_size=2, stride=2, return_indices=True)

        # Decoder
        dim_d = x_4.size()

        # Decoder Stage - 5
        x_4d = F.max_unpool2d(
            x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = self.decoder_convtr_42(x_4d)
        x_41d = self.decoder_convtr_41(x_42d)
        x_40d = self.decoder_convtr_40(x_41d)
        dim_4d = x_40d.size()

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(
            x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = self.decoder_convtr_32(x_3d)
        x_31d = self.decoder_convtr_31(x_32d)
        x_30d = self.decoder_convtr_30(x_31d)
        dim_3d = x_30d.size()

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(
            x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = self.decoder_convtr_22(x_2d)
        x_21d = self.decoder_convtr_21(x_22d)
        x_20d = self.decoder_convtr_20(x_21d)
        dim_2d = x_20d.size()

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(
            x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = self.decoder_convtr_11(x_1d)
        x_10d = self.decoder_convtr_10(x_11d)
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(
            x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = self.decoder_convtr_01(x_0d)
        x_00d = self.decoder_convtr_00(x_01d)
        dim_0d = x_00d.size()
        x_softmax = F.softmax(x_00d, dim=1)

        return x_00d, x_softmax

    def configure_optimizers(self):
        if self.hparams.lr is None:
            lr = 1e-3
        else:
            lr = self.hparams.lr
        optimizer = optim.Adam(self.parameters(), lr=lr)
        #, weight_decay=self.hparams.weight_decay
        return optimizer

    def training_step(self, batch, batch_idx):
        #img, mask = batch
        img, mask = batch['image'], batch['label']
        img, mask = img.cuda(), mask.cuda()
        img, mask = Variable(img, requires_grad=True), Variable(mask, requires_grad=False)
        logits, _ = self.forward(img)
        loss = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=255)(logits, mask)
        
        miou = iou(logits.argmax(axis=1), mask, ignore_index=self.IGNORE_IDX)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_miou', miou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        #img, mask = batch
        img, mask = batch['image'],  batch['label']#torch.LongTensor(batch['label'])
        #img, mask = img.cuda(), mask.cuda()
        img = Variable(img, requires_grad=False)
        logits, _ = self.forward(img)
        #weight=self.class_weights
        loss = nn.CrossEntropyLoss(
            weight=self.class_weights, 
            ignore_index=self.IGNORE_IDX)(logits, mask)
        
        miou = iou(logits.argmax(axis=1), mask, ignore_index=self.IGNORE_IDX)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_miou', miou, on_epoch=True)
    
    def load_vgg16_weight(self) -> None:
        model = vgg16(pretrained=True)
        assert self.encoder_conv_00[0].weight.size() == model.features[0].weight.size()
        self.encoder_conv_00[0].weight.data = model.features[0].weight.data
        assert self.encoder_conv_00[0].bias.size() == model.features[0].bias.size()
        self.encoder_conv_00[0].bias.data = model.features[0].bias.data

        assert self.encoder_conv_01[0].weight.size() == model.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = model.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == model.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = model.features[2].bias.data

        assert self.encoder_conv_10[0].weight.size() == model.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = model.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == model.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = model.features[5].bias.data

        assert self.encoder_conv_11[0].weight.size() == model.features[7].weight.size()
        self.encoder_conv_11[0].weight.data = model.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == model.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = model.features[7].bias.data

        assert self.encoder_conv_20[0].weight.size() == model.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = model.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == model.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = model.features[10].bias.data

        assert self.encoder_conv_21[0].weight.size() == model.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = model.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == model.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = model.features[12].bias.data

        assert self.encoder_conv_22[0].weight.size() == model.features[14].weight.size()
        self.encoder_conv_22[0].weight.data = model.features[14].weight.data
        assert self.encoder_conv_22[0].bias.size() == model.features[14].bias.size()
        self.encoder_conv_22[0].bias.data = model.features[14].bias.data

        assert self.encoder_conv_30[0].weight.size() == model.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = model.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == model.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = model.features[17].bias.data

        assert self.encoder_conv_31[0].weight.size() == model.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = model.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == model.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = model.features[19].bias.data

        assert self.encoder_conv_32[0].weight.size() == model.features[21].weight.size()
        self.encoder_conv_32[0].weight.data = model.features[21].weight.data
        assert self.encoder_conv_32[0].bias.size() == model.features[21].bias.size()
        self.encoder_conv_32[0].bias.data = model.features[21].bias.data

        assert self.encoder_conv_40[0].weight.size() == model.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = model.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == model.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = model.features[24].bias.data

        assert self.encoder_conv_41[0].weight.size() == model.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = model.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == model.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = model.features[26].bias.data

        assert self.encoder_conv_42[0].weight.size() == model.features[28].weight.size()
        self.encoder_conv_42[0].weight.data = model.features[28].weight.data
        assert self.encoder_conv_42[0].bias.size() == model.features[28].bias.size()
        self.encoder_conv_42[0].bias.data = model.features[28].bias.data

