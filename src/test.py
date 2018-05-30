"""Test for SegNet"""

from model import SegNet
import numpy as np
import torch
from __future__ import print_function


if __name__ == "__main__":
    # RGB
    input_channels = 3
    # Num classes
    output_channels = 10

    # Model
    model = SegNet(input_channels=input_channels, output_channels=output_channels)

    img = torch.zeros((1, 3, 224, 224))
    output, softmaxed_output = model(img)

    print(output.size())
    print(softmaxed_output.size())

    print(output[0,:,0,0])
    print(softmaxed_output[0,:,0,0].sum())