"""
© Lulof Pirée 2022

--------------------------------------------------------------------------------
Part of "rolim": Reproduction Of the paper "Latent World Models
For Intrinsically Motivated Exploration"
by Alexander Ermolov and Nice Sebe (2020).
The reproduction is performed by Lulof Pirée
as part of the course 'Seminar Advanced Deep Reinforcement Learning'
at Leiden University.

Author: Lulof Pirée
Fall 2022

Copyright (C) 2022 Lulof Pirée

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------
*File content:*

Definitions of PyTorch Neural Network models
(i.e. architectures of the layers and forward function).
Also offers functions for loading and saving networks.
"""
# Library imports:
import math
import torch
import torch.nn as nn
from torch import Tensor

class AtariCNN(nn.Module):
    """
    Implementation of a CNN as used in the paper 
    'Playing Atari with Deep Reinforcement Learning'
    by Mnih et al.. This is also the architecture
    used in the LWM paper for both the DQN and the encoder.

    Image dimensions are represented as `batch×channel×height×width`,
    and input to this network should follow the same convention.
    (batch runs over the different samples of the batch,
    channel over the different channels of an image (e.g.
    R,G,B channels, stacked frames of a game)).
    """

    def __init__(self, 
                 channels: int,
                 height:int,
                 width: int,
                 out_size: int):
        """
        Arguments:
        * channels: numbers of channels in an input image.
        * height: height of input images.
        * width: width of input images.
        * out_size: desired length of output vectors.

        """
        super().__init__()
        self.__channels = channels
        self.__height = height
        self.__width = width
        self.__out_size = out_size

        # Hardcoded parameters are taken from the paper by Mnih et al.
        conv1 = nn.Conv2d(in_channels = channels,
                          out_channels=16*channels, # 16 filters
                          kernel_size = 8,
                          stride = 4,
                          padding=0)
        
        conv2 = nn.Conv2d(in_channels = conv1.out_channels,
                          out_channels=32*conv1.out_channels, # 32 filters
                          kernel_size = 4,
                          stride = 2,
                          padding=0)

        conv1_out_size = conv_output_size(conv1, height, width)
        conv2_out_size = conv_output_size(conv2, conv1_out_size[0],
                                          conv1_out_size[1])

        # Total number of output features (tensor elements)
        # of the 2 convolutional layers for each image in the batch.
        # Just height*width*channels
        conv_out_features = conv2_out_size[0]*conv2_out_size[1]\
                *conv2.out_channels
        self.__check_output_possible(conv_out_features)

        linear = nn.Linear(in_features=conv_out_features,
                           out_features=out_size)

        self._layers = nn.Sequential(conv1, nn.ReLU(),
                                     conv2, nn.ReLU(),
                                     nn.Flatten(),
                                     linear)
    def __str__(self) -> str:
        return (f"AtariCNN("
              + f"{self.channels},"
              + f"{self.height},"
              + f"{self.width},"
              + f"{self.out_size})")

    def forward(self, t: Tensor) -> Tensor:
        return self._layers(t)

    def __check_output_possible(self, conv_out_features: int):
        if conv_out_features <= 0:
            raise RuntimeError("Convolution not possible:"
                "less than 0 elements after convolution layers.\n"
                "Are the input images too small?")

    @property
    def channels(self) -> int:
        """
        Number of expected channels in input images.
        """
        return self.__channels

    @property
    def height(self) -> int:
        """
        Expected height of input images.
        """
        return self.__height

    @property
    def width(self) -> int:
        """
        Expected width of input images.
        """
        return self.__width

    @property
    def out_size(self) -> int:
        """
        Dimension of output vector.
        """
        return self.__out_size


def conv_output_size(conv: nn.Conv2d, height: int, width:int
                     ) -> tuple[int, int]:
    """
    Given an instance of a convolution layer
    and the dimensions of an image,
    return the dimensions of the output of the layer.

    Arguments:
    * conv: a PyTorch 2D convolutional layer.
    * height: image height (pixels).
    * width: image width (pixels).

    Returns:
    * out_height: height of output image.
    * out_width: width of output image.

    Note:
    * The number of output channels is specified in Conv2d itself.
    * The batch dimension is not changed by convolution.
    * This function assumes strides, padding and kernel shapes
        are the same in both dimensions. It also assumes no
        dilation is used.
    """
    stride = conv.stride[0]
    assert isinstance(stride, int)
    padding = conv.padding[0]
    assert isinstance(padding, int)
    kernel_size = conv.kernel_size[0]
    assert isinstance(kernel_size, int)

    # See the equation in the PyTorch documentation for the computations:
    out_height = math.ceil((height + 2*padding -kernel_size -2 )/stride) + 1
    out_width  = math.ceil((width + 2*padding -kernel_size -2 )/stride) + 1

    return (out_height, out_width)


def save_network():
    raise NotImplementedError()
def load_network():
    raise NotImplementedError()
