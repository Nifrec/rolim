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
        # Hardcoded parameters are taken from the paper by Mnih et al.
        self._layers = nn.Sequential(
                nn.Conv2d(in_channels = channels,
                          out_channels=16,
                          kernel_size = 8,
                          stride = 4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=16 * channels,
                          out_channels=32,
                          kernel_size = 4,
                          stride=2,
                          padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=256, out_features=out_size)
                )

        def forward(self, t: Tensor) -> Tensor:
            return self._layers(t)

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
