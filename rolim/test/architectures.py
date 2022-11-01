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

Testcases for rolim/networks/architectures.py
"""
# Library imports:
import unittest
import torch
from torch import Tensor
import torch.nn as nn

# Local imports:
from rolim.networks.architectures import conv_output_size


class ConvOutputSizeTestCase(unittest.TestCase):
    """
    Tests the auxiliary function conv_output_size().
    """

    def test_conv_1(self):
        """
        Even kernel, padding and stride sizes.
        """
        self.run_conv_test(4, 4, 4, 180, 200)

    def test_conv_2(self):
        """
        Odd kernel, padding and stride sizes.
        """
        self.run_conv_test(5, 3, 7, 80, 100)

    def test_conv_3(self):
        """
        Mixed kernel, padding and stride size parities.
        """
        self.run_conv_test(2, 3, 1, 180, 198)

    def test_conv_4(self):
        """
        Odd image size. 
        """
        self.run_conv_test(2, 3, 1, 181, 199)

    def run_conv_test(self, kernel_size: int, stride: int, padding: int,
                      height: int, width: int):

        """
        Just input a random image
        of fixed dimensions into a Conv2d,
        and check if the actual output size matches the 
        predicted output size.

        Arguments:
        * kernel_size, stride, padding: parameters of the convolutional
            layer.
        * height, width: dimensions of the input image.
        """
        conv = nn.Conv2d(in_channels=1,
                         out_channels=1,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)
        image = generate_random_image(height, width)
        out_image = conv(image)
        prediction = conv_output_size(conv, height, width)
        self.assertTupleEqual(tuple(out_image.shape[-2:]), prediction)

def generate_random_image(height: int, width: int) -> Tensor:
    """
    Generate a random image of the specified dimensions.
    Return an image of shape `(1, 1, height, width)`.
    """
    return torch.rand((1, 1, height, width), dtype=torch.float)
    
if __name__ == "__main__":
    unittest.main()
