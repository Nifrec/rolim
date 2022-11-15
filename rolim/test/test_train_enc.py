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

Testcases for `rolim/encoder/train_enc.py`.
"""
# Library imports:
import unittest
import torch
from torch import Tensor

# Local imports:
from rolim.encoder.train_enc import pairwise_mse
from rolim.tools.testing import assert_tensor_eq

class PairWiseMSETestCase(unittest.TestCase):
    """
    Testcases for `pairwise_mse()`.
    """

    def test_3d_data(self):
       inp = torch.tensor([
           [1, 2, 1],
           [1, 1, 1], # Pair 1 loss: 1

           [2, 2.5, 2],
           [2, 2, 2], # Pair 2 loss: 0.5^2 = 0.25

           [1, 0, 0],
           [0, 0, 1]], # Pair 3 loss: 2
           dtype=torch.float)
       expected = torch.tensor((1/3)*(1+0.25+2), dtype=torch.float)
       output = pairwise_mse(inp)
       assert_tensor_eq(expected, output)
    

if __name__ == "__main__":
    unittest.main()

