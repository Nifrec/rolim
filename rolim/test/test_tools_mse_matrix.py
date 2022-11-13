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

Testcases for rolim/tools/mse_matrix.py
"""
# Library imports:
import unittest
import torch
from torch import Tensor

# Local imports:
from rolim.tools.mse_matrix import compute_mse_heatmap
from rolim.tools.testing import assert_tensor_eq

class MSEMatrixTestCase(unittest.TestCase):

    def test_2d_matrix(self):
        """
        Simple case: 2 classes, so the output should be a 2d matrix.
        """
        embeddings_class_1 = torch.tensor([        
                                           [1.0, 2.0, 3.0],
                                           [2.0, 2.0, 2.0]],
                                           dtype=torch.float,
                                           requires_grad=False)
        embeddings_class_2 = torch.tensor([        
                                           [3.0, 3.0, 3.0],
                                           [1.0, 2.0, 1.0]],
                                           dtype=torch.float,
                                           requires_grad=False)
        embeddings_per_class = [embeddings_class_1,
                                embeddings_class_2]

        x11 = embeddings_class_1[0, :]
        x12 = embeddings_class_1[1, :]
        x21 = embeddings_class_2[0, :]
        x22 = embeddings_class_2[1, :]
        
        M11 = (1/1) * torch.sum((x11 - x12)**2)
        M22 = (1/1) * torch.sum((x21 - x22)**2)
        M12 = M21 = (1/4) * (
                  torch.sum((x11 - x21)**2) \
                + torch.sum((x12 - x21)**2) \
                + torch.sum((x11 - x22)**2) \
                + torch.sum((x12 - x22)**2)
                )

        expected_M = torch.tensor([[M11, M12], [M21, M22]])

        result_M = compute_mse_heatmap(embeddings_per_class)

        assert_tensor_eq(expected_M, result_M)


if __name__ == "__main__":
    unittest.main()

























