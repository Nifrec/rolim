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

Testcases for rolim/tools/pairs.py
"""
# Library imports:
import unittest
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor

# Local imports:
from rolim.tools.pairs import get_odd_even_vectors
from rolim.tools.testing import assert_tensor_eq

class GetOddEvenVectorsTestCase(unittest.TestCase):

    def test_basic(self):
        matrix = torch.tensor([
            [1, 2, 3],
            [3, 2, 1],
            [1, 1, 1],
            [2, 2, 2]
            ])
        expected_evens = torch.tensor([
            [1, 2, 3],
            [1, 1, 1]
            ])
        expected_odds = torch.tensor([
            [3, 2, 1],
            [2, 2, 2]
            ])
        actual_odds, actual_evens = get_odd_even_vectors(matrix)

        assert_tensor_eq(expected_odds, actual_odds)
        assert_tensor_eq(expected_evens, actual_evens)

if __name__ == "__main__":
    unittest.main()
