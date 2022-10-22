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

Testcases for the whitening transform.
"""
# Library imports:
import unittest
import torch
from torch import Tensor

# Local imports:
from rolim.whitening.whitening import whiten
from rolim.tools.testing import assert_tensor_eq
from rolim.tools.stats import sample_covar

# Allowed numerical error (distance) between theoretical
# and computed mean/covariance matrices.
ATOL = 0.1

class WhitenTestCase(unittest.TestCase):
    
    def test_random_3x3(self):
        inp = torch.rand((3,3), dtype=torch.float)
        self.run_test_whiten(inp)

    def test_fixed_4x5(self):
        inp = torch.tensor([
            [3, 4, 7.5, 1, 2],
            [8, 0, -10, -100, -1000],
            [4, 3, 2, 1, 1000],
            [8.1, 1, 2, 3, 4]],
            dtype=torch.float)
        self.run_test_whiten(inp)

    def run_test_whiten(self, input_vectors: Tensor, atol: float = ATOL):
        """
        Call `whiten()` on the input,
        and test if the output has:
        * The same shape as the input.
        * Mean 0 between vectors (element-wise mean).
        * Identity matrix covariance.
        """
        inp_shape = input_vectors.shape
        self.assertEqual(len(inp_shape), 2, msg="Input is not a 2D matrix")
        vector_len = inp_shape[0]
        
        try:
            result = whiten(input_vectors)
        except Exception as e:
            self.fail("whiten() resulted in an error:\n" + str(e))

        expected_mean = torch.zeros(vector_len)
        result_mean = torch.mean(result, dim=1) # Mean of each row
        assert_tensor_eq(expected_mean, result_mean, atol=atol)

        expected_covar = torch.eye(vector_len)
        result_covar = sample_covar(result)
        assert_tensor_eq(expected_covar, result_covar, atol=atol)

if __name__ == "__main__":
    unittest.main()
