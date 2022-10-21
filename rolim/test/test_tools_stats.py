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

Testcases for rolim/tools/stats.py
"""
# Library imports:
import unittest
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor

# Local imports:
from rolim.whitening.whitening import whiten
from rolim.tools.testing import assert_tensor_eq
from rolim.tools.stats import sample_covar

class SampleCovarTestCase(unittest.TestCase):

    def test_covar_1(self):
        """
        Corner case: zero tensors have covariance 0.
        """
        inp_sample = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
            ], dtype=torch.float)
        output = sample_covar(inp_sample)
        expected_size = (int(inp_sample.shape[0]),)*2
        expected = torch.zeros(expected_size,
                               dtype=torch.float)
        assert_tensor_eq(expected, output)

    def test_covar_2(self):
        """
        Corner case: single value tensor.
        Uses non-standard ddof
        """
        inp_sample = torch.tensor([[1.3]])
        expected = torch.tensor([[0.]])
        output = sample_covar(inp_sample, ddof=0)
        assert_tensor_eq(expected, output)

    def test_covar_3(self):
        """
        Base case: very small example.
        """
        inp_sample = torch.tensor([
            [1, 2], [5, 3]], dtype=torch.float)
        expected_covar = torch.tensor([
            [0.5, -1], [-1, 2]], dtype=torch.float)
        output = sample_covar(inp_sample)
        assert_tensor_eq(expected_covar, output)

    def test_covar_empirical(self):
        """
        Sample a lot of vectors from a standard normal distribution,
        then the sample covariance shouldapproximately be the identity matrix.
        """
        mean=torch.tensor([0, 0, 0], dtype=torch.float)
        covar = torch.eye(3, 3)
        distr = MultivariateNormal(mean, covar)
        # Sample a matrix with 100 columns, each a 3-element random vector
        sample = distr.sample(sample_shape=torch.Size([100])).T
        output = sample_covar(sample)
        assert_tensor_eq(covar, output, atol=1)

if __name__ == "__main__":
    unittest.main()
