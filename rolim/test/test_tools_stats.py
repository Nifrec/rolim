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
import numpy as np

# Local imports:
from rolim.whitening.whitening import whiten
from rolim.tools.testing import assert_tensor_eq
from rolim.tools.stats import get_all_diag_entries, get_all_upper_entries, get_diagonal_entries, get_upper_triangular_entries, sample_covar

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

class DiagonalEntriesTestCase(unittest.TestCase):
    """
    Testcases for the functions that get diagonal entries of matrices.
    """

    def test_1_matrix(self):
        mat = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
        expected = np.array([1, 5, 9])
        result = get_diagonal_entries(mat)
        np.testing.assert_allclose(result, expected)

    def test_multiple_matrices(self):
        mat1 = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
        mat2 = np.array([
            [3, 4],
            [5, 6]])
        mat3 = np.eye(5)
        matrices = (mat1, mat2, mat3)
        expected = np.array([1, 5, 9] + [3, 6] + [1]*5)
        result = get_all_diag_entries(matrices)
        np.testing.assert_allclose(result, expected)

    def test_pytorch_input(self):
        mat1 = torch.tensor([[1, 2], [3, 4]])
        mat2 = torch.tensor([[5, 6], [7, 8]])
        matrices = (mat1, mat2)

        single_result = get_diagonal_entries(mat1)
        self.assertIsInstance(single_result, np.ndarray)
        np.testing.assert_allclose(single_result, np.array([1, 4]))

        mult_result = get_all_diag_entries(matrices)

        self.assertIsInstance(mult_result, np.ndarray)
        np.testing.assert_allclose(mult_result, np.array([1, 4, 5, 8]))

class UpperTriangEntriesTestCase(unittest.TestCase):
    """
    Testcases for the functions that get upper-triangular
    entries of matrices.

    The tests use the same matrices as for the diagonal function tests
    in `DiagonalEntriesTestCase`.
    """

    def test_1_matrix(self):
        mat = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
        expected = np.array([2, 3, 6])
        result = get_upper_triangular_entries(mat)
        np.testing.assert_allclose(result, expected)

    def test_multiple_matrices(self):
        mat1 = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
        mat2 = np.array([
            [3, 4],
            [5, 6]])
        mat3 = np.eye(5)
        matrices = (mat1, mat2, mat3)
        expected = np.array([2, 3, 6] + [4] + [0]*10)
        result = get_all_upper_entries(matrices)
        np.testing.assert_allclose(result, expected)

    def test_pytorch_input(self):
        mat1 = torch.tensor([[1, 2], [3, 4]])
        mat2 = torch.tensor([[5, 6], [7, 8]])
        matrices = (mat1, mat2)

        single_result = get_upper_triangular_entries(mat1)
        self.assertIsInstance(single_result, np.ndarray)
        np.testing.assert_allclose(single_result, np.array([2]))

        mult_result = get_all_upper_entries(matrices)

        self.assertIsInstance(mult_result, np.ndarray)
        np.testing.assert_allclose(mult_result, np.array([2, 6]))
if __name__ == "__main__":
    unittest.main()


































