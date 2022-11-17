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

Implementation of the "Whitening transform",
a pre-processing step between the output of
encoder (which is a latent representation of an observation)
and the computation of the MSE to optimize the encoder.
Note that it is not part of the encoder itself:
it is only used for computing the MSE for backpropagation.
"""

# Library imports:
import torch
from torch import Tensor
from warnings import warn
from rolim.tools.pairs import get_odd_even_vectors

# Local imports:
from rolim.tools.stats import sample_covar
from rolim.tools.testing import assert_tensor_eq
from rolim.settings import WHITEN_REG_EPS

MSE = torch.nn.MSELoss()

def whiten(vectors: Tensor, 
           reg_eps: float = WHITEN_REG_EPS,
           do_validate: bool=False) -> Tensor:
    """
    Given a matrix `vectors` of column-vectors,
    subtract the mean of the vectors from each vector,
    and 'divide by the standard deviation'
    (left-multiply each mean-subtracted vector with the matrix W,
    which has the property that W.t@W = Σ^{-1},
    the inverse of the sample covariance of `vectors`.

    The implementation details are given in the appendix of
    Aliaksandr Siarohin, Enver Sangineto & Nicu Sebe (2019)
    'Whitening and Colouring batch transform for GANs'.

    Arguments:
    * vectors: matrix of column-vectors to apply the
        whitening transform to.
        Must be of floating datatype.
    * reg_eps: small positive constant to add to the
        diagonal entries of the covariance matrix of `vectors`
        to make it better conditioned for Cholesky decomposition
        (it needs to be positive definite for that,
        but numerical error may not make it so).
    * do_validate: flag whether to run assertions
        on the correctness of decompositions and versions.

    Returns:
    * Copy of `vectors`, with the whitening transform applied
        to each column-vector.
    """
    mean = torch.mean(vectors, dim=1).reshape((-1, 1))
    raw_covar = sample_covar(vectors, ddof=1)
    identity = torch.eye(raw_covar.shape[0])
    reg_covar = (1 - reg_eps) * raw_covar + reg_eps*identity
    # The cited paper *first* runs the decomposition,
    # and thereafter inverts the decomposition-output.
    # The main text in papers make it look like done the other way around.
    w = torch.linalg.cholesky(reg_covar)
    w_inv = torch.linalg.solve(w, identity)
    if do_validate:
        _validate_whiten(w, w_inv, vectors)
    return w_inv @ (vectors - mean)

def _validate_whiten(w: Tensor, w_inv: Tensor, reg_covar: Tensor):
    # Check if the decomposition is correct
    assert_tensor_eq(w.T @ w, reg_covar)
    # Check if w is indeed invertible
    identity = torch.eye(w.shape[0])
    assert_tensor_eq(w_inv @ w, identity)

def whiten_naive(vectors: Tensor, do_validate: bool=False) -> Tensor:
    """
    WARNING: this is the naive implementation
    of the whitening transorm, following the
    description in the main text of papers.
    It generally fails to properly invert a covariance matrix.
    Use the function `whiten` for a better alternative.

    Given a matrix `vectors` of column-vectors,
    subtract the mean of the vectors from each vector,
    and 'divide by the standard deviation'
    (left-multiply each mean-subtracted vector with the matrix W,
    which has the property that W.t@W = Σ^{-1},
    the inverse of the sample covariance of `vectors`.
    """
    mean = torch.mean(vectors, dim=1).reshape((-1, 1))
    covar = sample_covar(vectors, ddof=1)
    identity = torch.eye(covar.shape[0])
    covar_inv = torch.linalg.solve(covar, identity)
    if do_validate and not torch.allclose(identity, covar_inv @ covar):
        warn("Unable to invert the matrix:\n"+str(covar)
                           + "\n Resulting product Σ^{-1} @ Σ:\n"
                           + str(covar_inv @ covar))
    w = torch.linalg.cholesky(covar_inv)
    return w @ (vectors - mean)
    
def compute_whiten_error(whiten_output: Tensor
                         ) -> tuple[float, float]:
    """
    Compute the error of whitening with respect
    to normalizing the mean and the covariance.

    Arguments:
    * whiten_output: a whitened matrix of column vectors.

    Returns:
    * mean_error: MSE between actual mean and expected mean (= zero vector).
    * covar_error: MSE between actual sample covariance
        and expected (=identity matrix) covariance.
    """
    dim = whiten_output.shape[0]
    expected_mean = torch.zeros(torch.Size((dim, 1)))
    expected_covar = torch.eye(dim)

    output_mean = torch.mean(whiten_output, dim=1).reshape((dim, 1))
    output_covar = sample_covar(whiten_output)

    mean_mse = MSE(output_mean, expected_mean).item()
    covar_mse = MSE(output_covar, expected_covar).item()

    return (mean_mse, covar_mse)


def dw_mse_loss(batch: Tensor) -> Tensor:
    r"""
    Given a batch of pairs of embeddings from an encoder network,
    compute the difference-whitening mean-squared error
    (DW-MSE).
    If the inputs are N pairs of vectors
        Z = [z_1, z_2, z_3, z_4, ..., z_{2N-1}, Z_{2N}]
    then the DW-MSE loss is defined as:
        L := (1/N) \sum_{i=1}^N (h_i - μ)^T Σ^{-1} (h_i - μ)
    where
        H := [h_1, h_2, ..., h_n]
          = [z_1-z_2, z_3-z_4, ..., z_{2N-1} - z_{2N}]
        μ := mean(h_1, h_2, ..., h_n)
        Σ := sample_covariance(h_1, h_2, ..., h_n)

    Arguments:
    * batch: Tensor of shape `(2*N, dim)` where `N` is the number
        of pairs of embeddings, and `dim` is the length of each
        embedding-vector.

    Returns:
    * loss: 0-dimensional vector according to the formula above.
    """
    num_samples = batch.shape[0]
    num_pairs = num_samples // 2
    if num_samples % 2 != 0:
        raise ValueError("Input batch must consists of pairs of vectors,"
                         "but got odd number of vectors.")

    odds, evens = get_odd_even_vectors(batch)
    H = odds - evens
    covar = sample_covar(H.T) # sample_covar expects column vectors
    mean = torch.mean(H, dim=0)
    
    # torch.linalg.solve() is supposed to be faster and more numerically
    # stable than torch.inv().
    loss = (H - mean) @ torch.linalg.solve(covar, 
                                             (H - mean).T@torch.eye(num_pairs))
    assert(len(loss) == num_pairs)

    loss = (1 / num_pairs) * torch.sum(loss)
    return loss



































