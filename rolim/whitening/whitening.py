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

# Local imports:
from rolim.tools.stats import sample_covar
from rolim.tools.testing import assert_tensor_eq
from rolim.settings import WHITEN_REG_EPS

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
        to make it better conditioned for inversion
        (a needed step to compute the whitening transform).
    * do_validate: flag whether to run assertions
        on the correctness of decompositions and versions.

    Returns:
    * Copy of `vectors`, with the whitening transform applied
        to each column-vector.
    """
    mean = torch.mean(vectors, dim=1)
    raw_covar = sample_covar(vectors, ddof=1)
    identity = torch.eye(raw_covar.shape[0])
    reg_covar = (identity - reg_eps) @ raw_covar + reg_eps*identity
    # The cited paper *first* runs the decomposition,
    # and thereafter inverts the decomposition-output.
    # The main text in papers make it look like done the other way around.
    w = torch.linalg.cholesky(reg_covar)

    if do_validate:
        _validate_whiten(w, vectors)

    # The output of the Cholesky transform is always lower-triangular
    return torch.triangular_solve(w, vectors - mean, upper=False)

def _validate_whiten(w: Tensor, vectors: Tensor):
    # Check if the decomposition is correct
    assert_tensor_eq(w @ w.T, vectors)
    # Check if w is indeed invertible
    identity = torch.eye(w.shape[0])
    w_inv = torch.triangular_solve(w, identity,
                                   upper=False),
    assert_tensor_eq(w_inv @ w, identity)

def whiten_naive(vectors: Tensor) -> Tensor:
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
    mean = torch.mean(vectors, dim=1)
    covar = sample_covar(vectors, ddof=1)
    identity = torch.eye(covar.shape[0])
    covar_inv = torch.linalg.solve(covar, identity)
    if not torch.allclose(identity, covar_inv @ covar):
        raise RuntimeError("Unable to invert the matrix:\n"+str(covar)
                           + "\n Resulting product Σ^{-1} @ Σ:\n"
                           + str(covar_inv @ covar))
    w = torch.linalg.cholesky(covar_inv)
    return w @ (vectors - mean)
    


