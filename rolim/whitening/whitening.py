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

def whiten(vectors: Tensor) -> Tensor:
    """
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
    # assert_tensor_eq(identity, covar_inv @ covar)
    w = torch.linalg.cholesky(covar_inv)
    return w @ (vectors - mean)
    


