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

This is a template file for new source-code files
in this project. 
TODO: substitute this placeholder with actual description.
"""
# Library imports:
import torch
from torch import Tensor

def sample_covar(vectors: Tensor, ddof: int=1) -> Tensor:
    r"""
    Compute the sample covariance of a collection of column vectors.

    Let $N$ be the length of each vector in `vectors`.$
    Then this computes:
    $$\frac{1}{N-\texttt{ddof}}\sum_{i=1}^N(V[i,:]-\mu)(V[i,:]-\mu)^{\top}$$
    where $\mu$ is the mean of `vectors`.

    Arguments:
    * vectors: matrix of column vectors, 
    * ddof: degrees of freedom lost during computation,
        subtracted from the sample size (see equation).

    Returns:
    * matrix `M` of shape `(len(v), len(v))`,
        such that `M[i, j]` is the covariance between
        samples at index `i` and samples at index `j`.
    """
    mean = torch.mean(vectors, dim=1)
    normalized = vectors - mean.reshape((-1, 1))
    sum_squares = normalized @ normalized.T
    sample_size = vectors.shape[1]
    assert sample_size != 0, "Cannot compute covariance of empty sample"
    return (1/(sample_size - ddof))*sum_squares

def square_vector(vec: Tensor) -> Tensor:
    """
    Compute the outer product of `vec` with itself.
    """
    vec = vec.view((-1, 1)) # Convert to column vector
    return vec @ vec.T