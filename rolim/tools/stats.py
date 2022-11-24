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

Statistics-related auxiliary functions, such as
computing the sample covariance of a set of vectors.
"""
# Library imports:
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Optional, Iterable
import numpy as np
from numpy.typing import NDArray

# Local imports:
from rolim.settings import WHITEN_REG_EPS

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

def randomized_multinormal_distr(dim: int,
                                 reg_eps: float=WHITEN_REG_EPS,
                                 parent_mean : Optional[Tensor] = None,
                                 covar_scale: Optional[float] = 1.0
                                 ) -> MultivariateNormal:
    """
    Sample a random mean vector and (the covariance of) a random matrix
    from a standard multinormal distribution (which can be
    modified with the optional arguments),
    and return a multinormal distribution with this mean and covariance.

    Arguments:
    * dim: dimension of the output distribution
        (i.e. length of sampled vectors, length of the distribution's mean,
        and the covariance is a `dim×dim` matrix).
    * reg_eps: number in [0, 1], proportion to which the
        sampled covariance matrix is blended with an identity matrix
        (used for numerical stability).
    * parent_mean: mean for the distribution where the mean for the output
        distribution is sampled from. By default the zero vector is used.
    * covar_scale: multiplier for the covariance of the distribution
        from which the mean and covariance are generated.

    Returns:
    * Instantiated mulinormal distribution object.


    NOTE: randomizing the input of the optional arguments seem to have more
    effect than the random sampling executed within this function.
    """
    if parent_mean is None:
        parent_mean = torch.zeros(dim)
    standard_normal = MultivariateNormal(
            loc=parent_mean,
            covariance_matrix=covar_scale*torch.eye(dim))
    mean = standard_normal.sample(torch.Size((1,)))
    # Sample 10 times as many vectors as the dimension of the covar matrix:
    # a bigger sample makes it more likely that the numerical
    # sample covariance is Positive Definite.
    random_mat = standard_normal.sample(torch.Size((10*dim,))).T
    assert(random_mat.shape == (dim, 10*dim))
    covar = sample_covar(random_mat)
    covar = (1-reg_eps) * covar + reg_eps * torch.eye(dim)

    return MultivariateNormal(loc=mean, covariance_matrix=covar)


def get_diagonal_entries(mat: Tensor | NDArray) -> NDArray:
    return np.asarray(np.diag(mat))

def get_all_diag_entries(matrices: Iterable[Tensor|NDArray]) -> NDArray:
    """
    Return the concatenated diagonal entries of multiple matrices
    as one 1D array.
    """
    return np.concatenate([get_diagonal_entries(mat) for mat in matrices])

def get_upper_triangular_entries(mat: Tensor | NDArray) -> NDArray:
    """
    Return the above-diagonal entries of a square matrix as a 1D array.
    (By inputting the transposed matrix, this also function
    can also be used to get the lower triangular entries).

    Note: the diagonal entries are NOT included.
    """
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Input matrix is not square: size: {mat.shape}")

    indices = np.triu_indices(mat.shape[0], 1)
    return np.asarray(mat[indices])

def get_all_upper_entries(matrices: Iterable[Tensor|NDArray]) -> NDArray:
    """
    Return the concatenated above-diagonal entries of multiple square matrices
    as one 1D array.
    The diagonal entries are not included.
    """
    output = np.concatenate([get_upper_triangular_entries(mat) for mat in
                           matrices])
    return np.asarray(output)



























