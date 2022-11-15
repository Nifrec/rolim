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

Auxiliary functions for dealing with matrices of pairs of vectors.
"""
# Library imports:
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Optional

def get_odd_even_vectors(batch: Tensor) -> tuple[Tensor, Tensor]:
    """
    Return the odd-indiced rows and the even-indiced rows
    of a matrix `batch` as two separate matrices.

    Arguments:
    * batch: matrix of shape `(2*N, M)` where `N` and `M`
        may be any nonzero natural number.

    Returns:
    * odds, evens: matrix of the odd rows of `batch`,
        and matrix with the even rows of `batch`.
        Both have shape `(N, M)`.
    """
    num_entries = batch.shape[0]
    even_indices = torch.arange(0, num_entries, 2)
    odd_indices = torch.arange(1, num_entries, 2)
    even_entries = torch.index_select(batch, dim=0, index=even_indices)
    odd_entries = torch.index_select(batch, dim=0, index = odd_indices)
    return odd_entries, even_entries







































