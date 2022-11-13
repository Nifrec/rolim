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

Function to compute a 'MSE heatmap'
given a set containing `n` embeddings for each of `m` classes.
"""
# Library imports:
import torch
from torch import Tensor

# Local imports:


def compute_mse_heatmap(embeddings_per_class: list[Tensor]) -> Tensor:
    """
    Compute a symmetric matrix $M$ whose entries $M_{w,z}$ are the
    average squared distance between embeddings of class $w$ and $z$
    respectively.

    See the notebook `encoder_experiment.ipynb`,
    section `### Heatmap`, for the mathematical formula.

    Arguments:
    * embeddings_per_class: list containing a matrix for each class.
        Each matrix should have the same shape,
        which is `num_embeddings × embedding_dim`
        where `num_embeddings` are the number of image-embeddings
        per class, and `embedding_dim` is the length of
        an individual embedding vector.

    Returns:
    * `M`: a symmetric  matrix of shape `num_classes × num_classes`
        whose entries are as described above.
    """
    raise NotImplementedError("TODO")







































