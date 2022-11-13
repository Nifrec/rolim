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
import math
import itertools

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

    NOTE: the gradients for this computation are not tracked.
    """
    num_classes = len(embeddings_per_class)
    num_emb_per_class = embeddings_per_class[0].shape[0]
    emb_dim = embeddings_per_class[0].shape[1]
    for embeddings in embeddings_per_class:
        assert embeddings.shape == (num_emb_per_class, emb_dim)

    # exp_num_pairs_diagonal = \binom{num_emb_per_class}{2}
    # = num distinct pairs of images of same class.
    exp_num_pairs_diagonal = (math.factorial(num_emb_per_class))// \
        (2 * math.factorial(num_emb_per_class - 2))

    # exp_num_pairs_off = number distinct pairs of one embedding
    # class w, and an embedding from class z, where w!=z.
    exp_num_pairs_off = num_emb_per_class**2



    M = torch.zeros(size=(num_classes, num_classes),
                    dtype=torch.float,
                    requires_grad=False)

    # Compute upper off-diagonal entries.
    # Indices correspond to the math formula in `encoder_experiment.ipynb`.
    for w in range(num_classes-1):
        for z in range(w+1, num_classes):
            assert w < z
            num_pairs_seen = 0
            all_pairs = itertools.product(range(num_emb_per_class),
                                          range(num_emb_per_class))
            for (i, j) in all_pairs: 
                num_pairs_seen += 1
                x_wi = embeddings_per_class[w][i,:]
                x_zj = embeddings_per_class[z][j,:]
                M[w, z] += torch.sum((x_wi - x_zj)**2)
            assert num_pairs_seen == exp_num_pairs_off
            M[w, z] = M[w,z] / num_pairs_seen


    # Lower off-diagonal entries are symmetric with the upper ones!
    M += M.clone().T

    # Diagonal entries
    for z in range(num_classes):
        num_pairs_seen = 0
        for i in range(num_emb_per_class - 1):
            for j in range(i+1, num_emb_per_class):
                assert i < j
                num_pairs_seen += 1
                x_zi = embeddings_per_class[z][i,:]
                x_zj = embeddings_per_class[z][j,:]
                M[z, z] += torch.sum((x_zi-x_zj)**2)
        assert num_pairs_seen == exp_num_pairs_diagonal
        M[z,z] = M[z,z] / num_pairs_seen

    return M







































