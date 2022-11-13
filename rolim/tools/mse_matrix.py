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

from typing import Optional, Any
import math
import itertools
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import Colormap
from matplotlib.colorbar import Colorbar

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

def plot_heatmap(heatmap: Tensor | np.ndarray,
                 ax: Optional[Axes] = None,
                 xtick_labels: Optional[list[str]] = None,
                 ytick_labels: Optional[list[str]] = None,
                 add_colorbar: bool = True,
                 cmap: str | Colormap="plasma_r",
                 imshow_kwargs: dict[str, Any] = {},

                 ) -> tuple[Axes, Figure | None, Colorbar | None]:
    """
    Plot a heatmap using matplotlib.

    Arguments:
    * heatmap: matrix to visualize as a heatmap.
    * ax: optional `Axes` instance to plot the heatmap on.

    Returns:
    * ax: `Axes` instance heatmap is plotted on.
        If the input `ax` is not `None`,
        this is the same as the input argument,
        otherwise a new instance.
    * fig: `Figure` in which the output `ax` resides.
        This is `None` if an input `ax` is given.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1) #type: ignore
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)
    else:
        fig = None

    heatmap = __convert_heatmap_if_needed(heatmap)

    num_rows, num_cols = heatmap.shape
    __add_ticklabels(ax, num_rows, num_cols, xtick_labels, ytick_labels)    

    image = ax.imshow(heatmap, cmap=cmap, **imshow_kwargs)

    if add_colorbar:
        colorbar = ax.figure.colorbar(image, ax=ax) #type: ignore
    else:
        colorbar = None
    ax.set_xlabel("Class") 
    ax.set_ylabel("Class") 

    # Add numbers to cell to give the quantitative value,
    # rather than only a colour.
    __add_cell_values(heatmap, num_rows, num_cols, ax)

    return (ax, fig, colorbar)

def __convert_heatmap_if_needed(heatmap: Tensor | np.ndarray
                                ) -> np.ndarray:
    if isinstance(heatmap, Tensor):
        heatmap = heatmap.cpu().numpy()
    if len(heatmap.shape) != 2:
        raise ValueError("Input heatmap must be a 2D matrix, but got "
                         f"{len(heatmap.shape)} dimensions.")
    assert (isinstance(heatmap, np.ndarray))
    return heatmap

def __add_ticklabels(ax: Axes, num_rows: int, num_cols: int,
                     xtick_labels: Optional[list[str]],
                     ytick_labels: Optional[list[str]]):
    # The pywright linter thinks Axes don't have these methods.
    # According to the matplotlib documentation they do...
    ax.set_xticks(np.arange(num_cols))  #type: ignore
    if xtick_labels is not None:
        if len(xtick_labels) != num_cols:
            raise ValueError("Number of xtick_labels given does not"
                             " match the number of columns of the heatmap.")
        ax.set_xticks(np.arange(num_cols))  #type: ignore
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right") #type: ignore

    ax.set_yticks(np.arange(num_rows))  #type: ignore
    if ytick_labels is not None:
        if len(ytick_labels) != num_rows:
            raise ValueError("Number of ytick_labels given does not"
                             " match the number of rows of the heatmap.")
        ax.set_yticklabels(ytick_labels)    #type: ignore


def __add_cell_values(heatmap: np.ndarray, num_rows: int, num_cols: int,
                      ax: Axes):
    min_val = np.min(heatmap)
    max_val = np.max(heatmap)
    num_values = heatmap.size
    assert(isinstance(num_values, int))
    text_colors = cm.summer(np.linspace(0, 1, num_values)) #type: ignore

    for row in range(num_rows):
        for col in range(num_cols):
            cell_value = heatmap[row, col] 
            label=f"{cell_value:.1e}"
            colour_idx = round(((cell_value - min_val)/(max_val-min_val)) 
                               * (num_values-1))
            ax.text(col, row, label, ha="center", va="center",
                color=text_colors[colour_idx], rotation=30,
                    fontsize="x-small")
































