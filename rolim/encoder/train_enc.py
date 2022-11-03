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

Functions to:
* Train an encoder network on 1 batch of images.
* Infer an encoding from an encoder network,
"""
# Library imports:
import torch
from torch import Tensor

# Local imports:
from rolim.networks.architectures import AtariCNN
from rolim.whitening.whitening import whiten

def train_encoder_distance(encode_net: AtariCNN, 
                           batch: Tensor, 
                           use_whitening: bool,
                           optim: torch.optim.Optimizer,
                           run_checks: bool=True) -> Tensor:
    """
    Train a observation-to-latent-state-space encoder φ on
    one batch of N pairs of observations (images)
    (o_{1,1}, o_{1,2}, o_{2,1}, o_{2,2}, ..., o_{N,1}, o_{N,2}). 
    The function to minimize is the
    squared Euclidean distance between encodings of the pairs.
    So the loss is: 
       (1/N) Σ_{i=1}^N ||φ(ο_{i,1}) - φ(o_{i,2})||²

    Arguments:
    * encode_net: the CNN that performs the observation-space to latent-space
        encoding. This object will be modified in-place.
    * batch: Tensor of shape `(2*N, num_channels, height, width)`
        where `num_channels`, `height` and `width` describe the shape
        of each image, and the first axis ranges over all images of the batch.
        The images that are pairs are assumed to be directly after each other
        (i.e., `batch[0:], batch[1:]` is the first pair,
        `batch[2:], batch[3:]` the second, and so on).
    * use_whitening: flag whether the compose the encoder with
        a Whitening Transform before taken the loss.
        This passes the output of the encoder (AFTER encoding)
        though the Whitening Transform, BEFORE computing the loss.
    * optim: torch.optim.Optimizer, optimizer to use the gradients
        to update the `encode_net`.
    * run_checks: flag whether or not to check for dimension compatibility.
        Usefull for debugging, but may hamper running time performance.

    Returns:
    * loss: a singleton tensor containing the loss.
        (Still on the same device as the batch and the encoder).
    """
    optim.zero_grad()
    
    encoded_batch = encode(encode_net, batch, run_checks)
    if use_whitening:
        # whiten() expects a 2D matrix where the columns (2nd axis)
        # are distinct entries.
        # Batch is a 4D Tensor where the first axis ranges over the
        # distinct images.

        # num_images=batch.shape[0]
        batch_as_2d = torch.flatten(batch).T
        # view_2d = encoded_batch.view(shape=torch.Size(num_images, -1))
        whitened_encoding = whiten(batch_as_2d)
        whitened_encoding = whitened_encoding.T.view_as(encoded_batch)
        encoding = whitened_encoding
    else:
        encoding = encoded_batch

    loss = pairwise_mse(encoding)
    loss.backward()
    optim.step()
    return loss

def encode(encode_net: AtariCNN, batch: Tensor, run_checks: bool) -> Tensor:
    """
    Encode a batch with a given encoder network.
    Optionally run assertions checking the compatibility
    between the batch's shape and the network's expected input shape.
    """
    if run_checks:
        (_, channels, height, width) = batch.shape
        assert channels == encode_net.channels
        assert height == encode_net.height
        assert width == encode_net.width
    return encode_net(batch)

def pairwise_mse(batch: Tensor) -> Tensor:
    """
    Given a batch of N pairs of vectors 
        (v_{1,1}, v_{1,2}, v_{2,1}, v{2,2}, ..., v_{N,1}, v_{N,2}),
    compute
       (1/N) Σ_{i=1}^N ||v_{i,1} - v_{i,2}||²

    Arguments:
    * batch: matrix of shape `(2*N, dim)` 
        of 2N pairs of vectors of length `dim`.
        Consecutive entries are considered pairs.

    Returns:
    * Mean squared Euclidean distance between the vectors in each pair.
    """
    raise NotImplementedError("TODO: write testcase first!")

