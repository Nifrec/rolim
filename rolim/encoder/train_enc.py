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
* Infer an encoding from an encoder network.
* Train an encoder on the CIFAR10 dataset.
"""
# Library imports:
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as vision
from torchvision.transforms.functional import to_tensor
from typing import Literal

# Local imports:
from rolim.networks.architectures import AtariCNN
from rolim.settings import (CIFAR10_CHANNELS, CIFAR10_DIR, CIFAR10_HEIGHT, 
                            CIFAR10_LATENT_DIM, CIFAR10_NUM_BATCHES, 
                            CIFAR10_WIDTH, CIFAR10_BATCH_SIZE, DEVICE, 
                            REDOWNLOAD_DATASET, RNG)
from rolim.tools.pairs import get_odd_even_vectors
from rolim.whitening.whitening import dw_mse_loss, whiten
from rolim.encoder.pairwise_sampler import (
        CorrelatedBatchSampler, load_cifar_10_dataset, PairWiseBatchSampler)

TrainingMethod = Literal["distance", "decoder", "predictor"]
def train_encoder(method: TrainingMethod,
                  distance_loss: Literal["MSE", "W-MSE", "DW-MSE"],
                  run_checks: bool = False,
                  num_batches: int = CIFAR10_NUM_BATCHES,
                  batch_size: int = CIFAR10_BATCH_SIZE,
                  correlated_batches: bool = False
                  ) -> tuple[AtariCNN, list[float]]:
    """
    Train an `AtariCNN` on the CIFAR10 dataset.
    Return the trained network and the losses per epoch.

    Arguments:
    * method: one of
        - "distance": optimise the network to minimize the distance
            between encodings related images (i.e., images of the same
            class). 
        - "decoder": optimise the network to minimize the error
            in a decoder (i.e., make `decoder(encoder(image))`
            as close to the input image as possible).
        - "predictor": optimize the network to minimize the error
            in a classifier network (i.e., make `classifier(encoder(image))`
            return the class of the image.
    * loss: indication which loss function should be used,
        in case `method` is `"distance"`:
        * MSE: sum of squared distances between vectors of the same pair.
        * W-MSE: same as MSE, but with the Whitening Transform applied to
            the batch AFTER encoding BEFORE computing the loss.
        * DW-MSE: same as MSE, but normalize the pair-wise difference
            vectors by subtracting the mean difference-vector
            and multiplying by the inverse covariance matrix.
    * run_checks: flag whether to run assertions for tensor shapes.
    * num_batches: number of batches of images to train on before the
        training stops.
    * batch_size: number of pairs of images in each minibatch.
    * correlated_batches: flag whether all pairs in each batch
        are of the same class label. If True,
        one class is randomly chosen for each batch.
        If False, a class is randomly chosen for each pair within the batch.

    Returns:
    * encoder: trained convolutional network.
    * losses: list containing the loss of each batch.
    """
    if method == "decoder":
        raise NotImplementedError("Training with decoder not yet implemented")
    elif method == "predictor":
        raise NotImplementedError("Training with predictor not yet implemented")

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    encoder = AtariCNN(channels=CIFAR10_CHANNELS, height=CIFAR10_HEIGHT, 
                       width=CIFAR10_WIDTH, out_size=CIFAR10_LATENT_DIM)
    optimizer = torch.optim.Adam(params=encoder.parameters())

    dataloader = __load_dataset(batch_size, num_batches, correlated_batches)
    losses: list[Tensor] = []
    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(DEVICE) 
        loss = _train_1_batch_distance_minimization(encoder, batch_images,
                                                    distance_loss,
                                                    optimizer, run_checks)
        losses.append(loss)

    losses_as_floats = [loss.item() for loss in losses]
    return (encoder, losses_as_floats)

def __load_dataset(batch_size: int,
                   num_batches: int,
                   correlated_batches: bool
                   ) -> DataLoader:
    trainset = vision.datasets.CIFAR10(root=CIFAR10_DIR, train=True, 
                                       download=REDOWNLOAD_DATASET, 
                                       transform=to_tensor)
    if not correlated_batches:
        sampler_type = PairWiseBatchSampler
    else:
        sampler_type = CorrelatedBatchSampler

    batch_sampler = sampler_type(trainset, RNG, batch_size=batch_size,
                                 epoch_size=batch_size*num_batches)
    dataloader = DataLoader(trainset, batch_sampler=batch_sampler,
                            num_workers=1)
    return dataloader

def _train_1_batch_distance_minimization(encode_net: AtariCNN, 
                           batch: Tensor, 
                           loss_fun: Literal["MSE", "W-MSE", "DW-MSE"],
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
    * loss_fun: indication which loss function should be used.
        * MSE: sum of squared distances between vectors of the same pair.
        * W-MSE: same as MSE, but with the Whitening Transform applied to
            the batch AFTER encoding BEFORE computing the loss.
        * DW-MSE: same as MSE, but normalize the pair-wise difference
            vectors by subtracting the mean difference-vector
            and multiplying by the inverse covariance matrix.
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
    if loss_fun == "DW-MSE":
        loss = dw_mse_loss(encoded_batch)
    elif loss_fun == "W-MSE":
        # whiten() expects a 2D matrix where the columns (2nd axis)
        # are distinct entries.
        # The encoder outputs a vector of shape (batch_size, out_size).
        # Here the *rows* are the distinct entries, so we need to
        # transpose before and after taking the whitening transform.
        whitened_encoding = whiten(encoded_batch.T)
        whitened_encoding = whitened_encoding.T
        encoding = whitened_encoding
        loss = pairwise_mse(encoding)
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
    * Mean squared Euclidean distance between the vectors in each pair,
        as 0-dimensional Tensor.
    """
    num_entries = batch.shape[0]
    odd_entries, even_entries = get_odd_even_vectors(batch) 
    loss = torch.sum(torch.pow(even_entries-odd_entries, 2))/(num_entries//2)
    return loss
