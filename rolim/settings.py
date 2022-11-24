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

Declarations of global constants.
"""
import os
import numpy as np
import torch

# Default regularization constant to make a sample covariance matrix
# better conditioned in the whitening transform.
WHITEN_REG_EPS = 0.001

# The path to the `rolim` directory, containing among others
# this `settings.py` file.
PROJECT_ROOT_DIR = os.getcwd()

# Directory for storing the results of performing multiple
# runs of training and evaluating an encoder.
MULT_RUNS_DIR = os.path.join(PROJECT_ROOT_DIR, "encoder", "runs")

# Directory to store the CIFAR10 dataset images.
CIFAR10_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "cifar_10")
# CIFAR10 uses 32×32 RGB images, so:
CIFAR10_WIDTH = 32
CIFAR10_HEIGHT = 32
CIFAR10_CHANNELS = 3
# Dimension of the latent space representation for the CIFAR10-images-encoder.
CIFAR10_LATENT_DIM = 32
# Number of batches to train the encoder on (default value)
CIFAR10_NUM_BATCHES = 3000
# Number of pairs of images in each batch of encoder training (default value)
CIFAR10_BATCH_SIZE = 10
# Class labels, in order corresponding to the label indices used:
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_NUM_CLASSES = len(CIFAR10_CLASSES)

# Whether or not to redownload the CIFAR10 dataset.
# Set this to False when working offline, otherwise it throws errors.
REDOWNLOAD_DATASET = False

# Numpy random number generator used thoughout this project.
# Could be used to fix the seed globally.
# Note that also other random functions from the `random` and `torch`
# libraries are used.
RNG = np.random.default_rng()

# This is currently the default value that sklearn uses.
TSNE_DEFAULT_PERPLEXITY = 30.0

if torch.cuda.is_available(): 
    DEVICE = torch.device("cuda")
else: 
    print("CUDA not found. Using the CPU for PyTorch tensors")
    DEVICE = torch.device("cpu")
