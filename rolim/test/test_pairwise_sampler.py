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

Testcases for rolim/encoder/pairwise_sampler.py
"""
# Library imports:
import unittest
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
from torch.utils.data import DataLoader

# Local imports:
from rolim.encoder.pairwise_sampler import (PairWiseBatchSampler,
    load_cifar_10_dataset)
from rolim.settings import (RNG, CIFAR10_CHANNELS, CIFAR10_HEIGHT,
                            CIFAR10_WIDTH)

class TestPairwiseSampler(unittest.TestCase):

    def setUp(self):
        self.train, self.test = load_cifar_10_dataset(download=True)
        self.batch_size = 10
        self.epoch_size = 30
        self.batch_sampler = PairWiseBatchSampler(self.train, RNG,
                                                  self.batch_size,
                                                  self.epoch_size)

    def test_epoch_size(self):
        """
        With batches of 10 pairs and an epoch size of 30 pairs,
        we should be able to query exactly 3 batches from the batch_sampler.
        """
        iterator = iter(self.batch_sampler)
        for _ in range(3):
            next(iterator)
        with self.assertRaises(StopIteration):
            next(iterator)


    def test_with_dataloader(self):
        """
        Test compatability with a PyTorch DataLoader
        """
        dataloader = DataLoader(self.train, batch_sampler=self.batch_sampler)
        iterator = iter(dataloader)
        batch_samples, batch_labels= next(iterator)
        expected_shape = (self.batch_size, CIFAR10_CHANNELS, CIFAR10_HEIGHT,
                          CIFAR10_WIDTH)
        self.assertSequenceEqual(batch_samples.shape, expected_shape, 
                                 msg=f"Sampled batch of shape: "
                                 f"{batch_samples.shape}")
        expected_shape_labels = (self.batch_size,)
        self.assertSequenceEqual(batch_labels.shape, expected_shape_labels, 
                                 msg=f"Sampled batch with labels of shape: "
                                 "{batch_labels.shape}")

if __name__ == "__main__":
    unittest.main()




















