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

Helper functions and classes to:
    * Download the CIFAR10 dataset
    * Sample batches of images, such that each batch consists
        of pairs with the same label.

Note 1: The file `rolim/notebooks/cifar_10.ipynb` was an initial prototype for
    the content of this file. This file hosts the newest version.

Note 2: The code in this file is probably compatible with multiple
    datasets from `tochvision`. However, the generic class `VisionDataset`
    raised typing errors, apparently not all fields used by `CIFAR10`
    are necessarily present in all tochvision datasets.
    Hence the type-hints suggest that the dataset must be `CIFAR10`.
"""
# Library imports:
import numpy as np
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision as vision
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor
# Local imports:
from rolim.settings import CIFAR10_CLASSES, CIFAR10_DIR, CIFAR10_NUM_CLASSES, RNG

def load_cifar_10_dataset(save_dir: str = CIFAR10_DIR,
                          download=False
                          ) -> tuple[CIFAR10, CIFAR10]:
    """
    Load/download the cifar_10 dataset from/to the provided
    directory, and return the trainset and testset
    as PyTorch datasets.
    """
    trainset = vision.datasets.CIFAR10(root=save_dir, train=True, 
                                       download=download, transform=to_tensor)
    testset = vision.datasets.CIFAR10(root=save_dir, train=False, 
                                      download=download, transform=to_tensor)
    return (trainset, testset)

class PairSampler(Sampler):
    """
    Class for sampling indices, such that in the sequence
    [i_1, i_2, i_3, i_4, ...],
    each pair (i_1, i_2), (i_3, i_4), (i_5, i_6) correspond
    to pictures of the same class.
    For each pair, the classes is uniformly randomly sampled.
    After sampling a class, the indices of the pair are uniformly 
    randomly sampled from all images of the chosen class.
    All sampling is done with replacement.
    
    NOTE: the __iter__ method returns 1 index at a time.
    One needs to draw 2 consecutive 
    elements from the iterator to obtain a pair.
    """

    def __init__(self, 
                dataset: CIFAR10,
                rng: np.random.Generator, 
                epoch_size:int):
        super().__init__(dataset)
        self.__dataset = dataset
        # List that maps the index of a class to a list of
        # image indices of that class.
        self.__indices_per_class: list[list[int]] = get_indices_per_class(dataset)
        self.__num_classes: int = len(dataset.classes)
        self.__rng = rng

        if epoch_size % 2 != 0:
            raise ValueError(f"Epoch size must be even, got: {epoch_size}")
        self.__epoch_size = epoch_size
        
        assert self.__num_classes == len(self.__indices_per_class)

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def epoch_size(self) -> int:
        """
        Number of images returned before iteration halts.
        """
        return self.__epoch_size

    def sample_from_class_idx(self, class_idx: int) -> int:
        """
        Randomly sample the index of an image whose class has the given
        index in the dataset's list of class labels.
        """
        all_indices = self.__indices_per_class[class_idx]
        idx = self.__rng.choice(all_indices)
        return idx

    def sample_from_class_label(self, label:str) -> Tensor:
        """
        Randomly sample an image whose class label is `label`.
        """
        class_idx = self.__dataset.classes.index(label)
        image_idx = self.sample_from_class_idx(class_idx)
        image = self.__dataset.data[image_idx]
        if not isinstance(image, Tensor):
            return to_tensor(image)
        else:
            return image

    def sample_random_class_idx(self) -> int:
        """
        Sample a random class-label-index
        in the range `0, self.num_classes-1`.
        """
        return int(self.__rng.integers(self.num_classes))

    def __iter__(self) -> Iterator[int]:
        """
        Randomly sample indices from the dataset.
        The indices come in pairs, such that each
        index in the pair refers to an element of the same class.
        After two yielded indices, a new class is sampled uniformly.
        Within a class, the elements of the pairs are sampled
        uniformly. All sampling is performed with replacement;
        it is possible a pair consist of the same index twice.
        """
        current_class = self.sample_random_class_idx()
        for _ in range(self.epoch_size//2):
            yield self.sample_from_class_idx(current_class)
            yield self.sample_from_class_idx(current_class)
            current_class = self.sample_random_class_idx()

    def __len__(self) -> int:
        return self.epoch_size

class PairWiseBatchSampler(PairSampler):
    """
    Sampler that returns a batch of indices [x_0, x_1, x_2, ..., x_{N}, x_{N+1}] 
    from a dataset such that every two consecutive indices
    belong to dataset-elements of the same class
    (i.e., (x_0, x_1) have the same class, 
    (x_2, x_3) have the same class, etc.).
    The classes for each pair are randomly uniformly sampled,
    and given a class, the pair elements are also uniformly sampled
    from all elements of the given class.
    """

    def __init__(self, dataset: CIFAR10,
                 rng: np.random.Generator,
                 batch_size:int,
                 epoch_size: int):
        """
        Arguments:
        * dataset: a map-style PyTorch dataset, e.g. CIFAR10.
        * rng: number generator used for the randomness in the sampling.
        * batch_size: number *pairs* per batch.
            Each batch thus contains `2*batch_size` images.
        * epoch_size: even integer, number of images returned
            before iteration halts. Images are sampled
            with replacement from the dataset, so this
            number may exceed the dataset size.
        """
        print("Created PairWiseBatchSampler sampling "
              f"{epoch_size//(batch_size*2)} "
              f"batches of {batch_size} pairs each, "
              f"for a total of {epoch_size} images.")
        super().__init__(dataset, rng, epoch_size)
        self.__batch_size = batch_size
        if epoch_size % 2 != 0:
            raise ValueError(f"Epoch size must be even, got: {epoch_size}")

    @property
    def batch_size(self) -> int:
        """
        Return the number of pairs of indices in each batch.
        """
        return self.__batch_size

    def __iter__(self) -> Iterator[list[int]]:
        super_iterator = super().__iter__()
        budget = self.epoch_size
        while budget >= self.batch_size:
            batch = [next(super_iterator) for _ in range(2*self.batch_size)]
            budget -= 2*self.batch_size
            yield batch

class CorrelatedBatchSampler(PairWiseBatchSampler):
    """
    Same as `PairWiseBatchSampler`,
    but each batch only contains pairs of the same class label.
    The class of the batch is uniformly randomly chosen
    for each batch.
    """

    def __iter__(self) -> Iterator[list[int]]:
        budget = self.epoch_size
        while budget >= self.batch_size:
            class_idx = RNG.choice(CIFAR10_NUM_CLASSES)
            # Note: batch_size are the number of **pairs**, hence the `2*`.
            batch = [self.sample_from_class_idx(class_idx)
                     for _ in range(2*self.batch_size)]
            budget -= 2*self.batch_size
            yield batch

def get_indices_per_class(dataset: CIFAR10) -> list[list[int]]:
    """
    Return a list of the same length as the dataset's list of classes,
    containing the image-indices of all images of the corresponding class.
    """
    # Inspired from:
    #   https://discuss.pytorch.org/t/how-to-
    #   sample-images-belonging-to-particular-classes/43776/6
    output = []
    # List giving the class idx for each image
    targets = np.array(dataset.targets) 
    for class_idx in range(len(dataset.classes)):
        output.append(np.where(targets == class_idx)[0])
    return output

def get_n_images_each_class(n: int, dataset: CIFAR10
                            ) -> list[list[Tensor]]:
    """
    Sample `n` random images for each class in `dataset.classes`.

    Arguments:
    * n: number of images per class to sample.
    * dataset: dataset to sample images from.

    Returns:
    * list of `n` images for each class in `dataset.classes`.
        The order of the presented classes matches the
        order in `dataset.classes`.

    NOTE: assumes that the dataset uses the `to_tensor` transform.
    If not, it may return `np.ndarray`s instead of PyTorch tensors.
    """
    num_classes = len(dataset.classes)
    pair_sampler = PairSampler(dataset, RNG, epoch_size = n*num_classes)
    
    output = [[pair_sampler.sample_from_class_label(label)
               for _ in range(n)]
              for label in dataset.classes]
    return output



































