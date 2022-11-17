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

This file provides a function to perform multiple runs of an
experiment and save the results to disk.
Another function can then be used to aggregate the results
into summarizing plots.
"""
# Library imports:
import torch
from torch import Tensor
import torch.nn as nn
import torchvision as vision
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader

from typing import Literal, Optional
from collections import namedtuple
import enum
import os
import multiprocessing

# Local imports:
from rolim.networks.architectures import AtariCNN
from rolim.settings import (MULT_RUNS_DIR, RNG)
from rolim.encoder.pairwise_sampler import PairWiseBatchSampler
from rolim.encoder.train_enc import train_encoder

WorkerJob = namedtuple("WorkerJob",
                       ("loss_fun", "num_batches", "batch_size",
                        "test_set_batch", "save_dir", "run_num")
                       )

class SubdirNames(enum.Enum):
    ENCODERS        = "encoders"
    LOSSES          = "losses"
    HEATMAPS        = "heatmaps"
    PREDICTIONS     = "predictions"
    TSNE_EMBEDDINGS = "tsne_embeddings"
    

def train_enc_multiple_runs(
                  num_runs: int,
                  loss_fun: Literal["MSE", "W-MSE", "DW-MSE"],
                  num_batches: int, 
                  batch_size: int, 
                  test_set_batch: int,
                  num_workers: Optional[int] = None,
                  root_dir: str = MULT_RUNS_DIR
                  ) -> str:
    """
    Train an `AtariCNN` on the CIFAR10 dataset for `num_runs`
    indepedent runs. The training objective is to minimize the distance of
    embeddings of images of the same class; the actual loss function
    is specified in `loss_fun`. Each run ends with a batch sampled
    from the test set, which is passed trough the newly trained network
    to get learned embeddings. These are used to compute an error matrix
    and are also fitted using t-SNE.
    Create a new directory `{distance_loss}_{data}.run`,
    containing subdirectories `encoders`, `losses`, `heatmaps`,
    `predictions` and `tsne_embeddings`.
    These will each contain as many files as `num_runs`,
    which are:
    - encoders: PyTorch networks, the saved encoders.
        These are saved using `torch.save()`.
    - losses: losses during training of the encoders.
        This are lists of floats saved as JSON files.
    - heatmaps: matrices representing the pairwise MSEs
        (computed using `compute_mse_heatmap` on a test-set batch).
        This are PyTorch tensors saved with `torch.save`.
    - predictions: embeddings computed for the test-set batch.
        Also PyTorch tensors.
    - tsne_embeddings: numpy matrix containing the embeddings for
        the test-set predictions.

    Arguments:
    * loss_fun: indication which loss function should be used,
        in case `method` is `"distance"`:
        * MSE: sum of squared distances between vectors of the same pair.
        * W-MSE: same as MSE, but with the Whitening Transform applied to
            the batch AFTER encoding BEFORE computing the loss.
        * DW-MSE: same as MSE, but normalize the pair-wise difference
            vectors by subtracting the mean difference-vector
            and multiplying by the inverse covariance matrix.
    * num_batches: number of batches of images to train on before the
        training stops.
    * batch_size: number of pairs of images in each minibatch.
    * test_set_batch: number of pairs to sample of each class
        for the test-set embeddings (used for the heatmap and t-SNE plot).
    * num_workers: number of parallell processes doing the training
        and evaluation. If `None`, equally many workers are created
        as the current machine has CPUs.
    * root_dir: directory in which the experiment directory is created.

    Returns:
    * Directory name of the newly created directory.
    """
    timestamp = get_timestamp()
    save_dir = os.path.join(root_dir, loss_fun + "_" + timestamp + ".run")
    for subdir in SubdirNames:
        path = os.path.join(save_dir, subdir.value)
        if not os.path.exists(path):
            os.makedirs(path)

    jobs = [WorkerJob(loss_fun=loss_fun,
                      num_batches=num_batches,
                      batch_size = batch_size,
                      test_set_batch = test_set_batch,
                      save_dir = save_dir,
                      run_num = num)
            for num in range(num_runs)]
    # To ensure GPU memory is freed after each run, kill the worker
    # process and create a new one for each run.
    pool = multiprocessing.Pool(processes=num_workers,
                                initializer=__worker_setup,
                                maxtasksperchild=1)
    print(pool.imap_unordered(__perform_run, jobs))
    pool.close()
    pool.join()

def __perform_run(job: WorkerJob):
    loss_fun, num_batches, batch_size, test_set_batch, save_dir, run_num = job
    print(f"Started run {run_num} on process PID {os.getpid()}")

    encoder, losses = train_encoder(method="distance", distance_loss=loss_fun,
                                    run_checks=False, num_batches=num_batches,
                                    batch_size=batch_size)

    testset = vision.datasets.CIFAR10(root=save_dir, train=False, 
                                      download=True, transform=to_tensor)
    batch_sampler = PairWiseBatchSampler(testset, RNG, batch_size=batch_size,
                                         epoch_size=batch_size*num_batches)
    testset_loader = DataLoader(testset, batch_sampler=batch_sampler)

    raise NotImplementedError("""TODO:
                              * save the losses and encoder
                              * Make test set predictions
                              * save test set predictions
                              * compute and save heatmap
                              * compute and save tsne
                              """)

def __worker_setup():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # global TRAINSET
    # global TESTSET
    # TRAINSET, TESTSET = load_cifar_10_dataset(download=False)

    # batch_sampler = PairWiseBatchSampler(trainset, RNG, batch_size=batch_size,
    #                                      epoch_size=batch_size*num_batches)
    # global TRAINSETLOADER
    # TRAINSETLOADER = DataLoader(TRAINSET, batch_sampler=batch_sampler)
    # global TESTSETLOADER
    # TRAINSETLOADER = DataLoader(TESTSETLOADER, batch_sampler=batch_sampler)



































