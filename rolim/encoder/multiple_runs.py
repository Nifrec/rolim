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

 # Provides same API as Python's `multiprocessing`
import torch.multiprocessing as multiprocessing

from typing import Literal, Optional
from collections import namedtuple
import enum
import os
import json
from sklearn.manifold import TSNE
import numpy as np
import warnings

# Local imports:
from rolim.networks.architectures import AtariCNN
from rolim.settings import (CIFAR10_DIR, MULT_RUNS_DIR, RNG, DEVICE, TSNE_DEFAULT_PERPLEXITY)
from rolim.encoder.pairwise_sampler import (PairWiseBatchSampler,
                                            get_n_images_each_class)
from rolim.encoder.train_enc import train_encoder
from rolim.tools.mse_matrix import compute_mse_heatmap, plot_heatmap
from rolim.tools.data import (get_timestamp,
                              jitter_data,
                              nested_tensors_to_np,
                              all_tensor_in_list_to_cpu)

WorkerJob = namedtuple("WorkerJob",
                       ("loss_fun", "num_batches", "batch_size",
                        "test_set_batch", "save_dir", "run_num")
                       )

class SubdirNames(enum.Enum):
    ENCODERS        = "encoders"
    LOSSES          = "losses"
    HEATMAPS        = "heatmaps"
    EMBEDDINGS      = "embeddings"
    TSNE_EMBEDDINGS = "tsne_embeddings"
    

PARAMETERS_FILENAME = "parameters.json"

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
    - embeddings: embeddings computed for the test-set batch.
        These are used as a list of (num_samples, embedding_dim) tensors
        during data analysis, but stored as a single stacked
        PyTorch tensor.
    - tsne_embeddings: numpy matrix containing the embeddings for
        the test-set predictions. This are `.npy' files.

    Also make a JSON file `settings.json` in this new directory
    storing the parameters of the experiment.

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
    __save_parameters(num_runs, loss_fun, num_batches, batch_size,
                      test_set_batch, num_workers, save_dir)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_dtype(torch.FloatTensor)

    jobs = [WorkerJob(loss_fun=loss_fun,
                      num_batches=num_batches,
                      batch_size = batch_size,
                      test_set_batch = test_set_batch,
                      save_dir = save_dir,
                      run_num = num)
            for num in range(num_runs)]
    procs = [multiprocessing.spawn(__perform_run, args=(job,), nprocs=1,
                              join=False) for job in jobs]
    for proc in procs:
        proc.join()
    # To ensure GPU memory is freed after each run, kill the worker
    # process and create a new one for each run.
    # pool = multiprocessing.Pool(processes=num_workers,
    #                             initializer=__worker_setup,
    #                             maxtasksperchild=1)
    # print(pool.map(__perform_run, jobs))
    # pool.close()
    # pool.join()
    return save_dir

def __worker_setup():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def __save_parameters(num_runs: int,
                      loss_fun: Literal["MSE", "W-MSE", "DW-MSE"],
                      num_batches: int, 
                      batch_size: int, 
                      test_set_batch: int,
                      num_workers: Optional[int] = None,
                      save_dir: str = MULT_RUNS_DIR):
    parameters = {
                  "num_runs": num_runs,
                  "loss_fun":loss_fun,
                  "num_batches":num_batches,
                  "batch_size":batch_size,
                  "test_set_batch":test_set_batch,
                  "num_workers": num_workers,
                  "save_dir":save_dir,
                  "time": get_timestamp()
                  }
    filename = os.path.join(save_dir, PARAMETERS_FILENAME)
    with open(filename, "w") as f:
        json.dump(parameters, f)

def __perform_run(i:int, job: WorkerJob):
    loss_fun, num_batches, batch_size, test_set_batch, save_dir, run_num = job
    print(f"Started run {run_num} on process PID {os.getpid()}")

    encoder, losses = train_encoder(method="distance", distance_loss=loss_fun,
                                    run_checks=False, num_batches=num_batches,
                                    batch_size=batch_size)

    testset = vision.datasets.CIFAR10(root=CIFAR10_DIR, train=False, 
                                      download=False, transform=to_tensor)
    # batch_sampler = PairWiseBatchSampler(testset, RNG, batch_size=batch_size,
    #                                      epoch_size=batch_size*num_batches)
    # testset_loader = DataLoader(testset, batch_sampler=batch_sampler)

    print(f"Run {run_num} finished training; starting test set evaluation")

    test_images = get_n_images_each_class(test_set_batch, testset)
    test_images = [torch.stack([image 
                    for image in images], dim=0).to(DEVICE)
                    for images in test_images]

    with torch.no_grad():
        test_embeddings = [encoder(batch).cpu() for batch in test_images]
        for embedding in test_embeddings:
            embedding.requires_grad_(False)

    num_embeddings = sum(class_embeddings.shape[0] for
                         class_embeddings in test_embeddings)
    assert num_embeddings ==test_set_batch*len(testset.classes) 
    if num_embeddings <= TSNE_DEFAULT_PERPLEXITY:
        perplexity = num_embeddings-1
        warnings.warn("Sample size too small for default TSNE perplexity.\n"
                      f"Using perplexity {perplexity} instead.")
    else:
        perplexity = TSNE_DEFAULT_PERPLEXITY

    tsne = TSNE(n_components=2,
            learning_rate="auto",
            init="pca",
            # Perplexity must be less than the number of samples.
            perplexity=perplexity,
            verbose=0)

    test_embeddings_np = nested_tensors_to_np(test_embeddings)
    all_encodings = np.concatenate(test_embeddings_np, axis=0)
    tsne_output = tsne.fit_transform(all_encodings)

    with torch.no_grad():
        heatmap = compute_mse_heatmap(test_embeddings)

    print(f"Finished run {run_num}. Starting to save data.")

    run_name = f"run_{run_num}"
    __save_losses(losses, save_dir, run_name)
    __save_encoder(encoder, save_dir, run_name)
    __save_embeddings(test_embeddings, save_dir, run_name)
    __save_tsne_embeddings(tsne_output, save_dir, run_name)
    __save_heatmap(heatmap, save_dir, run_name)

def __save_losses(losses: list[float], run_root_dir: str, run_name: str):
    filename = os.path.join(run_root_dir, 
                                   SubdirNames.LOSSES.value,
                                   run_name + ".json")
    with open(filename, "w") as f:
        json.dump(losses, f)
    print(f"Saved losses of {run_name} as '{filename}'")

def __save_encoder(encoder: AtariCNN, run_root_dir: str, run_name: str):
    filename = os.path.join(run_root_dir, SubdirNames.ENCODERS.value,
                            run_name + ".pt")
    torch.save(encoder, filename)
    print(f"Saved encoder of {run_name} as '{filename}'")

def __save_embeddings(embeddings: list[Tensor], 
                      run_root_dir: str, run_name: str):
    """
    Convert the embeddings to a 3-dimensional tensor,
    and save this to disk via PyTorch's save method.
    """
    filename = os.path.join(run_root_dir, SubdirNames.EMBEDDINGS.value,
                            run_name + ".pt")
    embeddings_stack = torch.stack(embeddings)
    torch.save(embeddings_stack, filename)
    print(f"Saved stack of embeddings of {run_name} as '{filename}'")

def __save_tsne_embeddings(tsne_embeddings: np.ndarray, 
                           run_root_dir: str, 
                           run_name: str):
    filename = os.path.join(run_root_dir, 
                                   SubdirNames.TSNE_EMBEDDINGS.value,
                                   run_name + ".npy")
    np.save(arr=tsne_embeddings, file=filename)
    print(f"Saved T-SNE embeddings of {run_name} as '{filename}'")

def __save_heatmap(heatmap: Tensor, run_root_dir: str, run_name: str):
    """
    Convert the embeddings to a 3-dimensional tensor,
    and save this to disk via PyTorch's save method.
    """
    filename = os.path.join(run_root_dir, SubdirNames.HEATMAPS.value,
                            run_name + ".pt")
    torch.save(heatmap, filename)
    print(f"Saved heatmap of {run_name} as '{filename}'")


def perform_test_execution():
    """
    Test the multiple runs training, saving, loading
    and aggregating pipeline,
    using very short training procedures
    (the minimal number to ensure it generalizes to multiple runs,
    multiple batches, etc.).
    """
    save_dir = train_enc_multiple_runs(num_runs=2,
                                       loss_fun="MSE",
                                       batch_size=2,
                                       num_batches=2,
                                       test_set_batch=2,
                                       num_workers=2,
                                       root_dir = os.path.join(MULT_RUNS_DIR,
                                                           "test_executions"))


if __name__ == "__main__":
    perform_test_execution()


























