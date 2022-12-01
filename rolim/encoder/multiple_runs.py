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

It can be run as a script, in which case it performs a test run.
"""
# Library imports:
import torch
from torch import Tensor
import torchvision as vision
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

 # Provides same API as Python's `multiprocessing`
import torch.multiprocessing as multiprocessing

from typing import Literal, Optional, Any, Callable, TypeVar, Iterable
from collections import namedtuple
import enum
import os
import json
import math
from sklearn.manifold import TSNE
from scipy.signal import savgol_filter
import numpy as np
from numpy.typing import NDArray
import warnings
import matplotlib.pyplot as plt

# Local imports:
from rolim.networks.architectures import AtariCNN
from rolim.settings import (CIFAR10_CLASSES, CIFAR10_DIR, MULT_RUNS_DIR, RNG, 
                            DEVICE, 
                            TSNE_DEFAULT_PERPLEXITY,
                            REDOWNLOAD_DATASET)
from rolim.encoder.pairwise_sampler import get_n_images_each_class
from rolim.encoder.train_enc import train_encoder
from rolim.tools.mse_matrix import compute_mse_heatmap, plot_heatmap
from rolim.tools.data import (get_timestamp,
                              jitter_data,
                              nested_tensors_to_np,
                              json_load)
from rolim.tools.stats import get_all_diag_entries, get_all_upper_entries
T = TypeVar("T")

WorkerJob = namedtuple("WorkerJob",
                       ("loss_fun", "num_batches", "batch_size",
                        "test_set_batch", "save_dir", "run_num",
                        "correlated_batches")
                       )

class SubdirNames(enum.Enum):
    ENCODERS        = "encoders"
    LOSSES          = "losses"
    HEATMAPS        = "heatmaps"
    EMBEDDINGS      = "embeddings"
    TSNE_EMBEDDINGS = "tsne_embeddings"
    

PARAMETERS_FILENAME = "parameters.json"
TSNE_PLOT_FILENAME  = "tsne_plots.pdf"
HEATMAPS_SUMMARY_FILENAME = "heatmaps_summary.json"

class PARAM_KEYS(enum.Enum):
    NUM_RUNS = "num_runs"
    LOSS_FUN = "loss_fun"
    NUM_BATCHES	= "num_batches"
    BATCH_SIZE = "batch_size"		
    TEST_SET_BATCH = "test_set_batch"		
    NUM_WORKERS = "num_workers"	
    SAVE_DIR = "save_dir"
    TIME = "time"	
    CORREL_BATCHES = "correl_batches"

# Settings for the plots
CURVE_LINE_COLOR = np.array((201, 42, 42), dtype=np.float_) / 255
CURVE_FILL_COLOR = np.append(CURVE_LINE_COLOR, [128/255])
SAVGOL_WINDOW = 25
HEATMAP_COLOR_MIN = 0
HEATMAP_COLOR_MAX = 0.0005
HEATMAP_IMSHOW_KWARGS = {
        "vmin" : HEATMAP_COLOR_MIN,
        "vmax" : HEATMAP_COLOR_MAX
        }

def train_enc_multiple_runs(
                  num_runs: int,
                  loss_fun: Literal["MSE", "W-MSE", "DW-MSE"],
                  num_batches: int, 
                  batch_size: int, 
                  test_set_batch: int,
                  num_workers: Optional[int] = None,
                  root_dir: str = MULT_RUNS_DIR,
                  correlated_batches: bool = False
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
    See `PARAM_KEYS` for the keys of this dictionary. 

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
    * batch_size: number of **pairs** of images in each minibatch.
    * test_set_batch: number of images sample of each class
        for the test-set embeddings (used for the heatmap and t-SNE plot).
    * num_workers: number of parallell processes doing the training
        and evaluation. If `None`, equally many workers are created
        as the current machine has CPUs.
    * root_dir: directory in which the experiment directory is created.
    * correlated_batches: if `True`, all pairs of images within a batch
        are of the same class. A class is randomly chosen for each batch.

    Returns:
    * Directory name of the newly created directory.
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    assert (isinstance(num_workers, int))
    timestamp = get_timestamp()
    save_dir = os.path.join(root_dir, loss_fun + "_" + timestamp + ".run")
    for subdir in SubdirNames:
        path = os.path.join(save_dir, subdir.value)
        if not os.path.exists(path):
            os.makedirs(path)
    __save_parameters(num_runs, loss_fun, num_batches, batch_size,
                      test_set_batch,
                      correlated_batches = correlated_batches,
                      num_workers=num_workers, 
                      save_dir=save_dir)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_dtype(torch.FloatTensor)

    jobs = [WorkerJob(loss_fun=loss_fun,
                      num_batches=num_batches,
                      batch_size = batch_size,
                      test_set_batch = test_set_batch,
                      save_dir = save_dir,
                      run_num = num,
                      correlated_batches = correlated_batches)
            for num in range(num_runs)]
    __dispatch_jobs(jobs, num_workers)
    return save_dir

def __dispatch_jobs(jobs: list[WorkerJob], num_workers: int):
    while len(jobs) > 0:
        next_jobs = [jobs.pop() for _ in range(min(len(jobs), num_workers))]
        print(f"Starting the next batch of {len(next_jobs)} jobs.")
        procs = [multiprocessing.spawn(__perform_run, args=(job,), nprocs=1,
                              join=False) for job in next_jobs]
        assert len(procs) <= num_workers, "Too many workers are being created."
        for proc in procs:
            proc.join()
        print(f"Current batch of {len(next_jobs)} jobs all finished.")
    print("All jobs (experiment runs) have finished.")


def __save_parameters(num_runs: int,
                      loss_fun: Literal["MSE", "W-MSE", "DW-MSE"],
                      num_batches: int, 
                      batch_size: int, 
                      test_set_batch: int,
                      correlated_batches: bool,
                      num_workers: Optional[int] = None,
                      save_dir: str = MULT_RUNS_DIR):
    parameters = {
          PARAM_KEYS.NUM_RUNS.value:       num_runs,
          PARAM_KEYS.LOSS_FUN.value:       loss_fun,
          PARAM_KEYS.NUM_BATCHES.value:    num_batches,
          PARAM_KEYS.BATCH_SIZE.value:     batch_size,
          PARAM_KEYS.TEST_SET_BATCH.value: test_set_batch,
          PARAM_KEYS.NUM_WORKERS.value:    num_workers,
          PARAM_KEYS.SAVE_DIR.value:       save_dir,
          PARAM_KEYS.TIME.value:           get_timestamp(),
          PARAM_KEYS.CORREL_BATCHES.value: correlated_batches
    }
    filename = os.path.join(save_dir, PARAMETERS_FILENAME)
    with open(filename, "w") as f:
        json.dump(parameters, f)

def __perform_run(i:int, job: WorkerJob):
    (loss_fun, num_batches, batch_size, test_set_batch, save_dir, run_num, \
        correlated_batches) = job
    print(f"Started run {run_num} on process PID {os.getpid()}")

    encoder, losses = train_encoder(method="distance", distance_loss=loss_fun,
                                    run_checks=False, num_batches=num_batches,
                                    batch_size=batch_size,
                                    correlated_batches=correlated_batches)

    print(f"Run {run_num} finished training; starting test set evaluation")

    testset = vision.datasets.CIFAR10(root=CIFAR10_DIR, train=False, 
                                      download=REDOWNLOAD_DATASET, 
                                      transform=to_tensor)
    test_images = get_n_images_each_class(test_set_batch, testset)
    test_images = [torch.stack([image 
                    for image in images], dim=0).to(DEVICE)
                    for images in test_images]

    with torch.no_grad():
        test_embeddings = [encoder(batch).cpu() for batch in test_images]
        for embedding in test_embeddings:
            embedding.requires_grad_(False)

    tsne_output = __make_tsne_embeddings(testset, test_set_batch, 
                                          test_embeddings)
    with torch.no_grad():
        heatmap = compute_mse_heatmap(test_embeddings)

    print(f"Finished run {run_num}. Starting to save data.")

    run_name = f"run_{run_num}"
    __save_losses(losses, save_dir, run_name)
    __save_encoder(encoder, save_dir, run_name)
    __save_embeddings(test_embeddings, save_dir, run_name)
    __save_tsne_embeddings(tsne_output, save_dir, run_name)
    __save_heatmap(heatmap, save_dir, run_name)

def __make_tsne_embeddings(testset: CIFAR10,
                           test_set_batch: int,
                           test_embeddings: list[Tensor]) -> np.ndarray:
    # Perplexity must be less than the number of samples.
    num_embeddings = sum(class_embeddings.shape[0] for
                         class_embeddings in test_embeddings)
    assert num_embeddings == test_set_batch*len(testset.classes), \
            "Different number of sampled embeddings than expected."
    if num_embeddings <= TSNE_DEFAULT_PERPLEXITY:
        perplexity = num_embeddings-1
        warnings.warn("Sample size too small for default TSNE perplexity.\n"
                      f"Using perplexity {perplexity} instead.")
    else:
        perplexity = TSNE_DEFAULT_PERPLEXITY

    tsne = TSNE(n_components=2,
            learning_rate="auto",
            init="pca",
            perplexity=perplexity,
            verbose=0)

    test_embeddings_np = nested_tensors_to_np(test_embeddings)
    all_encodings = np.concatenate(test_embeddings_np, axis=0)
    return tsne.fit_transform(all_encodings)

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


def perform_test_execution(correlated_batches: bool=False) -> str:
    """
    Test the multiple runs training, saving, loading
    and aggregating pipeline,
    using very short training procedures
    (the minimal number to ensure it generalizes to multiple runs,
    multiple batches, etc.).

    Return the created directory (can be used as input to `aggregate_results`).
    """
    save_dir = train_enc_multiple_runs(num_runs=2,
                                       loss_fun="MSE",
                                       batch_size=2,
                                       num_batches=50,
                                       test_set_batch=2,
                                       num_workers=2,
                                       root_dir = os.path.join(MULT_RUNS_DIR,
                                                           "test_executions"),
                                       correlated_batches=correlated_batches)
    return save_dir

def aggregate_results(save_dir: str,
                      tsne_plots_num_cols: int,
                      tsne_apply_jitter: bool):
    """
    Load the output of `train_enc_multiple_runs` from disk,
    and make:
    * A plot of the learning curves (losses), averaged out over the
        different runs, with smoothing and confidence bounds.
    * A multiple-figures plot of the t-SNE embeddings of each run.
    * A heatmap-plot of the mean values of the heatmaps.
    * A heatmap-plot of the standard deviations of the heatmaps.
    These outputs will be saved in the directory `save_dir`
    as PDF files.
    * A JSON file with summary statistics of the heatmaps.

    Arguments:
    * save_dir: path to the `*.run` directory created by 
        `train_enc_multiple_runs`, which contains the saved data.
    * tsne_plots_num_cols: number of columns of figures in the
        multiple-figure t-SNE plot. The number of rows will be
        chosen automatically.
    * tsne_apply_jitter: whether to add random noise to the
        t-SNE plots (may help to view composition of clusters).

    NOTE: it assumes that equally many predictions were made
        for each of the CIFAR10 classes and that the t-SNE
        outputs are stacked as a matrix.
        So if there are C classes and N samples in total of dimension D,
        then it is expected that the saved t-SNE .npy matrix has shape
        (N, D), where the first C rows are embeddings of images of class 0,
        the second C rows correspond to class 1, etc.
    """
    params = __load_parameters(save_dir)
    __make_tsne_mulitplot(save_dir, tsne_plots_num_cols, tsne_apply_jitter,
                          params)
    __aggregate_heatmaps(save_dir, params)
    __plot_learning_curves(save_dir, params)
    print("Finished aggregating data. Output has been saved to disk.")

def __load_parameters(save_dir: str) -> dict[str, Any]:
    params_filename = os.path.join(save_dir, PARAMETERS_FILENAME)
    with open(params_filename, "r") as f:
        params = json.load(f)
    return params

def __make_tsne_mulitplot(save_dir: str, num_cols: int,
                          apply_jitter: bool,
                          params: dict[str, Any]):
    tsne_embedddings_dir = os.path.join(save_dir,
                                        SubdirNames.TSNE_EMBEDDINGS.value)
    embeddingses = __load_files_ending_with(".npy", tsne_embedddings_dir,
                                            np.load)

    num_runs = len(embeddingses)
    num_rows = int(math.ceil(num_runs / num_cols))
    figsize = (5*num_cols, 5*num_rows)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    axs = axs.reshape((-1,))

    for i in range(num_runs):
        ax= axs[i]
        embeddings = embeddingses[i]
        if apply_jitter:
            embeddings = jitter_data(RNG, 10, embeddings)
        num_classes = len(CIFAR10_CLASSES)
        assert (embeddings.shape[0] % num_classes == 0), \
                "Cannot evenly divide the embeddings over the image classes."
        embeddings_per_class = embeddings.shape[0] // num_classes
        for j in range(num_classes):
            label = CIFAR10_CLASSES[j]
            # Pop first `embeddings_per_class` rows
            class_embeddings = embeddings[:embeddings_per_class]
            embeddings = embeddings[embeddings_per_class:]
            ax.plot(class_embeddings[:, 0], class_embeddings[:, 1], "+", 
                    label=label)

    # No need to repeat the legend in each subfigure, in first is enough.
    axs[0].legend()
    title = f"t_SNE embeddings per run using {params['loss_fun']} loss"
    fig.suptitle(title, weight="bold", size="xx-large")
    filename = os.path.join(save_dir, TSNE_PLOT_FILENAME)
    fig.savefig(filename) # type: ignore
    print(f"Saved t-SNE multiplot as {filename}.")

def __plot_learning_curves(save_dir: str, params: dict[str, Any]):
    losses_dir = os.path.join(save_dir, SubdirNames.LOSSES.value)
    losseses: list[list[float]]
    losseses = __load_files_ending_with(".json", losses_dir, json_load)
    loss_mat = np.array(losseses)
    loss_means = np.mean(loss_mat, axis=0).reshape((-1,))
    loss_stds = np.std(loss_mat, axis=0, ddof=1).reshape((-1,))
    num_batches = params[PARAM_KEYS.NUM_BATCHES.value]
    assert len(loss_means) == num_batches, \
            f"Got {len(loss_means)} losses, but expected {num_batches} batches."
    
    loss_means_smooth = __smoothe_plot(loss_means)
    loss_stds_smooth = __smoothe_plot(loss_stds)
    conf_ival_top = loss_means_smooth + loss_stds_smooth
    conf_ival_bot = loss_means_smooth - loss_stds_smooth

    fig, ax = plt.subplots(nrows=1, ncols=1)
    x_points = list(range(num_batches))
    ax.semilogy(x_points, loss_means_smooth, "-",
            color=CURVE_LINE_COLOR)
    ax.fill_between(x_points, conf_ival_bot, conf_ival_top,
                    color=CURVE_FILL_COLOR)
    loss_name = params[PARAM_KEYS.LOSS_FUN.value]
    ax.set_title(f"Mean losses per batch ({loss_name} loss)",
                 weight="bold")
    ax.set_xlabel("Batch index")
    ax.set_ylabel(f"{loss_name} loss")
    fig.tight_layout()
    
    filename = os.path.join(save_dir, "learn_curves.pdf")
    fig.savefig(filename)
    print(f"Saved learning curves as:\n{filename}")

def __smoothe_plot(y_values: np.ndarray) -> np.ndarray:
    window = max(len(y_values), SAVGOL_WINDOW)
    return savgol_filter(y_values, window_length=window, mode="nearest",
                         polyorder=2)

def __aggregate_heatmaps(save_dir: str, params: dict[str, Any]):
    """
    Load heatmaps, compute and save the following:
    * PDF, per-entry mean of heatmaps.
    * PDF, per-entry sample standard deviation of heatmaps.
    * JSON, summary statistics of heatmaps.
    """
    heatmaps_dir = os.path.join(save_dir, SubdirNames.HEATMAPS.value)
    heatmaps = __load_files_ending_with(".pt", heatmaps_dir, torch.load)
    __plot_heatmaps(heatmaps, save_dir, params)
    __heatmap_summary_statistics(heatmaps, save_dir)

def __plot_heatmaps(heatmaps: list[Tensor],
                    save_dir: str, params: dict[str, Any]):
    stacked = torch.stack(heatmaps)
    mean_heatmap = torch.mean(stacked, dim=0)
    std_heatmap = torch.std(stacked, dim=0)
    m_ax, m_fig, m_cb = plot_heatmap(mean_heatmap, 
                                     xtick_labels=CIFAR10_CLASSES,
                                     ytick_labels=CIFAR10_CLASSES,
                                     imshow_kwargs=HEATMAP_IMSHOW_KWARGS)
    m_fig.suptitle(f"Mean of pairwise MSE errors for {params['loss_fun']} loss",
                   weight="bold", size="large") 
    m_fig.tight_layout() 
    m_filename = os.path.join(save_dir, "heatmaps_mean.pdf")
    m_fig.savefig(m_filename)
    s_ax, s_fig, s_cb = plot_heatmap(std_heatmap, 
                                     xtick_labels=CIFAR10_CLASSES,
                                     ytick_labels=CIFAR10_CLASSES,
                                     imshow_kwargs=HEATMAP_IMSHOW_KWARGS)
    s_fig.suptitle(f"Standard deviation of pairwise MSE errors "
                   f"for {params['loss_fun']} loss",
                   weight="bold", size="large")
    s_fig.tight_layout()
    s_filename = os.path.join(save_dir, "heatmaps_std.pdf")
    s_fig.savefig(s_filename)
    print(f"Saved aggregated heatmaps as\n{m_filename}\nand\n{s_filename}")

def __heatmap_summary_statistics(heatmaps:Iterable[Tensor|NDArray],
                                 save_dir: str):
    """
    Compute summary statistics of the heatmaps and save it to a JSON
    file (as a dictionary).
    Statistics with JSON keys:
    * "diag_entries": list[float], diagonal entries of all heatmaps
        (concatenated into a single list).
    * "upper_entries": list[float], upper triangular entries of
        all heatmaps, not including the diagonal entries themselves
        (also concatenated into a single list).
    * "diag_mean": float, mean of diagonal entries.
    * "diag_std": float, sample standard deviation of diagonal entries.
    * "upper_mean": float, mean of upper-triangular entries.
    * "upper_std": float, std of upper-triangular entries.

    Note that the summary statistics take the concatenation of
    all diagonal/upper-triangular entries as the sample.
    """
    # Numpy floating types are not serializible, need to convert to Python
    # floats.
    diag_entries = get_all_diag_entries(heatmaps)
    diag_mean = float(np.mean(diag_entries))
    diag_std = float(np.std(diag_entries, ddof=1))
    diag_entries_list = diag_entries.tolist()

    upper_entries = get_all_upper_entries(heatmaps)
    upper_mean = float(np.mean(upper_entries))
    upper_std = float(np.std(upper_entries, ddof=1))
    upper_entries_list = upper_entries.tolist()

    dictionary = {
            "diag_entries":diag_entries_list,
            "upper_entries":upper_entries_list,
            "diag_mean":diag_mean,
            "diag_std":diag_std,
            "upper_mean":upper_mean,
            "upper_std":upper_std
            }
    filename = os.path.join(save_dir, HEATMAPS_SUMMARY_FILENAME)
    with open(filename, "w") as f:
        json.dump(dictionary, f)
    print(f"Saved heatmap summary statistics as:\n{filename}")


            
def __load_files_ending_with(extension: str, directory: str,
                             load_function: Callable[[str], T]
                             ) -> list[T]:
    output = []
    files = sorted(os.scandir(directory),
                   key=lambda x : x.name)
    for file in files:
        if file.name.endswith(extension):
            output.append(load_function(file.path))
    return output

if __name__ == "__main__":
    testrun_dir = perform_test_execution(correlated_batches=False)
    aggregate_results(testrun_dir, tsne_plots_num_cols=2, 
                      tsne_apply_jitter=False)




























