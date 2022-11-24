# Reproduction Of LWM for Intrinsic Motivation (short: Rolim)
This repository is a project to reproduce
parts of the paper *Latent World Models
For Intrinsically Motivated Exploration*
by Alexander Ermolov and Nice Sebe (2020).

## Overview

### Most interesting
* `rolim/encoder/multiple_runs.py` contains functions implementing the
    main experiment. These functions work in 2 phases:
    - The first function `train_enc_multiiple_runs`
        trains multiple encoders in parallel, evaluates them on the test set
        (makes error heatmap and t-SNE embeddings), and stores all
        results to disk. The implementation is parallelized.
    - The second function `aggregate_results` can then be used
        to read the results, make a multi-figure t-SNE plot,
        and aggregated heatmaps. These are also stored to disk.
* `rolim/notebooks/encoder_experiment.ipynb` contains single-run
    versions of the experiments used to train an encoder,
    including visualizations of the results
    (this notebook was created before the ideas from it were
    generalized to the multi-run script).

### Other files of interest
* `rolim/notebooks/whitening.ipynb` explores the Whitening
    Transform, Cholesky decomposition, and performs
    a small empirical experiment to compare 2 different implementation
    of Whitening Transform.
* `rolim/noteboos/cifar_10.ipynb` was an experimental approach to finding
    out how to iterate ove random pairs of images of the same
    class in the CIFAR10 dataset.
    This resulted in the neat version of the code in 
    `rolim/encoder/pairwise_sampler.py`.
* Unit-tests are collected in `rolim/test`.
    Most functions have at least 1 testcase.
* `rolim/networks/architectures.py` contains the definitions
    of Neural Network Architectures.
* `rolim/encoder/` contains functions for training the encoder-network,
    as well as encoder-specific experiments.
* `rolim/tools/` contains several files of helper functions.

## Requirements
The file `rolim/tools/libraries.py` imports all required libraries
and print their versions, so it is a simple tool to verify the installation.
This project has been tested with libraries:
* python:       3.10.6
* torch:        1.12.1+cu102
* torchvision:  0.13.1+cu102
* numpy:        1.23.4
* sklearn:      1.1.3
* matplotlib:   3.6.1
* pandas:       1.5.1
* scipy:        1.9.3
Other versions may also be compatible,
but have not been tested.

## Notes/potential bugs
* The code may accidentally try to store tensors on a GPU
    even when specified differently in `settings.py`.
* You may need to redownload the CIFAR10 dataset. Set `REDOWNLOAD_DATASET`
    in `settings.py` to `True`. It is best to run scripts while using
    `rolim` (the directory containing `settings.py`) as your current working
    directory.
* The test run breaks on my machine when I raise the number of batches to 100
    instead of 50. A PyTorch error I cannot really interpret.
    It seems it fails to release CUDA memory when multiprocessing,
    since `torch` calls a function of `torch` that is undefined...
    ```
    File "/home/nifrec/.pyenv/versions/rolim/lib/python3.10/site-packages/torch/storage.py", line 757, in _expired
    return eval(cls.__module__)._UntypedStorage._expired(*args, **kwargs)
    AttributeError: module 'torch.cuda' has no attribute '_UntypedStorage
    ```
