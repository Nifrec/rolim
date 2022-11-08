# Reproduction Of LWM for Intrinsic Motivation (short: Rolim)
This repository is a project to reproduce
parts of the paper *Latent World Models
For Intrinsically Motivated Exploration*
by Alexander Ermolov and Nice Sebe (2020).

## Overview

* `rolim/notebooks/whitening.ipynb` explores the Whitening
    Transform, Cholesky decomposition, and performs
    a small empirical experiment to compare 2 different implementation
    of Whitening Transform.
* `rolim/noteboos/cifar_10.ipynb` was an experimental approach to finding
    out how to iterate ove random pairs of images of the same
    class in the CIFAR10 dataset.
    This resulted in the neat version of the code in `rolim/encoder/pairwise_sampler.py`.
* Unit-tests are collected in `rolim/test`.
    Most functions have at least 1 testcase.
* `rolim/networks/architectures.py` contains the definitions
    of Neural Network Architectures.
* `rolim/encoder/` contains functions for training the encoder-network,
    as well as encoder-specific experiments.

