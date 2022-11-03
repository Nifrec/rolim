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
* Unit=tests are collected in `rolim/test`.
    Most functions have at least 1 testcase.
* `rolim/networks/architectures.py` contains the definitions
    of Neural Network Architectures.
* `rolim/encoder/` contains functions for training the encoder-network,
    as well as encoder-specific experiments.
