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

This is a script that runs the actual experiments
concerning training encoders on CIFAR10.
"""
# Library imports:
import torch
from torch import Tensor

# Local imports:
from rolim.settings import MULT_RUNS_DIR
from rolim.encoder.multiple_runs import (
        aggregate_results,
        train_enc_multiple_runs)


NUM_RUNS = 50
BATCH_SIZE = 25
NUM_WORKERS = 4
TEST_SET_BATCH = 300
NUM_BATCHES = 1000
CORRELATED_BATCHES = False

def run_default_experiments() -> list[str]:
    """
    Train an encoder for multiple runs
    on the CIFAR10 task
    using the following losses:
    MSE, W-MSE, DW-MSE.
    """
    save_dirs : list[str] = []
    for loss_fun in ["MSE", "W-MSE", "DW-MSE"]:
        save_dir = train_enc_multiple_runs(num_runs=NUM_RUNS,
                                           loss_fun=loss_fun, #type: ignore
                                           batch_size=BATCH_SIZE,
                                           num_batches=NUM_BATCHES,
                                           test_set_batch=TEST_SET_BATCH,
                                           num_workers=NUM_WORKERS,
                                           root_dir = MULT_RUNS_DIR,
                                           correlated_batches=CORRELATED_BATCHES
                                           )
        save_dirs.append(save_dir)
    return save_dirs

if __name__ == "__main__":
    save_dirs = run_default_experiments()
    for save_dir in save_dirs:
        aggregate_results(save_dir, tsne_plots_num_cols=5,
                          tsne_apply_jitter=False)
























