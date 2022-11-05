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

# Default regularization constant to make a sample covariance matrix
# better conditioned in the whitening transform.
WHITEN_REG_EPS = 0.001

# The path to the `rolim` directory, containing among others
# this `settings.py` file.
PROJECT_ROOT_DIR = os.getcwd()

# Directory to store the CIFAR10 dataset images.
CIFAR10_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "cifar10")

