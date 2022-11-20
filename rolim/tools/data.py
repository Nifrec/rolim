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

Helper functions for converting or transforming data,
generating filenames, etc.
"""
# Library imports:
import time
import numpy as np
import torch
from torch import Tensor

def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d___%H-%M")

def nested_tensors_to_np(tensors: list[Tensor]) -> list[np.ndarray]:
    output = [tensor.cpu().numpy() for tensor in tensors]
    return output

def jitter_data(rng: np.random.Generator, 
                scale: float, 
                data: np.ndarray
                ) -> np.ndarray:
    """
    Add standard normal noise, multiplied with `scale`,
    to each value in the data.
    """
    noise = rng.standard_normal(size=data.shape)*scale
    return data+noise

def all_tensor_in_list_to_cpu(tensors: list[Tensor]) -> list[Tensor]:
    return [tensor.cpu() for tensor in tensors]





























