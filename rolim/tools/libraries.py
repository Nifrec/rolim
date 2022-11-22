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

Script for testing if libraries are installed,
and for printing library versions.
"""
# Non-standard libraries used in this project:
import torch
import torchvision as vision
import numpy as np
import sklearn
import matplotlib
import pandas as pd
import scipy

def main():
    print("Found:",
          f"torch:{torch.__version__}",
          f"torchvision:{vision.__version__}",
          f"numpy:{np.__version__}",
          f"sklearn:{sklearn.__version__}",
          f"matplotlib:{matplotlib.__version__}",
          f"pandas:{pd.__version__}",
          f"scipy:{scipy.__version__}",
          sep="\n")

if __name__ == "__main__":
    main()
