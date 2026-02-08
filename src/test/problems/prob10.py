#**************************************************
# prob. n.10 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-MI: A surrogate model algorithm for computationally 
# expensive nonlinear mixed-integer black-box global optimization problems
# Computers & Operations Research, 40(5):1383-1400 (2013)
#**************************************************
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "SOMI prob.10"
n: int = 5
starting_point: npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) 

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return x[3] * np.sin(x[3]) + 1.7 * x[4] * np.sin(x[3]) - 1.5 * x[0] - 0.1 * x[1] * np.cos(x[1] + x[2] - x[3]) + 0.2 * x[2] ** 2 - x[4] - 1
