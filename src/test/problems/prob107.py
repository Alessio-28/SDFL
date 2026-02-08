#**************************************************
# prob. n.7 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-I: a surrogate model algorithm for expensive nonlinear
# integer programming problems including global optimization applications
# Journal of Global Optimization, 59(4):865-889 (2014)
#**************************************************
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "SO-I prob. 7"
n: int = 10
starting_point: npt.NDArray[np.float64] = 6 * np.ones(n, dtype = np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(np.log(x - 2) ** 2) + np.sum(np.log(10 - x) ** 2) - np.prod(x ** 0.2)
