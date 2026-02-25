#**************************************************
# prob. n.13 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-I: a surrogate model algorithm for expensive nonlinear
# integer programming problems including global optimization applications
# Journal of Global Optimization, 59(4):865-889 (2014)
#**************************************************
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "SO-I prob. 13"
n: int = 10
starting_point: npt.NDArray[np.float64] = 51 * np.ones(n, dtype=np.float64) 

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    log1 = np.log(x - 2)
    log2 = np.log(100 - x)

    np.square(log1, out=log1)
    np.square(log2, out=log2)

    return np.sum(log1) * np.sum(log2) - np.prod(x**0.2)
