#**************************************************
# prob. n.16 described in paper:
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

name: str = "SO-I prob. 16"
n: int = 8
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype=np.float64) 

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return (x*x) @ _Y # pyright: ignore[reportReturnType]

_Y = np.array([3.1, 7.6, 6.9, 0.004, 19, 3, 1, 4], dtype=np.float64)
