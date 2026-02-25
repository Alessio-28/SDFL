#**************************************************
# prob. n.8 described in paper:
# J. Müller
# MISO: Mixed-Integer Surrogate Optimization Framework
# Optimization and Engineering, 17(1):177-203 (2016)
#**************************************************
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "MISO prob. 8"
n: int = 15
starting_point: npt.NDArray[np.float64] = 7 * np.ones(n, dtype=np.float64) 

def feval(x: npt.NDArray[np.float64]) -> np.float64:  
    y = x[:_m]**2
    return (x[5] - 1)**2 + _Z @ (2*y[1:] - y[:-1])

_m: int = n - 1
_Z = np.arange(1, _m, dtype=np.float64)
