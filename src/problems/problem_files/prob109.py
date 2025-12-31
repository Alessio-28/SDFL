#**************************************************
# prob. n.9 described in paper:
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

name   : str = "SO-I prob. 9"
n      : int = 12
startp : npt.NDArray[np.float64] =  1 * np.ones(n, dtype = np.float64) 

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    return np.sum(x ** 2 - np.cos(2 * np.pi * x))
