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

name   : str = "SO-I prob. 16"
n      : int = 8
startp : npt.NDArray[np.float64] = np.ones(n, dtype = np.float64) 

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    return 3.1 * x[0] ** 2 + 7.6 * x[1] ** 2 + 6.9 * x[2] ** 2 + 0.004 * x[3] ** 2 + 19 * x[4] ** 2 + 3 * x[5] ** 2 + x[6] ** 2 + 4 * x[7] ** 2
