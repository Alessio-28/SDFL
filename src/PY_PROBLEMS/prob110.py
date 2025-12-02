#**************************************************
# prob. n.10 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-I: a surrogate model algorithm for expensive nonlinear
# integer programming problems including global optimization applications
# Journal of Global Optimization, 59(4):865-889 (2014)
# 
# N.B. variables u_i are x(i), i = 1..nu
#      variables x_i are x(r+i), i = 1..nx
#**************************************************
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "SO-I prob. 10"
n         : int = 30
nint      : int = 15
ncont     : int = n - nint
lb        : npt.NDArray[np.float64] = -1 * np.ones(n, dtype = np.float64)
ub        : npt.NDArray[np.float64] =  3 * np.ones(n, dtype = np.float64)
lbmix     : npt.NDArray[np.float64] = -1 * np.ones(n, dtype = np.float64)
ubmix     : npt.NDArray[np.float64] =  3 * np.ones(n, dtype = np.float64)
startp    : npt.NDArray[np.float64] =  1 * np.ones(n, dtype = np.float64) 
x_initial : npt.NDArray[np.float64] =  1 * np.ones(n, dtype = np.float64) 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    f = np.sum(x ** 2 - np.cos(2 * np.pi * x))
    return f


