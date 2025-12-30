# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "rosen-suzuki"
startp    : npt.NDArray[np.float64] = np.array([0, 0, 0, 0], dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 2
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)


def feval(x : npt.NDArray[np.float64]) -> np.float64:
    f : npt.NDArray[np.float64] = np.zeros(4, dtype = np.float64)

    f[0] =  x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2 
    f[0] =  f[0] - 5 * (x[0] + x[1]) - 21 * x[2] + 7 * x[3] 

    f[1] =  x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
    f[1] =  f[1] + x[0] - x[1] + x[2] - x[3] - 8

    f[2] =  x[0] ** 2 + 2 * x[1] ** 2 + x[2] ** 2 + 2 * x[3] ** 2
    f[2] =  f[2] - x[0] - x[3] - 10

    f[3] =  x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 2 * x[0] 
    f[3] =  f[3] - x[1] - x[3] - 5

    f[1] = 10 * f[1] + f[0]
    f[2] = 10 * f[2] + f[0]
    f[3] = 10 * f[3] + f[0]
    
    return np.max(f)
