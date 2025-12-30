# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "watson"
startp    : npt.NDArray[np.float64] = np.zeros(20, dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 10
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1, 1)
    f = np.zeros(31, dtype = np.float64)
    I = np.tile(np.arange(3, 32), (20, 1))
    J = np.tile(np.arange(1, 21).reshape(-1, 1), (1, 29)) 
    X = np.tile(x, (1, 29))
    f[0:29] = np.sum((J - 1) * X * ((I - 2) / 29) ** (J - 2), axis = 0) - np.sum(X * ((I - 2) / 29) ** (J - 1), axis = 0) ** 2
    f[29] = x[0]
    f[30] = x[1] - x[0] ** 2 - 1
    y = np.max(np.abs(f))
    return y

