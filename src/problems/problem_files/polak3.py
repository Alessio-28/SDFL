# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "polak 3"
startp    : npt.NDArray[np.float64] = np.ones(11, dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 5
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    J = np.tile(np.arange(0, 11, dtype = np.float64).reshape(-1, 1), (1, 10))
    I = np.tile(np.arange(1, 11, dtype = np.float64), (11, 1))
    f : npt.NDArray[np.float64] = np.sum((J + I) * np.exp((np.tile(x.reshape(-1, 1), (1, 10)) - np.sin(I - 1 + 2 * J)) ** 2), 0)
    y = np.max(f)
    return y

