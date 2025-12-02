# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "l1hilb(20)"
startp    : npt.NDArray[np.float64] = np.ones(20, dtype = np.float64)
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
    i : npt.NDArray[np.float64] = np.arange(1, 21, dtype = np.float64)
    j : npt.NDArray[np.float64] = np.arange(1, 21, dtype = np.float64)
    I = np.tile(i, (20, 1))
    J = np.tile(j.reshape(-1, 1), (1, 20))
    X = np.tile(x.reshape(-1, 1), (1, 20))
    y : np.float64 = np.sum(np.abs(np.sum(X / (I + J - 1))))
    return y
