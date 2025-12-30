# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "elattar"
startp    : npt.NDArray[np.float64] = np.array([2, 2, 7, 0, -2, 1], dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 3
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    t : npt.NDArray[np.float64] = np.zeros(51,  dtype = np.float64)
    y : npt.NDArray[np.float64] = np.zeros(51,  dtype = np.float64)
    f : npt.NDArray[np.float64] = np.zeros(102, dtype = np.float64)
    for i in range(1, 52):
        t[i - 1] = (i - 1) / 10
        y[i - 1] = np.exp(t[i - 1]) / 2 - np.exp(-2 * t[i - 1])
        y[i - 1] = y[i - 1] + np.exp(-3 * t[i - 1]) / 2
        y[i - 1] = y[i - 1] + 1.5 * np.exp(-1.5 * t[i - 1]) * np.sin(7 * t[i - 1])
        y[i - 1] = y[i - 1] + np.exp(-2.5 * t[i - 1]) * np.sin(5 * t[i - 1])
        f[i - 1] = x[0] * np.exp(-x[1] * t[i - 1]) * np.cos(x[2] * t[i - 1] + x[3])
        f[i - 1] = f[i - 1] + x[4] * np.exp(-x[5] * t[i - 1]) - y[i - 1]

    for i in range(52, 103):
        f[i - 1] = -f[i - 52]
    
    z : np.float64 = np.max(f)
    
    return z

