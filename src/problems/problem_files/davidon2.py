# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "davidon2"
startp    : npt.NDArray[np.float64] = np.array([25, 5, -5, -1], dtype = np.float64)
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
    f : npt.NDArray[np.float64] = np.zeros(21, dtype = np.float64)
    for i in range(21):
        t = 0.25 + (0.75 / 20) * i
        f[i] = x[3] - (x[0] * t ** 2 + x[1] * t + x[2]) ** 2 - np.sqrt(t)
    y : np.float64 = np.max(np.abs(f));
    return y
