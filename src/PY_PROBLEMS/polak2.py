# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "polak 2"
startp    : npt.NDArray[np.float64] = 0.1 * np.ones(10, dtype = np.float64)
startp[0] = 100
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
    e2 : npt.NDArray[np.float64] = np.zeros(10, dtype = np.float64); e2[1] = 1
    w : npt.NDArray[np.float64] = x + 2 * e2
    f1 : np.float64 = np.exp(1e-8 * w[0] ** 2 + w[1] ** 2 + w[2] ** 2 + 4 * w[3] ** 2 + w[4] ** 2 + w[5] ** 2 + w[6] ** 2 + w[7] ** 2 + w[8] ** 2 + w[9] ** 2)
    w = x - 2 * e2
    f2 : np.float64 = np.exp(1e-8 * w[0] ** 2 + w[1] ** 2 + w[2] ** 2 + 4 * w[3] ** 2 + w[4] ** 2 + w[5] ** 2 + w[6] ** 2 + w[7] ** 2 + w[8] ** 2 + w[9] ** 2)    
    y = np.max((f1, f2))
    return y

