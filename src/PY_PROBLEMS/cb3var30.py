# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "cb3(30)"
n         : int = 30
startp    : npt.NDArray[np.float64] = 2 * np.ones(n, dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
nint      : int = 15
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    y : np.float64 = np.float64(0)
    for i in range(n - 1):
        fx1 : np.float64 = x[i] ** 4 + x[i + 1] ** 2
        fx2 : np.float64 = (2 - x[i]) ** 2+(2 - x[i + 1]) ** 2
        fx3 : np.float64 = 2 * np.exp(-x[i] + x[i + 1])
        y += np.max([fx1, fx2, fx3])
        
    return y
