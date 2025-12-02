# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "crescent"
n         : int = 2
startp    : npt.NDArray[np.float64] = np.array([-1.5, 2], dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
nint      : int = 2
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64)
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64) 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    f1 : np.float64 =   x[0]**2 + (x[1]-1)**2 + x[1] - 1
    f2 : np.float64 = - x[0]**2 - (x[1]-1)**2 + x[1] + 1

    return np.maximum(f1, f2)
