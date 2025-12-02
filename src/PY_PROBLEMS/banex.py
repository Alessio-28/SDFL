# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "banex"
n         : int = 2
startp    : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)
lb        : npt.NDArray[np.float64] = np.array([-50,   0], dtype = np.float64)
ub        : npt.NDArray[np.float64] = np.array([ 10, 100], dtype = np.float64)
nint      : int = 2
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64)
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64) 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    return (x[0] - 1) ** 2 + 100 * (x[0] ** 2 - x[1]) ** 2
