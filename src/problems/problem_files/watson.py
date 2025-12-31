# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name   : str = "watson"
n      : int = 20
startp : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1, 1)
    f = np.zeros(31, dtype = np.float64)
    I = np.tile(np.arange(1, 30), (20, 1)) / 29
    J = np.tile(np.arange(20).reshape(-1, 1), (1, 29)) 
    X = np.tile(x, (1, 29))
    f[0:29] = np.sum(J * X * I ** (J - 1), axis = 0) - np.sum(X * I ** J, axis = 0) ** 2
    f[29] = x[0]
    f[30] = x[1] - x[0] ** 2 - 1

    return np.max(np.abs(f))
