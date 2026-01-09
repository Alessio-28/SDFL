# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "gill"
n: int = 10
starting_point: npt.NDArray[np.float64] = -0.1 * np.ones(n, dtype = np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    fx: npt.NDArray[np.float64] = np.zeros(3, dtype = np.float64)
    fx[0] = np.sum((x - 1) ** 2) + 0.001 * np.sum((x ** 2 - 0.25) ** 2)
    
    j = np.arange(2, 11)
    i = np.arange(2, 31)
    A = np.tile((x[j - 1] * (j - 1)).reshape(-1, 1), (1, 29)) * np.tile(((i - 1) / 29), (9, 1)) ** np.tile((j - 2).reshape(-1, 1), (1, 29))
    
    j = np.arange(1, 11)
    B = np.tile(((i - 1) / 29), (10, 1)) ** np.tile((x * (j - 1)).reshape(-1, 1), (1, 29))
    fx[1] = np.sum((np.sum(A, 0) - (np.sum(B, 0)) ** 2 - 1) ** 2) + x[0] ** 2 + (x[1] - x[0] ** 2 - 1) ** 2
    
    i = np.arange(2, 11)
    fx[2] = np.sum(100 * (x[i - 1] - x[i - 2] ** 2) ** 2 + (1 - x[i - 1]) ** 2)
    

    return np.max(fx)
