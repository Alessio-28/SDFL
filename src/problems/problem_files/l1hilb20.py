# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

n      : int = 20
name   : str = f"l1hilb({n})"
startp : npt.NDArray[np.float64] = np.ones(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    i : npt.NDArray[np.float64] = np.arange(1, n + 1, dtype = np.float64)
    j : npt.NDArray[np.float64] = np.arange(1, n + 1, dtype = np.float64)
    I = np.tile(i, (n, 1))
    J = np.tile(j.reshape(-1, 1), (1, n))
    X = np.tile(x.reshape(-1, 1), (1, n))

    return np.sum(np.abs(np.sum(X / (I + J - 1))))
