# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "mxhilb"
n: int = 50
startin_point: npt.NDArray[np.float64] = np.ones(n, dtype = np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    i = np.arange(1, n + 1, dtype = np.float64)
    j = np.arange(1, n + 1, dtype = np.float64)
    I = np.tile(i, (n, 1))
    J = np.tile(j.reshape(-1, 1), (1, n))
    X = np.tile(x.reshape(-1, 1), (1, n))

    return np.max(np.abs(np.sum(X / (I + J - 1))))
