# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "polak 2"
n: int = 10
starting_point: npt.NDArray[np.float64] = 0.1 * np.ones(n, dtype=np.float64)
starting_point[0] = 100

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    w = np.array([x+_E2, x-_E2])
    np.square(w, out=w)
    f = np.exp(w @ _C)

    return np.maximum(f[0], f[1])

_C  = np.array([1e-8, 1, 1, 4, 1, 1, 1, 1, 1, 1], dtype=np.float64)
_E2 = np.array([   0, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
