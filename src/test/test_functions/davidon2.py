# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "davidon2"
starting_point: npt.NDArray[np.float64] = np.array([25, 5, -5, -1], dtype = np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    m: int = 21
    f: npt.NDArray[np.float64] = np.zeros(m, dtype = np.float64)
    for i in range(m):
        t = 0.25 + (0.75 / 20) * i
        f[i] = x[3] - (x[0] * t ** 2 + x[1] * t + x[2]) ** 2 - np.sqrt(t)

    return np.max(np.abs(f))
