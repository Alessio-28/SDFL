# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

n: int = 50
name: str = f"cb3({n})"
starting_point: npt.NDArray[np.float64] = 2 * np.ones(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = np.empty((_m, n-1), dtype=np.float64)
    f[0] = x[:-1]**4 + x[1:]**2
    f[1] = (2 - x[:-1])**2 + (2 - x[1:])**2
    f[2] = 2 * np.exp(-x[:-1] + x[1:])

    return np.sum(np.max(f, axis=0))

_m: int = 3
