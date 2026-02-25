# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "cb2"
n: int = 2
starting_point: npt.NDArray[np.float64] = 2 * np.ones(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = np.empty(3, dtype=np.float64)
    f[0] = x[0]**2 + x[1]**4
    f[1] = (2 - x[0])**2 + (2 - x[1])**2
    f[2] = 2 * np.exp(x[1] - x[0])

    return np.max(f)
