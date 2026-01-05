# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "oet5"
n: int = 4
startin_point: npt.NDArray[np.float64] = np.ones(n, dtype = np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    m: int = 21
    I = np.arange(m, dtype = np.float64).reshape(m, -1)
    t = 0.25 + (0.75 / 20) * I
    f = x[3] - (x[0] * t ** 2 + x[1] * t + x[2]) ** 2 - np.sqrt(t)

    return np.max(np.abs(f))
