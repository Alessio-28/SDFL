# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "oet6"
startin_point: npt.NDArray[np.float64] = np.array([1, 1, -3, -1], dtype = np.float64)
n: int = startin_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    m: int = 21
    I = np.arange(m, dtype = np.float64).reshape(m, -1)
    t = -0.5 + I / 20;
    f = x[0] * np.exp(x[2] * t) + x[1] * np.exp(x[3] * t) - (t + 1) ** (-1)

    return np.max(np.abs(f))
