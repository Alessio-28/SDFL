# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "elattar"
starting_point: npt.NDArray[np.float64] = np.array([2, 2, 7, 0, -2, 1], dtype = np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    m: int = 51
    t: np.float64
    y: npt.NDArray[np.float64] = np.zeros(m,     dtype = np.float64)
    f: npt.NDArray[np.float64] = np.zeros(m * 2, dtype = np.float64)
    for i in range(m):
        t = i / 10
        y[i] = np.exp(t) / 2 - np.exp(-2 * t)
        y[i] = y[i] + np.exp(-3 * t) / 2
        y[i] = y[i] + 1.5 * np.exp(-1.5 * t) * np.sin(7 * t)
        y[i] = y[i] + np.exp(-2.5 * t) * np.sin(5 * t)
        f[i] = x[0] * np.exp(-x[1] * t) * np.cos(x[2] * t + x[3])
        f[i] = f[i] + x[4] * np.exp(-x[5] * t) - y[i]

    # for i in range(m, m * 2):
    #     f[i] = -f[i - m]
    
    f[m:] = -f[:m]

    return np.max(f)

