# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name   : str = "elattar"
startp : npt.NDArray[np.float64] = np.array([2, 2, 7, 0, -2, 1], dtype = np.float64)
n      : int = startp.size

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    m : int = 51
    t : npt.NDArray[np.float64] = np.zeros(m,     dtype = np.float64)
    y : npt.NDArray[np.float64] = np.zeros(m,     dtype = np.float64)
    f : npt.NDArray[np.float64] = np.zeros(m * 2, dtype = np.float64)
    for i in range(m):
        t[i] = i / 10
        y[i] = np.exp(t[i]) / 2 - np.exp(-2 * t[i])
        y[i] = y[i] + np.exp(-3 * t[i]) / 2
        y[i] = y[i] + 1.5 * np.exp(-1.5 * t[i]) * np.sin(7 * t[i])
        y[i] = y[i] + np.exp(-2.5 * t[i]) * np.sin(5 * t[i])
        f[i] = x[0] * np.exp(-x[1] * t[i]) * np.cos(x[2] * t[i] + x[3])
        f[i] = f[i] + x[4] * np.exp(-x[5] * t[i]) - y[i]

    # for i in range(m, m * 2):
    #     f[i] = -f[i - m]
    
    f[m:] = -f[:m]

    return np.max(f)

