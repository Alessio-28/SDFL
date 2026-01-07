# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "evd61"
starting_point: npt.NDArray[np.float64] = np.array([2, 2, 7, 0, -2, 1], dtype = np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    t: npt.NDArray[np.float64] = 0.1 * (np.arange(1, 52, dtype = np.float64) - 1)
    z: npt.NDArray[np.float64] = 0.5 * np.exp(-t) - np.exp(-2 * t) + 0.5 * np.exp(-3 * t) + 1.5 * np.exp(-1.5 * t) * np.sin(7 * t) + np.exp(-2.5 * t) * np.sin(5 * t)
    f: npt.NDArray[np.float64] = x[0] * np.exp(-x[1] * t) * np.cos(x[2] * t + x[3]) + x[4] * np.exp(-x[5] * t) - z

    return np.max(np.abs(f))
