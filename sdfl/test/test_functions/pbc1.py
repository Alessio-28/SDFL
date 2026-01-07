# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "pbc1"
starting_point: npt.NDArray[np.float64] = np.array([0, -1, 10, 1, 10], dtype = np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    t = -1 + (2 / 29) * np.arange(30, dtype = np.float64)
    f = (x[0] + x[1] * t + x[2] * t ** 2) / (1 + x[3] * t + x[4] * t ** 2) - (np.sqrt((8 * t - 1) ** 2 + 1) * np.arctan(8 * t)) / (8 * t)

    return np.max(np.abs(f))
