# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "osborne2"
starting_point: npt.NDArray[np.float64] = np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = _Y - x[0]*np.exp(-x[4]*_T) - x[1]*np.exp(-x[5] * (_T-x[8])**2) - x[2]*np.exp(-x[6] * (_T-x[9])**2) - x[3]*np.exp(-x[7] * (_T-x[10])**2)
    
    return np.max(np.abs(f))

_Y = np.array(
    [
        1.366,
        1.191,
        1.112,
        1.013,
        0.991,
        0.885,
        0.831,
        0.847,
        0.786,
        0.725,
        0.746,
        0.679,
        0.608,
        0.655,
        0.616,
        0.606,
        0.602,
        0.626,
        0.651,
        0.724,
        0.649,
        0.649,
        0.694,
        0.644,
        0.624,
        0.661,
        0.612,
        0.558,
        0.553,
        0.495,
        0.500,
        0.423,
        0.395,
        0.375,
        0.372,
        0.391,
        0.396,
        0.405,
        0.428,
        0.429,
        0.523,
        0.562,
        0.607,
        0.653,
        0.672,
        0.708,
        0.633,
        0.668,
        0.645,
        0.632,
        0.591,
        0.559,
        0.597,
        0.625,
        0.739,
        0.710,
        0.729,
        0.720,
        0.636,
        0.581,
        0.428,
        0.292,
        0.162,
        0.098,
        0.054
    ],
    dtype=np.float64
)
_T = np.arange(_Y.size, dtype=np.float64) / 10
