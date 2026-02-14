# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "davidon2"
starting_point: npt.NDArray[np.float64] = np.array([25, 5, -5, -1], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.max(np.abs(x[3] - (x[0]*_T2 + x[1]*_T + x[2])**2 - _sqrtT))

_m: int = 21
_T = 0.25 + (0.75/20)*np.arange(_m, dtype=np.float64)
_T2 = _T**2
_sqrtT = np.sqrt(_T)
