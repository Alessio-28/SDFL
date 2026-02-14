# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "exp"
starting_point: npt.NDArray[np.float64] = np.array([0.5, 0, 0, 0, 0], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.max((x[0] + x[1]*_T) / (1 + x[2]*_T + x[3]*_T2 + x[4]*_T3) - _expT)

_m: int = 21
_T = -1 + np.arange(_m, dtype=np.float64)/10
_T2 = _T**2
_T3 = _T**3
_expT = np.exp(_T)
