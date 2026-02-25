# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "oet6"
starting_point: npt.NDArray[np.float64] = np.array([1, 1, -3, -1], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f1 = x[0] * np.exp(x[2]*_T)
    f2 = x[1] * np.exp(x[3]*_T)
    return np.max(np.abs(f1 + f2 - _overT))

_T = np.linspace(-0.5, 0.5, 21, dtype=np.float64)
_overT = 1/(_T+1)
