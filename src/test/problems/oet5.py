# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "oet5"
n: int = 4
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype=np.float64)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = (x[:2] @ _U) + x[2]
    np.square(f, out=f)
    return np.max(np.abs(x[3] - f - _sqrtT))


_T = np.linspace(0.25, 1, 21, dtype=np.float64)
_U = np.array([_T, _T**2])
_sqrtT = np.sqrt(_T)

del _T
