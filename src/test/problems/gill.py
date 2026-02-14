# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "gill"
n: int = 10
starting_point: npt.NDArray[np.float64] = -0.1 * np.ones(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    XJ = x * _J

    A = XJ[1:, np.newaxis] * _IpowJ
    B = _I ** XJ[:, np.newaxis]

    U = np.sum(A, axis=0)
    V = np.sum(B, axis=0)

    f = np.empty(_l, dtype=np.float64)
    f[0] = np.sum((x - 1)**2) + np.sum((x**2 - 0.25)**2)/1000
    f[1] = np.sum((U - V**2 - 1)**2) + x[0]**2 + (x[1] - x[0]**2 - 1)**2
    f[2] = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[1:])**2)

    return np.max(f)

_l: int = 3
_m: int = 29
_I = (1 + np.arange(_m, dtype=np.float64)) / _m
_J = np.arange(n, dtype=np.float64)
_IpowJ = _I**_J[:-1, np.newaxis]
