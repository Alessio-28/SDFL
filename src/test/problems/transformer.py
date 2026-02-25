# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "transformer"
starting_point: npt.NDArray[np.float64] = np.array([0.8, 1.5, 1.2, 3, 0.8, 6], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    tmp = x[2*_loop_range][:, np.newaxis]

    Tx = _THETA * (x[2*_loop_range - 1][:, np.newaxis])
    a = np.cos(Tx)
    b = np.sin(Tx)
    c = b.copy()
    b /= tmp
    c *= tmp

    v = np.empty((_l, _m), dtype=np.complex128)
    w = np.empty((_l, _m), dtype=np.complex128)
    v[-1] = 1
    w[-1] = 10
    for k in _loop_range:
        v[k] = a[k]*v[k+1] + b[k]*w[k+1]*1j
        w[k] = a[k]*w[k+1] - c[k]*v[k+1].imag + b[k]*v[k+1].real*1j

    f = np.abs(1 - 2*v[0] / (w[0]+v[0]))

    return np.max(f)

_THETA = (np.pi / 2) * np.array([0.5, 0.6, 0.7, 0.77, 0.9, 1, 1.1, 1.23, 1.3, 1.4, 1.5], dtype=np.float64)
_m: int = _THETA.size

_l: int = 4
_loop_range = np.arange(_l-1, dtype=int)[::-1]
