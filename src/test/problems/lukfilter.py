# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "filter"
starting_point: npt.NDArray[np.float64] = np.array([0, 1, 0, -0.15, 0, -0.68, 0, -0.72, 0.37], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    def expression_term(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (p[0] + (1+p[1])*_COS_ETA)**2 + ((1-p[1])*_SIN_ETA)**2

    def expression(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return expression_term(p[:2]) / expression_term(p[2:])

    A = expression(x[:4])
    B = expression(x[4:8])

    return np.max(np.abs(x[8]*np.sqrt(A)*np.sqrt(B) - _U))

def _compute_T() -> npt.NDArray[np.float64]:
    m: int = 41
    mid: int = m // 2
    r1: int = 6
    r2: int = m - r1
    t = np.zeros(m, dtype=np.float64)
    tmp = 0.03 * np.arange(r2, dtype=np.float64)

    t[:r1]        = 0.01 * np.arange(r1, dtype=np.float64)
    t[r1:r1+r2]   = 0.07 + tmp
    t[mid]        = 0.5
    t[-r1-r2:-r1] = 0.54 + tmp
    t[-r1:]       = 0.95 + t[:r1]

    return t

_T = _compute_T()

_ETA = np.pi * _T
_COS_ETA = np.cos(_ETA)
_SIN_ETA = np.sin(_ETA)

_U = np.abs(1 - 2*_T)
