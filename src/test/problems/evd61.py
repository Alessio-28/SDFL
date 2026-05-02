# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "evd61"
starting_point: npt.NDArray[np.float64] = np.array(
    [2, 2, 7, 0, -2, 1],
    dtype=np.float64,
)
n: int = starting_point.size


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f1 = x[0] * np.exp(-x[1] * _T) * np.cos(x[2] * _T + x[3])
    f2 = x[4] * np.exp(-x[5] * _T)

    return np.max(np.abs(f1 + f2 - _exprT))


_T = np.linspace(0, 5, 51, dtype=np.float64)
_exprT = (
    np.exp(-_T) / 2
    - np.exp(-2 * _T)
    + np.exp(-3 * _T) / 2
    + np.exp(-2.5 * _T) * np.sin(5 * _T)
    + 1.5 * np.exp(-1.5 * _T) * np.sin(7 * _T)
)
