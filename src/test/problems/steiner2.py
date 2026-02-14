# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "steiner 2"

_A = np.array(
    [
        [0, 2,  3,    4, 5, 6],
        [2, 3, -1, -0.5, 2, 2],
    ],
    dtype=np.float64
)

n: int = _A.size

def _compute_starting_point() -> npt.NDArray[np.float64]:
    xbar = np.zeros(_A.shape, dtype=np.float64)

    xbar[:, 0]    = (               _A[:, 0]    + _A[:, 1] )/3
    xbar[:, 1:-1] = (xbar[:, :-2] + _A[:, 1:-1] + _A[:, 2:])/3

    xbar[0, -1]   = (xbar[0, -2]  + _A[0, -1]   + 5.5     )/3
    xbar[1, -1]   = (xbar[1, -2]  + _A[1, -1]   - 1       )/3

    return xbar.reshape(xbar.size)

starting_point: npt.NDArray[np.float64] = _compute_starting_point()

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    x1 = x[:_m]
    x2 = x[_m:]

    f = (np.sqrt(x1[0]**2 + x2[0]**2)
         + np.sqrt((5.5 - x1[-1])**2 + (1 + x2[-1])**2)
         + np.sum(_P * np.sqrt((_A[0] - x1)**2 + (_A[1] - x2)**2), axis=0)
         + np.sum(_P_TILDE * np.sqrt((x1[:-1] - x1[1:])**2 + (x2[:-1] - x2[1:])**2), axis=0))

    return f

_m: int = n // 2
_P = np.array([2, 1, 1, 5, 1, 1], dtype=np.float64)
_P_TILDE = np.array([1, 1, 2, 3, 2], dtype=np.float64)
