# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "shell dual"
n: int = 15
starting_point: npt.NDArray[np.float64] = 1e-4 * np.ones(n, dtype=np.float64)
starting_point[6] = 60

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    y = x[:_m]
    z  = x[_m:]

    Q = np.sum(np.minimum(x, 0))
    P = (_A @ y) - 2*z*_C - 3*_D * z**2 - _E

    return 2*np.abs(np.dot(_D, z**3)) + np.sum(z * z[:, np.newaxis] * _C) - np.dot(_B, y) + 100*(np.sum(np.maximum(P, 0)) - Q)

_A = np.array(
    [
        [-16,  0, -3.5,  0,    0,  2, -1, -1, 1, 1],
        [  2, -2,    0, -2,   -9,  0, -1, -2, 2, 1],
        [  0,  0,    2,  0,   -2, -4, -1, -3, 3, 1],
        [  1,  4,    0, -4,    1,  0, -1, -2, 4, 1],
        [  0,  2,    0, -1, -2.8,  0, -1, -1, 5, 1]
    ],
    dtype=np.float64
)
_B = np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1], dtype=np.float64)
_C = np.array(
    [
        [ 30, -20, -10,  32, -10],
        [-20,  39,  -6, -31,  32],
        [-10,  -6,  10,  -6, -10],
        [ 32, -31,  -6,  39, -20],
        [-10,  32, -10, -20,  30]
    ],
    dtype=np.float64
)
_D = np.array([4, 8, 10, 6, 2], dtype=np.float64)
_E = np.array(
    [
        [-15],
        [-27],
        [-36],
        [-18],
        [-12]
    ],
    dtype=np.float64
)

_m = _A.shape[1]
