# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "wong1"
starting_point: npt.NDArray[np.float64] = np.array([1, 2, 0, 4, 0, 1, 1], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = (_A @ (x*x)) + (_B @ x) + _C
    f[0] += 10*x[4]**6 + x[2]**4 + x[6]**4 - 4*x[5]*x[6]
    f[1] += 3*x[1]**4

    return f[0] + 10*np.maximum(0, np.max(f[1:]))

_A = np.array(
    [
        [1, 5,  0, 3, 0, 7, 0],
        [2, 0,  0, 4, 0, 0, 0],
        [0, 0, 10, 0, 0, 0, 0],
        [0, 1,  0, 0, 0, 6, 0],
        [4, 1,  2, 0, 0, 0, 0]
    ],
    dtype=np.float64
)

_B = np.array(
    [
        [-20, -120,  0, -66,  0, -10,  -8],
        [  0,    0,  1,   0,  5,   0,   0],
        [  7,    3,  0,   1, -1,   0,   0],
        [ 23,    0,  0,   0,  0,   0,  -8],
        [  0,    0,  0,   0,  0,   5, -11]
    ],
    dtype=np.float64
)

_C = np.array([1183, -127, -282, -196, -11], dtype=np.float64)
