# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "rosen-suzuki"
n: int = 4
starting_point: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = (_A @ (x * x)) + (_B @ x) + _C

    return f[0] + 10 * np.maximum(0, np.max(f[1:]))


# fmt: off
_A = np.array(
    [
        [1, 1, 2, 1],
        [1, 1, 1, 1],
        [1, 2, 1, 2],
        [1, 1, 1, 0],
    ],
    dtype=np.float64,
)
_B = np.array(
    [
        [-5, -5, -21,  7],
        [ 1, -1,   1, -1],
        [-1,  0,   0, -1],
        [ 2, -1,   0, -1],
    ],
    dtype=np.float64,
)
_C = np.array([0, -8, -10, -5], dtype=np.float64)
