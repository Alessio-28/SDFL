# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "steiner 2"

# fmt: off
_A = np.array(
    [
        [0, 2,  3,    4, 5, 6],
        [2, 3, -1, -0.5, 2, 2],
    ],
    dtype=np.float64,
)
# fmt: on

n: int = _A.size


def _compute_starting_point(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    xbar = np.zeros_like(A)

    xbar[:, 0] = (A[:, 0] + A[:, 1]) / 3

    for j in range(1, xbar.shape[1] - 1):
        xbar[:, j] = (xbar[:, j - 1] + A[:, j] + A[:, j + 1]) / 3

    terms = np.array([5.5, -1], dtype=np.float64)
    xbar[:, -1] = (xbar[:, -2] + A[:, -1] + terms) / 3

    return xbar.ravel()


starting_point: npt.NDArray[np.float64] = _compute_starting_point(_A)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    x1 = x[:_m]
    x2 = x[_m:]

    f1 = np.sqrt(x1[0] ** 2 + x2[0] ** 2)
    f2 = np.sqrt((5.5 - x1[-1]) ** 2 + (1 + x2[-1]) ** 2)
    f3 = np.sqrt((_A[0] - x1) ** 2 + (_A[1] - x2) ** 2) @ _P
    f4 = np.sqrt((x1[:-1] - x1[1:]) ** 2 + (x2[:-1] - x2[1:]) ** 2) @ _P_TILDE

    return f1 + f2 + f3 + f4


_m: int = n // 2
_P = np.array([2, 1, 1, 5, 1, 1], dtype=np.float64)
_P_TILDE = np.array([1, 1, 2, 3, 2], dtype=np.float64)

del _compute_starting_point
