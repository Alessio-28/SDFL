# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "polak 3"
n: int = 11
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype=np.float64)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = x - _B
    np.square(f, out=f)
    np.exp(f, out=f)

    return np.max(np.vecdot(_A, f))


def _compute_A_B(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    k = np.arange(n, dtype=np.float64)
    i = np.arange(1, n, dtype=np.float64)[:, np.newaxis]

    A = i + k
    B = np.sin(i - 1 + 2 * k)
    return (A, B)


_A, _B = _compute_A_B(n)

del _compute_A_B
