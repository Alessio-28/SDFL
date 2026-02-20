# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "maxquad"
n: int = 10
starting_point: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.max(((_A @ x) - _B) @ x)

def _compute_A_B() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    i = 1 + np.arange(n, dtype=np.float64)
    j = i[:, np.newaxis]
    jT = j.T

    jk1 = np.triu(np.matmul(j, jT), 1)
    jk1 += jk1.T
    jk2 = np.triu(np.matmul(j, 1/jT), 1)
    jk2 += jk2.T

    m: int = 5
    k = 1 + np.arange(m)[:, np.newaxis]
    k_sin = np.sin(k)
    k_sin_3d = k_sin[:, :, np.newaxis]
    base = np.exp(jk2) * np.cos(jk1)
    base_sum_abs = np.sum(np.abs(base))

    rows, cols = np.diag_indices(n)
    A = base * k_sin_3d
    A[:, rows, cols] += (i/10) * k_sin + base_sum_abs * np.abs(k_sin)
    B = np.exp(i/k) * np.sin(i*k)
    return (A, B)

_A, _B = _compute_A_B()
