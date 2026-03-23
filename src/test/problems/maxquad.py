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

def _compute_A_B(n: int, m: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    j = 1 + np.arange(n, dtype=np.float64)
    k = 1 + np.arange(m, dtype=np.float64)[:, np.newaxis]

    prod  = j[:, np.newaxis] * j
    ratio = j[:, np.newaxis] / j
    ratio = np.minimum(ratio, ratio.T)

    np.fill_diagonal(prod, 0)
    np.fill_diagonal(ratio, 0)

    k_sin = np.sin(k)
    base = np.exp(ratio) * np.cos(prod)
    base_sum_abs = np.sum(np.abs(base))

    diag_idx = np.arange(n)
    A = base * k_sin[:, :, np.newaxis]
    A[:, diag_idx, diag_idx] += (j/10) * k_sin  + base_sum_abs * np.abs(k_sin)

    B = np.exp(j/k) * np.sin(j*k)
    
    return (A, B)

_A, _B = _compute_A_B(n, 5)

del _compute_A_B
