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
    return np.max(np.sum(_A * np.exp((x[:, np.newaxis] - _B)**2), axis=0))

def _compute_A_B() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    j = np.arange(n, dtype=np.float64)
    i = j[1:]
    j = j[:, np.newaxis]
    return (j + i, np.sin(i - 1 + 2*j))

_A, _B = _compute_A_B()
