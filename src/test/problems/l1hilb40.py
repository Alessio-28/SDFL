# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

n: int = 40
name: str = f"l1hilb({n})"
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.abs(np.sum(x[:, np.newaxis] / _I))

def _compute_I() -> npt.NDArray[np.float64]:
    i = 1 + np.arange(n, dtype=np.float64)
    return i + i[:, np.newaxis] - 1

_I = _compute_I()
