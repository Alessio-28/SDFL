# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "l1hilb"
n: int = 50
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.abs(x @ _W) # pyright: ignore[reportReturnType]

def _compute_weights(n: int) -> npt.NDArray[np.float64]:
    harmonics = 1 / np.arange(1, 2*n, dtype=np.float64)
    return np.convolve(harmonics, np.ones(n), mode="valid")

_W = _compute_weights(n)

del _compute_weights
