# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "mxhilb"
n: int = 50
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.abs(np.sum(x[:, np.newaxis] / _J))

def _compute_J() -> npt.NDArray[np.float64]:
    _I = 1 + np.arange(n, dtype=np.float64)
    return _I + _I[:, np.newaxis] - 1

_J = _compute_J()
