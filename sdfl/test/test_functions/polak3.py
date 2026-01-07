# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "polak 3"
n: int = 11
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype = np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    J = np.tile(np.arange(n, dtype = np.float64).reshape(-1, 1), (1, n - 1))
    I = np.tile(np.arange(1, n, dtype = np.float64), (n, 1))
    f: npt.NDArray[np.float64] = np.sum((J + I) * np.exp((np.tile(x.reshape(-1, 1), (1, n)) - np.sin(I - 1 + 2 * J)) ** 2), 0)

    return np.max(f)

