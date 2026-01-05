# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "wong1"
startin_point: npt.NDArray[np.float64] = np.array([1, 2, 0, 4, 0, 1, 1], dtype = np.float64)
n: int = startin_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1, 1)
    f = np.zeros(5, dtype = np.float64)
    f[0] = (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2 + x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
    f[1] = f[0] + 10 * ( 2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2 + 5 * x[4] - 127)
    f[2] = f[0] + 10 * ( 7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4] - 282)
    f[3] = f[0] + 10 * (23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6] - 196)
    f[4] = f[0] + 10 * ( 4 * x[0] ** 2 + x[1] ** 2 - 3 * x[0] * x[1] + 2 * x[2] ** 2 + 5 * x[5] - 11 * x[6])

    return np.max(f)
