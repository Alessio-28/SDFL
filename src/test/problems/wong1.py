# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "wong1"
starting_point: npt.NDArray[np.float64] = np.array([1, 2, 0, 4, 0, 1, 1], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = np.empty(_m, dtype=np.float64)

    f[0] = (10*x[4]**6 + x[2]**4 + x[6]**4
            + (x[0]-10)**2 + 5*(x[1]-12)**2 + 3*(x[3]-11)**2 + 7*x[5]**2
            - 10*x[5] - 8*x[6] - 4*x[5]*x[6])

    f[1] = 3*x[1]**4 + 2*x[0]**2 + 4*x[3]**2 + x[2] + 5*x[4] - 127
    f[2] = 10*x[2]**2 + 7*x[0] + 3*x[1] + x[3] - x[4] - 282
    f[3] = x[1]**2 + 6*x[5]**2 + 23*x[0] - 8*x[6] - 196
    f[4] = 4*x[0]**2 + x[1]**2 + 2*x[2]**2 + 5*x[5] - 11*x[6] - 3*x[0]*x[1]

    f[1:] = f[0] + 10*f[1:]

    return np.max(f)

_m: int = 5
