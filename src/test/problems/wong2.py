# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "wong2"
starting_point: npt.NDArray[np.float64] = np.array([2, 3, 5, 5, 1, 2, 7, 3, 6, 10], dtype = np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = np.empty(_m, dtype = np.float64)

    f[0] = (x[0]**2 + x[1]**2 + (x[2]-10)**2 + 4*(x[3]-5)**2 + (x[4]-3)**2
            + 2*(x[5]-1)**2 + 5*x[6]**2 + 7*(x[7]-11)**2 + 2*(x[8]-10)**2 + (x[9]-7)**2
            - 14*x[0] - 16*x[1] + x[0]*x[1] + 45)

    f[1] = 3*(x[0]-2)**2 + 4*(x[1]-3)**2 + 2*x[2]**2 - 7*x[3] - 120
    f[2] = 5*x[0]**2 + (x[2]-6)**2 + 8*x[1] - 2*x[3] - 40
    f[3] = 0.5*(x[0]-8)**2 + 2*(x[1]-4)**2 + 3*x[4]**2 - x[5] - 30
    f[4] = x[0]**2 + 2*(x[1]-2)**2 + 14*x[4] - 6*x[5] - 2*x[0]*x[1]
    f[5] = 12*(x[8]-8)**2 - 3*x[0] + 6*x[1] - 7*x[9]
    f[6] = 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7] - 105
    f[7] = 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]
    f[8] = -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12

    f[1:] = f[0] + 10*f[1:]

    return np.max(f)

_m: int = 9
