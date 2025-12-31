# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name : str = "steiner 2"
aa = np.array(
    [
        [0,    2],
        [2,    3],
        [3,   -1],
        [4, -0.5],
        [5,    2],
        [6,    2]
    ],
    dtype = np.float64
)

m : int = aa.shape[0]
n : int = m * 2

j = np.arange(2, m)

xbar = np.zeros(n, dtype = np.float64)
xbar[0] = (aa[0, 0] + aa[1, 0]) / 3
xbar[j - 1] = (xbar[j - 2] + aa[j - 1, 0] + aa[j, 0]) / 3
xbar[m - 1] = (xbar[m - 2] + aa[m - 1, 0] + 5.5) / 3
xbar[m] = (aa[0, 1] + aa[1, 1]) / 3
xbar[j + m - 1] = (xbar[j - 2 + m] + aa[j - 1, 1] + aa[j, 1]) / 3
xbar[2 * m - 1] = (xbar[2 * m - 2] + aa[m - 1, 1] - 1) / 3

startp : npt.NDArray[np.float64] = xbar

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1, 1)
    a = np.array(
        [
            [0,    2],
            [2,    3],
            [3,   -1],
            [4, -0.5],
            [5,    2],
            [6,    2]
        ],
        dtype = np.float64
    )
    p = np.array(
        [
            [2],
            [1],
            [1],
            [5],
            [1],
            [1]
        ],
        dtype = np.float64
    )
    ptilde = np.array(
        [
            [1],
            [1],
            [2],
            [3],
            [2]
        ],
        dtype = np.float64
    )
    jm  = np.arange(1, m + 1)
    jm1 = np.arange(1, m)

    return np.sqrt(x[0] ** 2 + x[m] ** 2) + np.sqrt((5.5 - x[m - 1]) ** 2 + (-1 - x[2 * m - 1]) ** 2) + np.sum(p * np.sqrt((a[jm - 1, 0].reshape(-1, 1) - x[jm - 1]) ** 2 + (a[jm - 1, 1].reshape(-1, 1) - x[jm + m - 1]) ** 2), axis = 0) + np.sum(ptilde * np.sqrt((x[jm1 - 1] - x[jm1]) ** 2 + (x[jm1 + m - 1] - x[jm1 + m]) ** 2), axis = 0)
