# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "kowalik-osborne"
startp    : npt.NDArray[np.float64] = np.array([0.25, 0.39, 0.415, 0.39], dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 2
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    z : npt.NDArray[np.float64] = np.array(
        [
            [0.1957,      4],
            [0.1947,      2],
            [0.1735,      1],
            [0.1600,    0.5],
            [0.0844,   0.25],
            [0.0627, 0.1670],
            [0.0456, 0.1250],
            [0.0342,    0.1],
            [0.0323, 0.0833],
            [0.0235, 0.0714],
            [0.0246, 0.0625]
        ],
        dtype = np.float64
    )
    u : npt.NDArray[np.float64] = z[:,1]
    f : npt.NDArray[np.float64] = (x[0] * (u ** 2 + x[1] * u)) / (u ** 2 + x[2] * u + x[3]) - z[:,0]
    y : np.float64 = np.max(np.abs(f));
    return y
