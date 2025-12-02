# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "shell dual"
startp    : npt.NDArray[np.float64] = 1e-4 * np.ones(15, dtype = np.float64)
startp[6] = 60
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 7
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)


def feval(x : npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1, 1)
    A = np.array(
        [
            [ -16,  2,  0,  1,    0],
            [   0, -2,  0,  4,    2],
            [-3.5,  0,  2,  0,    0],
            [   0, -2,  0, -4,   -1],
            [   0, -9, -2,  1, -2.8],
            [   2,  0, -4,  0,    0],
            [  -1, -1, -1, -1,   -1],
            [  -1, -2, -3, -2,   -1],
            [   1,  2,  3,  4,    5],
            [   1,  1,  1,  1,    1]
        ],
        dtype = np.float64
    )
 
    b = np.array(
        [
            [   -40],
            [    -2],
            [ -0.25],
            [    -4],
            [    -4],
            [    -1],
            [   -40],
            [   -60],
            [     5],
            [     1]
        ],
        dtype = np.float64
    )
    
    C = np.array(
        [
            [ 30, -20, -10,  32, -10],
            [-20,  39,  -6, -31,  32],
            [-10,  -6,  10,  -6, -10],
            [ 32, -31,  -6,  39, -20],
            [-10,  32, -10, -20,  30]
        ],
        dtype = np.float64
    )
     
    d = np.array(
        [
            [ 4],
            [ 8],
            [10],
            [ 6],
            [ 2]
        ],
        dtype = np.float64
    )
    
    ee = np.array([-15, -27, -36, -18, -12], dtype = np.float64)
    
    J10 = np.arange(1, 11)
    J5  = np.arange(1, 6)
    
    Q = np.sum(np.minimum(0, x))
    P = np.matmul(A.T, x[J10-1]) - 2 * C * x[J5 + 10-1] - 3 * d * x[J5 + 10 - 1] ** 2 - ee.T
    X = np.tile(x[J5 + 10-1], (1, 5))
    
    y = 2 * np.abs(np.sum(d * x[J5 + 10 - 1] ** 3)) + np.sum(np.sum(C * X * X.T)) - np.sum(b * x[J10 - 1]) + 100 * (np.sum(np.maximum(0, P)) - Q)
    return y

