# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "shor"
startp    : npt.NDArray[np.float64] = np.array([0, 0, 0, 0, 1], dtype = np.float64)
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
    x = x.reshape(-1, 1)
    A = np.array(
        [
            [0, 0, 0, 0, 0],
            [2, 1, 1, 1, 3],
            [1, 2, 1, 1, 2],
            [1, 4, 1, 2, 2],
            [3, 2, 1, 0, 1],
            [0, 2, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 2, 1],
            [0, 0, 2, 1, 0],
            [1, 1, 2, 0, 0]
        ],
        dtype = np.float64
    )
    b = np.squeeze(np.array(
        [
            [  1],
            [  5],
            [ 10],
            [  2],
            [  4],
            [  3],
            [1.7],
            [2.5],
            [  6],
            [3.5]
        ],
        dtype = np.float64
    ))
    X = np.tile(x.T, (10, 1))
    fx = b * np.sum((X - A) ** 2, 1)
    y= np.max(fx)
    return y

