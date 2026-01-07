# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "colville 1"
starting_point: npt.NDArray[np.float64] = np.array([0, 0, 0, 0, 1], dtype = np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1, 1)
    A: npt.NDArray[np.float64] = np.array(
        [
            [ -16,  2,  0,  1,   0],
            [   0, -2,  0,  4,   2],
            [-3.5,  0,  2,  0,   0],
            [   0, -2,  0, -4,  -1],
            [   0, -9, -2,  1,-2.8],
            [   2,  0, -4,  0,   0],
            [  -1, -1, -1, -1,  -1],
            [  -1, -2, -3, -2,  -1],
            [   1,  2,  3,  4,   5],
            [   1,  1,  1,  1,   1]
        ],
        dtype = np.float64
    )
 
    b: npt.NDArray[np.float64] = np.array([ [-40], [-2], [-0.25], [-4], [-4], [-1], [-40], [-60], [5], [1] ], dtype = np.float64)

    C: npt.NDArray[np.float64] = np.array(
        [
            [  30, -20, -10,  32, -10],
            [ -20,  39,  -6, -31,  32],
            [ -10,  -6,  10,  -6, -10],
            [  32, -31,  -6,  39, -20],
            [ -10,  32, -10, -20,  30]
        ],
        dtype = np.float64
    )
 
    d: npt.NDArray[np.float64] = np.array([ [4], [8], [10], [6], [2] ], dtype = np.float64)

    ee: npt.NDArray[np.float64] = np.array([ [-15, -27, -36, -18, -12] ], dtype = np.float64)

    y: npt.NDArray[np.float64] = np.sum(d * x ** 3) + np.sum(np.sum(C * (x * x))) + np.matmul(ee, x) + 50 * np.maximum(0, np.max(b - np.matmul(A, x)))

    return y[0][0]
