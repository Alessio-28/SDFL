# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "shor"
starting_point: npt.NDArray[np.float64] = np.array([0, 0, 0, 0, 1], dtype=np.float64)
n: int = starting_point.size


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = x - _A
    np.square(f, out=f)
    return np.max(_B * np.sum(f, axis=1))


_A = np.array(
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
        [1, 1, 2, 0, 0],
    ],
    dtype=np.float64,
)
_B = np.array([1, 5, 10, 2, 4, 3, 1.7, 2.5, 6, 3.5], dtype=np.float64)
