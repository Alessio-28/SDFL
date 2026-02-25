# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "hs78"
starting_point: npt.NDArray[np.float64] = np.array([-2, 1.5, 2, -1, -1], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = np.empty(3, dtype=np.float64)
    f[0] = x @ x - 10
    f[1] = x[1]*x[2] - 5*x[3]*x[4]
    f[2] = x[0]**3 + x[1]**3 + 1
    np.abs(f, out=f)

    return np.prod(x) + 10*np.sum(f) # pyright: ignore[reportReturnType]
