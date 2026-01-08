# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "maxl"
starting_point: npt.NDArray[np.float64] = np.concatenate([np.array(np.arange(1, 11, dtype = np.float64)), np.array(np.arange(-11, -21, -1, dtype = np.float64))])
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f: npt.NDArray[np.float64] = np.zeros(n * 2, dtype = np.float64)
    f[:n]   =  x[:]
    f[n:n*2] = -x[:]
    return np.max(f)
