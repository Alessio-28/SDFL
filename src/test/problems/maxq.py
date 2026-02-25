# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "maxq"
n: int = 20
starting_point: npt.NDArray[np.float64] = 1 + np.arange(n, dtype=np.float64)
starting_point[n//2:] *= -1

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.max(x*x)
