# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "crescent"
startin_point: npt.NDArray[np.float64] = np.array([-1.5, 2], dtype = np.float64)
n: int = startin_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f1: np.float64 =   x[0]**2 + (x[1]-1)**2 + x[1] - 1
    f2: np.float64 = - x[0]**2 - (x[1]-1)**2 + x[1] + 1

    return np.maximum(f1, f2)
