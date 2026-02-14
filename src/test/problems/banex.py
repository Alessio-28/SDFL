# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "banex"
n: int = 2
starting_point: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return (x[0] - 1)**2 + 100 * (x[0]**2 - x[1])**2
