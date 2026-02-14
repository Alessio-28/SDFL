# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "goffin"
n: int = 50
starting_point: npt.NDArray[np.float64] = np.arange(n, dtype=np.float64) + 1 - 25.5

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.max(50*x - np.sum(x))
