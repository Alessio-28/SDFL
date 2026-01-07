# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

starting_point: npt.NDArray[np.float64] = np.concatenate([np.array(np.arange(1, 26, dtype = np.float64)), np.array(np.arange(-26, -51, -1, dtype = np.float64))])
n: int = starting_point.size
name: str = f"maxq({n})"

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.max(x ** 2)
