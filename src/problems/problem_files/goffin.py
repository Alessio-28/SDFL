# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name   : str = "goffin"
startp : npt.NDArray[np.float64] = np.arange(1, 51, dtype = np.float64) - 25.5
n      : int = startp.size

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    f : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)
    f = 50 * x - np.sum(x)

    return np.max(f);
