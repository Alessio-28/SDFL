# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name   : str = "exp"
startp : npt.NDArray[np.float64] = np.array([0.5, 0, 0, 0, 0], dtype = np.float64)
n      : int = startp.size

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    t = -1 + (np.arange(1,22)-1)/10     
    f = (x[0] + x[1]*t)/(1 + x[2]*t + x[3]*t**2 + x[4]*t**3) - np.exp(t)

    return np.max(f)
