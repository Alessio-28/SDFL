# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "pbc1"
startp    : npt.NDArray[np.float64] = np.array([0, -1, 10, 1, 10], dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 2
ncont     : int = n-nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64); lbmix[:ncont] = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100*np.ones(n, dtype = np.float64); ubmix[:ncont] = ub[:ncont]
x_initial : npt.NDArray[np.float64] = 50*np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont])/2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:
    t = -1 + 2*(np.arange(1,31)-1)/29
    f = (x[0] + x[1]*t + x[2]*t**2)/(1 + x[3]*t + x[4]*t**2) - (np.sqrt((8*t - 1)**2 + 1)*np.arctan(8*t))/(8*t)

    y = np.max(np.abs(f))
    return y
