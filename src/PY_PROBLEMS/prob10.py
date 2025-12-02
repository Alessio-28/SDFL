#**************************************************
# prob. n.10 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-MI: A surrogate model algorithm for computationally 
# expensive nonlinear mixed-integer black-box global optimization problems
# Computers & Operations Research, 40(5):1383-1400 (2013)
# 
# N.B. variables u_i are x(i), i = 1..nu
#      variables x_i are x(r+i), i = 1..nx
#**************************************************
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "SOMI prob.10"
#devono essere continue le prime 3
n         : int = 5
nint      : int = 2
ncont     : int = n - nint
lb        : npt.NDArray[np.float64] = -100 * np.ones(n, dtype = np.float64)
ub        : npt.NDArray[np.float64] =  100 * np.ones(n, dtype = np.float64)
lbmix     : npt.NDArray[np.float64] = -100 * np.ones(n, dtype = np.float64)
ubmix     : npt.NDArray[np.float64] =  100 * np.ones(n, dtype = np.float64)
startp    : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) 
x_initial : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:  #*x[0] *x[1] *x[2] x[3] x[4]
    f = x[3] * np.sin(x[3]) + 1.7 * x[4] * np.sin(x[3]) - 1.5 * x[0] - 0.1 * x[1] * np.cos(x[1] + x[2] - x[3]) + 0.2 * x[2] ** 2 - x[4] - 1
    return f

