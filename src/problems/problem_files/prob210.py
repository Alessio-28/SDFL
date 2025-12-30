#**************************************************
# prob. n.10 described in paper:
# J. Müller
# MISO: Mixed-Integer Surrogate Optimization Framework
# Optimization and Engineering, 17(1):177-203 (2016)
# 
# N.B. variables u_i are x(i), i = 1..nu
#      variables x_i are x(r+i), i = 1..nx
#**************************************************
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "MISO prob. 10"
n         : int = 60
nint      : int = 30
ncont     : int = n - nint
lb        : npt.NDArray[np.float64] = -15 * np.ones(n, dtype = np.float64)
ub        : npt.NDArray[np.float64] =  30 * np.ones(n, dtype = np.float64)
lbmix     : npt.NDArray[np.float64] = -15 * np.ones(n, dtype = np.float64)
ubmix     : npt.NDArray[np.float64] =  30 * np.ones(n, dtype = np.float64)
startp    : npt.NDArray[np.float64] =   7 * np.ones(n, dtype = np.float64) 
x_initial : npt.NDArray[np.float64] =   7 * np.ones(n, dtype = np.float64) 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)

def feval(x : npt.NDArray[np.float64]) -> np.float64:  
    f = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / 15)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / 15)
    return f

