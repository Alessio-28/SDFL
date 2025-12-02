# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name      : str = "transformer"
startp    : npt.NDArray[np.float64] = np.array([0.8, 1.5, 1.2, 3, 0.8, 6], dtype = np.float64)
lb        : npt.NDArray[np.float64] = startp - 10
ub        : npt.NDArray[np.float64] = startp + 10
n         : int = len(lb)
nint      : int = 3
ncont     : int = n - nint
lbmix     : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64);      lbmix[:ncont]     = lb[:ncont]
ubmix     : npt.NDArray[np.float64] = 100 * np.ones(n, dtype = np.float64); ubmix[:ncont]     = ub[:ncont]
x_initial : npt.NDArray[np.float64] =  50 * np.ones(n, dtype = np.float64); x_initial[:ncont] = (ub[:ncont] + lb[:ncont]) / 2 
xmix      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64)


def feval(x : npt.NDArray[np.float64]) -> np.float64:
    x = x.reshape(-1, 1)
    t = np.array([0.5, 0.6, 0.7, 0.77, 0.9, 1, 1.1, 1.23, 1.3, 1.4, 1.5], dtype = np.float64).reshape(-1, 1)
    f = np.zeros(11, dtype = np.float64)
    v = np.zeros(4, dtype = np.complex128)
    w = np.zeros(4, dtype = np.complex128)
    for i in range(0, 11):
        theta = np.pi * t[i] / 2
        v[3] = np.array([ 1], dtype = np.complex128)
        w[3] = np.array([10], dtype = np.complex128)
        for k in range(2, -1, -1):
            a = np.cos(theta * x[2 * k-1])             
            b = np.sin(theta * x[2 * k-1]) / x[2 * k]            
            c = np.sin(theta * x[2 * k-1]) * x[2 * k]            
            v[k] = np.array([a * v[k+1].real - b * w[k+1].imag + (a * v[k+1].imag + b * w[k+1].real) * 1j], dtype = np.complex128)
            w[k] = np.array([a * w[k+1].real - c * v[k+1].imag + (a * w[k+1].imag + b * v[k+1].real) * 1j], dtype = np.complex128)
        f[i] = np.abs(1 - 2 * v[0] / (w[0] + v[0]))
    y = np.max(f)
    return y

