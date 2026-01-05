# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "filter"
startin_point: npt.NDArray[np.float64] = np.array([0, 1, 0, -0.15, 0, -0.68, 0, -0.72, 0.37], dtype = np.float64)
n: int = startin_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    t: npt.NDArray[np.float64] = np.zeros(41, dtype = np.float64)
    t[0:6]   = 0.01 * (np.arange(1, 7) - 1)
    t[6:20]  = 0.07 + 0.03 * (np.arange(7, 21) - 7);
    t[20]    = 0.5
    t[21:35] = 0.54 + 0.03 * (np.arange(22, 36) - 22);
    t[35:41] = 0.95 + 0.01 * (np.arange(36, 42) - 36);

    z: npt.NDArray[np.float64] = np.abs(1 - 2 * t) 
    eta: npt.NDArray[np.float64] = np.pi * t
    A: npt.NDArray[np.float64] = ((x[0] + (1 + x[1]) * np.cos(eta)) ** 2 + ((1 - x[1]) * np.sin(eta)) ** 2) / ((x[2] + (1 + x[3]) * np.cos(eta)) ** 2 + ((1 - x[3]) * np.sin(eta)) ** 2)
    B: npt.NDArray[np.float64] = ((x[4] + (1 + x[5]) * np.cos(eta)) ** 2 + ((1 - x[5]) * np.sin(eta)) ** 2) / ((x[6] + (1 + x[7]) * np.cos(eta)) ** 2 + ((1 - x[7]) * np.sin(eta)) ** 2)
    f: npt.NDArray[np.float64] = x[8] * np.sqrt(A) * np.sqrt(B) - z

    return np.max(np.abs(f))
