# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "kowalik-osborne"
starting_point: npt.NDArray[np.float64] = np.array([0.25, 0.39, 0.415, 0.39], dtype=np.float64)
n: int = starting_point.size

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f1 = (_U2 + x[1]*_U) * x[0]
    f2 =  _U2 + x[2]*_U  + x[3]
    return np.max(np.abs(f1 / f2 - _Z))

_Z = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246], dtype=np.float64)
_U = np.array([     4,      2,      1,    0.5,   0.25, 0.1670, 0.1250,    0.1, 0.0833, 0.0714, 0.0625], dtype=np.float64)
_U2 = _U**2
