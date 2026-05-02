# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "gill"
n: int = 10
starting_point: npt.NDArray[np.float64] = -0.1 * np.ones(n, dtype=np.float64)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    A = x[1:] @ _term
    B = np.sum(np.exp(_J * x * _logI), axis=1)

    y = x * x
    f = np.empty(3, dtype=np.float64)
    f[0] = np.sum((x - 1) ** 2) + np.sum((y - 0.25) ** 2) / 1000
    f[1] = np.sum((A - B**2 - 1) ** 2) + y[0] + (x[1] - y[0] - 1) ** 2
    f[2] = np.sum(100 * (x[1:] - y[:-1]) ** 2 + (1 - x[1:]) ** 2)

    return np.max(f)


_m: int = 29
_I = np.linspace(1 / _m, 1.0, _m, dtype=np.float64)
_J = np.arange(n, dtype=np.float64)

_logI = np.log(_I)[:, np.newaxis]
_term = _J[1:, np.newaxis] * _I ** _J[:-1, np.newaxis]

del _m
del _I
