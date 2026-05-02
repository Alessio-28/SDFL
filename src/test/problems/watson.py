# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "watson"
n: int = 20
starting_point: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = np.empty(_m, dtype=np.float64)

    f[:-2] = ((_J * x) @ _A) - (x @ _B) ** 2
    f[-2] = x[0]
    f[-1] = x[1] - x[0] ** 2 - 1

    np.abs(f, out=f)
    return np.max(f)


_m: int = 31
_l: int = _m - 2
_I = np.linspace(1 / _l, 1, _l, dtype=np.float64)
_J = np.arange(n, dtype=np.float64)
_A = _I ** (_J[:, np.newaxis] - 1)
_B = _I ** (_J[:, np.newaxis])

del _l
del _I
