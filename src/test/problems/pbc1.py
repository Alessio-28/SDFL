# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "pbc1"
starting_point: npt.NDArray[np.float64] = np.array([0, -1, 10, 1, 10], dtype=np.float64)
n: int = starting_point.size


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f1 = x[0] + x[1] * _Y + x[2] * _Y2
    f2 = 1 + x[3] * _Y + x[4] * _Y2

    return np.max(np.abs((f1 / f2) - _Z))


_Y = np.linspace(-1, 1, 30, dtype=np.float64)
_Y2 = _Y**2
_Z = (np.sqrt((8 * _Y - 1) ** 2 + 1) * np.arctan(8 * _Y)) / (8 * _Y)
