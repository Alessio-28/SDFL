# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "filter"
starting_point: npt.NDArray[np.float64] = np.array(
    [0, 1, 0, -0.15, 0, -0.68, 0, -0.72, 0.37],
    dtype=np.float64,
)
n: int = starting_point.size


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    y = x[:-1]
    t_cos = (1 + y[_odd, np.newaxis]) * _COS_ETA + y[_even, np.newaxis]
    t_sin = (1 - y[_odd, np.newaxis]) * _SIN_ETA

    np.square(t_cos, out=t_cos)
    np.square(t_sin, out=t_sin)

    t = t_cos + t_sin

    A = (t[0] * t[2]) / (t[1] * t[3])

    np.sqrt(A, out=A)
    return np.max(np.abs(x[-1] * A - _U))


def _compute_T(size: int, first_slice: int) -> npt.NDArray[np.float64]:
    a: int = first_slice
    b: int = size // 2 - a

    s1 = 0.01 * np.arange(a, dtype=np.float64)
    s2 = 0.07 + 0.03 * np.arange(b, dtype=np.float64)
    s3 = np.array([0.5], dtype=np.float64)
    s4 = 0.54 + 0.03 * np.arange(b, dtype=np.float64)
    s5 = 0.95 + 0.01 * np.arange(a, dtype=np.float64)

    return np.concatenate([s1, s2, s3, s4, s5], dtype=np.float64)


_T = _compute_T(size=41, first_slice=6)
_ETA = np.pi * _T

_COS_ETA = np.cos(_ETA)
_SIN_ETA = np.sin(_ETA)

_U = np.abs(1 - 2 * _T)

_even = np.arange(n - 1, step=2)
_odd = _even + 1

del _T
del _ETA
del _compute_T
