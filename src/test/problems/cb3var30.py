# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

n: int = 30
name: str = f"cb3({n})"
starting_point: npt.NDArray[np.float64] = 2 * np.ones(n, dtype=np.float64)

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    y = x[:-1]
    z = x[1:]

    f = y**4 + z**2
    g = (2 - y)**2 + (2 - z)**2
    np.maximum(f, g, out=g)

    f = 2 * np.exp(z - y)
    np.maximum(g, f, out=f)

    return np.sum(f)
