#**************************************************
# prob. n.2 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-I: a surrogate model algorithm for expensive nonlinear
# integer programming problems including global optimization applications
# Journal of Global Optimization, 59(4):865-889 (2014)
#**************************************************
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "SO-I prob. 2"
n: int = 5
starting_point: npt.NDArray[np.float64] = 50 * np.ones(n, dtype=np.float64) 

def feval(x: npt.NDArray[np.float64]) -> np.float64:
    sin_x3 = np.sin(x[3])
    cos_term = np.cos(x[1]+x[2]-x[3])

    return (
        - 1.5*x[0]
        - 0.1*x[1]*cos_term
        + 0.2*x[2]**2
        + x[3]*sin_x3
        + 1.7*x[4]*sin_x3
        - 1
    )
