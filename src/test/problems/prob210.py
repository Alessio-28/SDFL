# **************************************************
# prob. n.10 described in paper:
# J. Müller
# MISO: Mixed-Integer Surrogate Optimization Framework
# Optimization and Engineering, 17(1):177-203 (2016)
# **************************************************
"""
Created on Fri Oct 16 17:38:11 2020

@author: giamp
"""

import numpy as np
import numpy.typing as npt

name: str = "MISO prob. 10"
n: int = 60
starting_point: npt.NDArray[np.float64] = 7 * np.ones(n, dtype=np.float64)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f1 = np.exp(-0.2 * np.sqrt((x @ x) / 15))
    f2 = np.exp(np.sum(np.cos(2 * np.pi * x)) / 15)
    return -20 * f1 - f2
