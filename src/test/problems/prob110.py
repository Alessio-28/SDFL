# **************************************************
# prob. n.10 described in paper:
# J. Müller, C.A. Shoemaker, R. Piché
# SO-I: a surrogate model algorithm for expensive nonlinear
# integer programming problems including global optimization applications
# Journal of Global Optimization, 59(4):865-889 (2014)
# **************************************************

import numpy as np
import numpy.typing as npt

name: str = "SO-I prob. 10"
n: int = 30
starting_point: npt.NDArray[np.float64] = np.ones(n, dtype=np.float64)


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.sum(x**2 - np.cos(2 * np.pi * x))
