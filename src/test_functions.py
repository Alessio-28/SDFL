import numpy as np
from numpy import float64
from numpy.typing import NDArray

def sphere(x : NDArray[float64]) -> float64:
    return np.sum(x**2)

def rosenbrock(x : NDArray[float64]) -> float64:
    n = x.size
    A = 1
    B = 100

    res : float64 = sphere(A - x[:n-1])

    aux : float64 = float64(0)
    for i in range(n-1):
        aux += (x[i+1] - x[i]**2)**2

    return res + B * aux
