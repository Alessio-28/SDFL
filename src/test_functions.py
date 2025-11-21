import numpy as np
from numpy import float64
from numpy.typing import NDArray

def sphere(x : NDArray[float64]) -> float64:
    return np.sum(x**2)

def rosenbrock(x : NDArray[float64]) -> float64:
    n = x.size
    A = 1
    B = 100

    res : float64 = sphere(A - x)

    aux : float64 = float64(0)
    for i in range(n-1):
        aux += (x[i+1] - x[i]**2)**2

    return res + B * aux

def rastrigin(x : NDArray[float64]) -> float64:
    n = x.size
    A = 1

    return A * n + sphere(x) - A * np.sum(np.cos(2 * np.pi * x))

def ackley(x : NDArray[float64]) -> float64:
    n = x.size
    A = 20
    B = 0.2
    C = 2 * np.pi

    return -A * np.exp(-B * np.sqrt(sphere(x) / n)) - np.exp(np.sum(np.cos(C * x)) / n) + A + np.e

def himmelblau(x : NDArray[float64]) -> float64:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
