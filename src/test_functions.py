import numpy as np
from numpy import float64
from numpy.typing import NDArray

def sphere(x : NDArray[float64]) -> float64:
    return np.sum(x**2)

def rosenbrock(x : NDArray[float64]) -> float64:
    n : int = x.size
    A : float64 = float64(1)
    B : float64 = float64(100)

    res : float64 = sphere(A - x)

    aux : float64 = float64(0)
    for i in range(n-1):
        aux += (x[i+1] + x[i]**2)**2

    return res + B * aux

def rastrigin(x : NDArray[float64]) -> float64:
    n : int = x.size
    A : float64 = float64(1)

    res : float64 = A * n + sphere(x)

    aux : float64 = float64(0)
    for i in range(n):
        aux += np.cos(2 * np.pi * x[i])

    return res - A * aux

def ackley(x : NDArray[float64]) -> float64:
    n : int = x.size
    A : float64 = float64(20)
    B : float64 = float64(0.2)
    C : float64 = float64(2 * np.pi)

    res : float64 = -A * np.exp(-B * np.sqrt(sphere(x) / n)) + A + np.e
    
    aux : float64 = float64(0)
    for i in range(n):
        aux += np.cos(C * x[i])

    return res - np.exp(aux / n)

def himmelblau(x : NDArray[float64]) -> float64:
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2)**2
