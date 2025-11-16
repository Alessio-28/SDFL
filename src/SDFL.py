from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class Parameters:
    theta   : np.float64
    gamma   : np.float64
    c       : np.float64
    eta     : np.float64
    epsilon : np.float64

def bound(param : Parameters, step : np.float64) -> np.float64:
    return -param.gamma * param.c * param.epsilon * (step ** 2)

# Placeholder
def F(x : npt.NDArray[np.float64]) -> np.float64:
    return np.sum(x**2)

def SDFL(x_0 : npt.NDArray[np.float64], step_0 : npt.NDArray[np.float64], param : Parameters) -> npt.NDArray[np.float64]:
    nF : int = 0
    n  : int = x_0.size
    success : bool = True
    base : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) # Usato per i vettori della base canonica

    step      : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) # \alpha_k^i
    init_step : npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) # \bar{\alpha}_k^i
    next_step : npt.NDArray[np.float64]                                   # \tilde{\alpha}_k^i

    x : npt.NDArray[np.float64] = x_0.copy()
    y : npt.NDArray[np.float64]
    while nF < n: # Placeholder
        y = x.copy()
        next_step = step_0.copy()
        max_next_step : np.float64 = param.eta * np.max(next_step)

        for i in range(n):
            success = True
            base[i] = 1
            init_step[i] = np.max((next_step[i], max_next_step))

            temp : np.float64 = bound(param, init_step[i])
            nF += 2
            F_y : np.float64 = F(y)
            if F(y + init_step * base) - F_y > temp:
                nF += 1
                if F(y - init_step * base) - F_y > temp:
                    step[i] = 0
                    success = False
                else:
                    base[i] = -1

            if success:
                a : np.float64 = init_step[i]
                b : np.float64 = init_step[i] * 2
                nF += 1
                while F(y + b * base) - F(y + a * base) <= bound(param, b - a):
                    a = b
                    b = a * 2
                    nF += 1
                step[i] = a
                y += a * base
            base[i] = 0

        if np.array_equal(y, x):
            next_step = param.theta * init_step
        else:
            x = y.copy()
            next_step = np.maximum(step, init_step)

    return x
