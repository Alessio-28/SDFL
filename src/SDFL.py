import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Callable

class Parameters:
    theta   : float64
    gamma   : float64
    c       : float64
    eta     : float64
    epsilon : float64
    _bound_coeff_ : float64

    def __init__(self : Parameters, theta : float64, gamma : float64, c : float64, eta : float64, epsilon : float64, ) -> None:
       self.theta   = theta
       self.gamma   = gamma
       self.c       = c
       self.eta     = eta
       self.epsilon = epsilon
       self._bound_coeff_ = -gamma * c * epsilon

    def bound(self : Parameters, step : float64) -> float64:
        return self._bound_coeff_ * (step ** 2)

def SDFL(F : Callable[[NDArray[float64]], float64], x_0 : NDArray[float64], step_0 : NDArray[float64], param : Parameters) -> NDArray[float64]:
    LIMIT : float64 = float64(1e-6)
    nF : int = 0
    n  : int = x_0.size
    success : bool = True

    base           : NDArray[float64] = np.zeros(n, dtype = float64) # Usato per i vettori della base canonica | Soluzione probabilmente provvisoria
    accepted_step  : NDArray[float64] = np.zeros(n, dtype = float64) # \alpha_k^i
    init_step      : NDArray[float64] = np.zeros(n, dtype = float64) # \bar{\alpha}_k^i
    tentative_step : NDArray[float64] = step_0.copy()                # \tilde{\alpha}_k^i

    max_tentative_step : float64 = np.max(tentative_step)
    x : NDArray[float64] = x_0.copy()
    y : NDArray[float64] = x_0.copy()
    while max_tentative_step > LIMIT:
        for i in range(n):
            success = True
            base[i] = 1
            init_step[i] = np.max((tentative_step[i], param.eta * max_tentative_step))

            temp : float64 = param.bound(init_step[i])
            nF += 2
            F_y : float64 = F(y)
            if F(y + init_step * base) - F_y > temp:
                nF += 1
                if F(y - init_step * base) - F_y > temp:
                    accepted_step[i] = 0
                    success = False
                else:
                    base[i] = -1

            if success:
                a : float64 = init_step[i] # b = 2 * a per ogni ciclo while
                b : float64 = init_step[i] * 2
                nF += 1
                F_a : float64 = F(y + a * base)
                F_b : float64 = F(y + b * base)
                while F_b - F_a <= param.bound(b - a):
                    a = b
                    b = a * 2

                    nF += 1
                    F_a = F_b
                    F_b = F(y + b * base)
                accepted_step[i] = a
                y += a * base
            base[i] = 0

        if np.array_equal(y, x):
            tentative_step = param.theta * init_step
        else:
            x = y.copy()
            tentative_step = np.maximum(accepted_step, init_step)

        max_tentative_step = np.max(tentative_step)

    return x
