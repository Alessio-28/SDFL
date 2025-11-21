import json
from typing import Callable

import test_functions as tf

import numpy as np
from numpy import float64
from numpy.typing import NDArray

class Parameters:
    theta   : float64
    gamma   : float64
    c       : float64
    eta     : float64
    epsilon : float64

    def __init__(self : Parameters, theta : float64, gamma : float64, c : float64, eta : float64, epsilon : float64, ) -> None:
       self.theta   = theta
       self.gamma   = gamma
       self.c       = c
       self.eta     = eta
       self.epsilon = epsilon

    def bound(self : Parameters, param : Parameters, step : float64) -> float64:
        return -param.gamma * param.c * param.epsilon * (step ** 2)

#########################################################################################


def SDFL(F : Callable[[NDArray[float64]], float64], x_0 : NDArray[float64], step_0 : NDArray[float64], param : Parameters) -> NDArray[float64]:
    LIMIT : float64 = float64(1e-6)
    nF : int = 0
    n  : int = x_0.size
    success : bool = True

    base      : NDArray[float64] = np.zeros(n, dtype = float64) # Usato per i vettori della base canonica
    step      : NDArray[float64] = np.zeros(n, dtype = float64) # \alpha_k^i
    init_step : NDArray[float64] = np.zeros(n, dtype = float64) # \bar{\alpha}_k^i
    next_step : NDArray[float64] = step_0.copy()                # \tilde{\alpha}_k^i

    max_next_step : float64 = np.max(next_step)
    x : NDArray[float64] = x_0.copy()
    y : NDArray[float64]
    while max_next_step > LIMIT:
        y = x.copy()
        max_next_step = param.eta * max_next_step

        for i in range(n):
            success = True
            base[i] = 1
            init_step[i] = np.max((next_step[i], max_next_step))

            temp : float64 = param.bound(param, init_step[i])
            nF += 2
            F_y : float64 = F(y)
            if F(y + init_step * base) - F_y > temp:
                nF += 1
                if F(y - init_step * base) - F_y > temp:
                    step[i] = 0
                    success = False
                else:
                    base[i] = -1

            if success:
                a : float64 = init_step[i]
                b : float64 = init_step[i] * 2
                nF += 1 # Dovrebbe essere nF += 2 ?
                while F(y + b * base) - F(y + a * base) <= param.bound(param, b - a):
                    a = b
                    b = a * 2
                    nF += 1 # Dovrebbe essere nF += 2 ?
                step[i] = a
                y += a * base
            base[i] = 0

        if np.array_equal(y, x):
            next_step = param.theta * init_step
        else:
            x = y.copy()
            next_step = np.maximum(step, init_step)

        max_next_step = np.max(next_step)

    return x

def main() -> None:
    ## I parametri possono essere anche impostati dal file parameters.json

    # with open("parameters.json") as p:
    #     data = json.load(p)
    #
    # param : Parameters = Parameters(
    #     theta   = data["theta"],
    #     gamma   = data["gamma"],
    #     c       = data["c"],
    #     eta     = data["eta"],
    #     epsilon = data["epsilon"]
    # )

    param : Parameters = Parameters(
        theta   = float64(0.5),
        gamma   = float64(2.5),
        c       = float64(1),
        eta     = float64(1),
        epsilon = float64(1) 
    )

    f = tf.sphere
    x_0    = np.array( [3, -1, 2], dtype = float64)
    step_0 = np.array( [1]*x_0.size, dtype = float64) # Passo iniziale unitario
    x = SDFL(f, x_0, step_0, param)
    print("Funzione: Sphere")
    print("Punto di minimo: 0")
    print(f"Punto iniziale: {x_0}")
    print(f"Minimo trovato: {x}")
    print()

    f = tf.rosenbrock
    x_0    = np.array( [-9, 4], dtype = float64)
    step_0 = np.array( [1]*x_0.size, dtype = float64)
    x = SDFL(f, x_0, step_0, param)
    print("Funzione: Rosenbrock")
    print("Punto di minimo: 1")
    print(f"Punto iniziale: {x_0}")
    print(f"Minimo trovato: {x}")
    print()

    f = tf.rastrigin
    x_0    = np.array( [5, -2.5, 2], dtype = float64)
    step_0 = np.array( [1]*x_0.size, dtype = float64)
    x = SDFL(f, x_0, step_0, param)
    print("Funzione: Rastrigin")
    print("Punto di minimo: 0")
    print(f"Punto iniziale: {x_0}")
    print(f"Minimo trovato: {x}")
    print()

    f = tf.ackley
    x_0    = np.array( [-4, 1], dtype = float64)
    step_0 = np.array( [1]*x_0.size, dtype = float64)
    x = SDFL(f, x_0, step_0, param)
    print("Funzione: Ackley")
    print("Punto di minimo: 0")
    print(f"Punto iniziale: {x_0}")
    print(f"Minimo trovato: {x}")
    print()

if __name__ == "__main__":
    main()
