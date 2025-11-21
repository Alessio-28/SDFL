import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Callable
import SDFL

import test_functions as tf

import json

def test(name : str, f : Callable[[NDArray[float64]], float64], minimum : list[float], x : list[float], step : list[float], param : Parameters) -> None:
    m      = np.array( minimum, dtype = float64)
    x_0    = np.array( x, dtype = float64)
    step_0 = np.array( step, dtype = float64) # Passo iniziale unitario
    x_m = SDFL.SDFL(f, x_0, step_0, param)

    print(f"Funzione:        {name}")
    print(f"Punto di minimo: {m}")
    print(f"Punto iniziale:  {x_0}")
    print(f"Minimo trovato:  {x_m}")
    print()

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

    param : SDFL.Parameters = SDFL.Parameters(
        theta   = float64(0.5),
        gamma   = float64(2.5),
        c       = float64(1),
        eta     = float64(1),
        epsilon = float64(1) 
    )

    x_0 : list[float] = [3, -1, 2]
    test("Sphere", tf.sphere, [0]*len(x_0), x_0, [1]*len(x_0), param)

    x_0 = [-9, -4]
    test("Rosenbrock", tf.rosenbrock, [1]*len(x_0), x_0, [1]*len(x_0), param)

if __name__ == "__main__":
    main()
