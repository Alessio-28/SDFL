import numpy as np
from numpy import float64
from numpy.typing import NDArray

import sdfl
from PY_PROBLEMS import Problems

import json

def test(name : str, f : sdfl.ObjectiveFunction, dim : int, x_0 : NDArray[float64], param : sdfl.Parameters) -> None:
    step_0 = np.array([1]*dim, dtype = float64) # Passo iniziale unitario
    x_m = sdfl.SDFL(f, x_0, step_0, param)

    print(f"Funzione:       {name}")
    print(f"Punto iniziale: {x_0}")
    print(f"Minimo trovato: {x_m}\n")

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

    param : sdfl.Parameters = sdfl.Parameters(
        theta   = float64(0.5),
        gamma   = float64(2.5),
        c       = float64(1),
        eta     = float64(1),
        epsilon = float64(1) 
    )

    Problems.list_prob_names = [
        # "banex",
        # "cb2",
        # "cb3var20",
        # "cb3var30",
        # "cb3var40",
        # "cb3var50",
        # "colville1",
        # "crescent",
        # "davidon2",
        # "elattar",
        # "evd61",
        # "gill",
        # "goffin",
        # "hs78",
        # "kowalik",
        # "l1hilb",
        # "l1hilb20",
        # "l1hilb30",
        # "l1hilb40",
        # "lukexp",
        # "lukfilter",
        # "lukgamma",
        # "maxl",
        # "maxq",
        # "maxq30",
        # "maxq40",
        # "maxq50",
        # "maxquad",
        # "mxhilb",
        # "oet5",
        # "oet6",
        # "osborne2",
        # "pbc1",
        # "polak2",
        # "polak3",
        # "polak6",
        # "prob10",
        # "prob102",
        # "prob107",
        # "prob109",
        # "prob110",
        # "prob113",
        # "prob115",
        # "prob116",
        # "prob206",
        # "prob208",
        # "prob210",
        "rosen",
        # "shelldual",
        # "shor",
        # "steiner2",
        # "tr48",
        # "transformer",
        # "watson",
        # "wong1",
        # "wong2",
        # "wong3"
    ]

    Problems.prob_collection = {}
    Problems.set_problems()

    for prob in Problems.list_prob_names:
        test(
            Problems.prob_collection[prob].name,
            Problems.prob_collection[prob].feval,
            Problems.prob_collection[prob].n,
            Problems.prob_collection[prob].startp,
            param
        )

if __name__ == "__main__":
    main()
