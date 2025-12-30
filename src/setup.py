import numpy as np
from time import time
import logging
import os

from . import sdfl
from .problems import problems

LIMIT_EVAL : int = 1_000_000
LIMIT_STEP : np.float64 = np.float64(1e-8)

LOGGING : bool = False
LOG_PATH : str = "./log"
LOG_FILE : str = "sdfl.log"
MODE : str = "a"

main_fh : logging.FileHandler
main_log : logging.Logger


def setup_logging() -> None:
    global main_fh
    global main_log

    if not (os.path.exists(LOG_PATH) and os.path.isdir(LOG_PATH)):
        os.mkdir(LOG_PATH)

    main_fh = logging.FileHandler(filename = f"{LOG_PATH}/{LOG_FILE}", mode = MODE)
    main_fh.setLevel(logging.DEBUG)

    main_log = logging.getLogger(name = __name__)
    main_log.setLevel(logging.DEBUG)
    main_log.addHandler(main_fh)

    sdfl.LOGGING = True
    sdfl.LOG_PATH = LOG_PATH
    sdfl.LOG_FILE = LOG_FILE
    sdfl.MODE = MODE

    sdfl._setup_logging()


# def run(functions : list[str], starting_point : sdfl.Point | None = None, starting_step : npt.NDArray[np.float64] | None = None, param : sdfl.Parameters | None = None, limit_eval : int = LIMIT_EVAL, limit_step : np.float64 = LIMIT_STEP) -> None:
def run(functions : list[str], param : sdfl.Parameters, limit_eval : int = LIMIT_EVAL, limit_step : np.float64 = LIMIT_STEP) -> None:
    if LOGGING:
        setup_logging()

    prob_collection : dict[str, problems.Problem] = problems.set_problems(functions)

    for prob in functions:
        test(
            prob_collection[prob].name,
            prob_collection[prob].feval,
            prob_collection[prob].n,
            prob_collection[prob].startp,
            param,
            limit_eval,
            limit_step
        )

def test(name : str, obj_func : sdfl.ObjectiveFunction, dim : int, starting_point : sdfl.Point, param : sdfl.Parameters, limit_eval : int, limit_step : np.float64) -> None:
    starting_step = np.array([1] * dim, dtype = np.float64)

    if LOGGING:
        main_log.debug(f"Algorithm: {name}")

    start = time()
    minimum = sdfl.SDFL(obj_func, starting_point, starting_step, param, limit_eval, limit_step)
    end = time()

    if LOGGING:
        main_log.debug(f"Algorithm: {name}")
        main_log.debug(f"Time: {end - start:0.3f}")
        main_log.debug(f"Start: {starting_point}")
        main_log.debug(f"Min: {minimum}")

    print(f"Funzione:       {name}")
    print(f"Time:           {end - start:0.3f}")
    print(f"Punto iniziale: {starting_point}")
    print(f"Minimo trovato: {minimum}")
    print()
