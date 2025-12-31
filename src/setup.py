import numpy as np
from time import time

import logging
from os import mkdir
from os.path import exists, isdir

from . import sdfl
from .problems import problems

LIMIT_EVAL : int = 1_000_000
LIMIT_STEP : np.float64 = np.float64(1e-8)

LOGGING : bool = False
LOG_DIR : str = "log"
LOG_FILE : str = "sdfl.log"
MODE : str = "a"
LEVEL = logging.INFO 

_main_log : logging.Logger

def setup_logging() -> None:
    global _main_log

    if not (exists(LOG_DIR) and isdir(LOG_DIR)):
        mkdir(LOG_DIR)

    main_fh : logging.FileHandler = logging.FileHandler(filename = f"./{LOG_DIR}/{LOG_FILE}", mode = MODE)
    main_fh.setLevel(LEVEL)

    _main_log = logging.getLogger(name = __name__)
    _main_log.setLevel(LEVEL)
    _main_log.addHandler(main_fh)
    
    sdfl.LOGGING = True
    sdfl.LOG_DIR = LOG_DIR
    sdfl.LOG_FILE = LOG_FILE
    sdfl.MODE = MODE
    sdfl.LEVEL = LEVEL

    logging._srcfile = None # pyright: ignore[reportPrivateUsage]
    logging.logProcesses = False
    logging.logThreads = False
    logging.logMultiprocessing = False

    # np.set_printoptions(precision = 4, suppress = True)
    sdfl._setup_logging() # pyright: ignore[reportPrivateUsage]


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
        _main_log.info(f"Algorithm: %s", name)

    start = time()
    minimum = sdfl.SDFL(obj_func, starting_point, starting_step, param, limit_eval, limit_step)
    end = time()

    if LOGGING:
        _main_log.info(f"Algorithm: %s\nTime: %0.3f\nStart: %s\nMin: %s", name, end - start, starting_point, minimum)

    print(f"Funzione:       {name}\nTime:           {end - start:0.3f}\nPunto iniziale: {starting_point}\nMinimo trovato: {minimum}\n")
