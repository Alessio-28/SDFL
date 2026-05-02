############################################################################################
#
# description  (problems for mixed-integer problems with AT LEAST
#              2 discrete variables and 2 continuous variables)
#
# This python module exports the following objects:
# - problem : a python class
# - prob_collection : A dictionary of entries "probname" => problem object
#
# a problem object is a structured type that has the following attributes:
# name   : string - name of the problem
# startp : numpy array - the starting point for the continuous problem
# n      : int - the total number of variables (>= 4)
# feval  : function handle - function to compute the objective function value
#
############################################################################################
import os
import importlib
import pathlib
import numpy as np

from sdfl.core.typing import Point, ObjectiveFunction


class Problem:
    name: str
    starting_point: Point
    n: int
    feval: ObjectiveFunction

    def __init__(
        self: Problem,
        name: str,
        starting_point: Point,
        n: int,
        feval: ObjectiveFunction,
    ) -> None:
        if not isinstance(starting_point, np.ndarray):
            raise ValueError("Starting_point must be a ndarray.")
        if len(starting_point.shape) != 1:
            raise ValueError("starting_point must be a 1-dimensional array.")
        if starting_point.size != n:
            raise ValueError(f"starting_point must of size {n}.")

        self.name = name
        self.starting_point = starting_point
        self.n = n
        self.feval = feval


def import_problem(problem_module: str) -> Problem:
    try:
        module = importlib.import_module(problem_module)
        problem = Problem(
            name=module.name,
            starting_point=module.starting_point,
            n=module.n,
            feval=module.feval,
        )
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Problem file not found.")
    except ImportError:
        raise ImportError("Problem file does not have the required variables/function.")

    return problem


_problems: dict[str, Problem] | None = None


def _get_problems(problem_module: str) -> dict[str, Problem]:
    global _problems
    if _problems is not None:
        return _problems
    _problems = {}
    for file in os.listdir(pathlib.PurePath(problem_module.replace(".", "/"))):
        filename = file.split(".")
        if len(filename) == 2 and filename[1] == "py":
            problem = import_problem(f"{problem_module}.{filename[0]}")
            _problems[problem.name] = problem
    return _problems


def get_default_problems() -> dict[str, Problem]:
    _TEST_FUNCTION_MODULE: str = "src.test.problems"
    return _get_problems(_TEST_FUNCTION_MODULE)


def get_problem_names() -> list[str]:
    return list(get_default_problems().keys())


def print_problem_names() -> None:
    problems: list[str] = get_problem_names()
    print(*problems, sep="\n")
