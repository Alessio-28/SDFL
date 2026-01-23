############################################################################################
#
#description  (problems for mixed-integer problems with AT LEAST
#              2 discrete variables and 2 continuous variables)
#
#This python module exports the following objects:
#- problem : a python class
#- prob_collection : A dictionary of entries "probname" => problem object
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
import types

from ..sdfl.core.typing import Point, ObjectiveFunction

class Problem:
    name: str
    starting_point: Point
    n: int
    feval: ObjectiveFunction

    def __init__(self: Problem, name: str, starting_point: Point, n: int, feval: ObjectiveFunction) -> None:
        if len(starting_point.shape) != 1:
            raise ValueError("starting_point must be a 1-dimensional array.")
        if starting_point.size != n:
            raise ValueError("starting_point must have size n.")
        self.name = name
        self.starting_point = starting_point
        self.n = n
        self.feval = feval

_TEST_FUNCTION_MODULE: str = "src.test.test_functions"
_TEST_FUNCTION_DIR: str = _TEST_FUNCTION_MODULE.replace(".", "/")
def _set_problem(problem_module: types.ModuleType) -> Problem:
    return Problem(
        name=problem_module.name,
        starting_point=problem_module.starting_point,
        n=problem_module.n,
        feval=problem_module.feval
    )

_problems: dict[str, Problem] | None = None
def get_problems() -> dict[str, Problem]:
    global _problems
    if _problems != None:
        return _problems
    _problems = {}
    for file in os.listdir(pathlib.PurePath(_TEST_FUNCTION_DIR)):
        filename = file.split(".")
        if len(filename) == 2 and filename[1] == "py":
            module = importlib.import_module(f"{_TEST_FUNCTION_MODULE}.{filename[0]}")
            _problems[module.name] = _set_problem(module)
    return _problems

def get_problem_names() -> list[str]:
   return list(get_problems().keys())

def print_problem_names() -> None:
    problems: list[str] = get_problem_names()
    print(*problems, sep = "\n")
