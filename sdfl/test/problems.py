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

from ..sdfl.core.typing import Point, ObjectiveFunction

class Problem:
    name: str
    starting_point: Point
    n: int
    feval: ObjectiveFunction

    def __init__(self: Problem, name: str, starting_point: Point, n: int, feval: ObjectiveFunction) -> None:
        self.name   = name
        self.starting_point = starting_point
        self.n      = n
        self.feval  = feval

_TEST_FUNCTION_MODULE: str = "sdfl.test.test_functions"
def set_problems(problems: list[str]) -> dict[str, Problem]:
    problem_collection: dict[str, Problem] = {}
    for problem in problems:
        module = importlib.import_module(f"{_TEST_FUNCTION_MODULE}.{problem}")
        problem_collection[problem] = Problem(
            name = module.name,
            starting_point = module.starting_point,
            n = module.n,
            feval = module.feval
        )
    return problem_collection

_problems: dict[str, str] | None = None
def get_problems() -> dict[str, str]:
    global _problems
    if _problems is not None:
        return _problems
    _problems = {}
    for file in os.listdir(pathlib.PurePath(_TEST_FUNCTION_MODULE.replace(".", "/", count = 1))):
        filename = file.split(".")
        if len(filename) == 2 and filename[1] == "py":
            module = importlib.import_module(f"{_TEST_FUNCTION_MODULE}.{filename[0]}")
            _problems[module.name] = filename[0]
    return _problems

def get_problem_names() -> list[str]:
    problems: dict[str, str] = get_problems()
    return list(problems.keys())

def get_problem_files() -> list[str]:
    problems: dict[str, str] = get_problems()
    return list(problems.values())

def print_problem_names() -> None:
    problems: list[str] = get_problem_names()
    print(*problems, sep = "\n")
