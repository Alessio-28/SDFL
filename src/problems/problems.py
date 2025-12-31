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
from typing import Callable
from numpy import float64
from numpy.typing import NDArray

from os import listdir
from importlib import import_module
from pathlib import Path

class Problem:
    name   : str
    startp : NDArray[float64]
    n      : int
    feval  : Callable[[NDArray[float64]], float64]

    def __init__(self : Problem, name : str, startp : NDArray[float64], n : int, feval : Callable[[NDArray[float64]], float64]) -> None:
        self.name   = name
        self.startp = startp
        self.n      = n
        self.feval  = feval

def set_problems(problems : list[str]) -> dict[str, Problem]:
    prob_collection : dict[str, Problem] = {}
    for pname in problems:
        mname = import_module(f"src.problems.problem_files.{pname}")
        prob_collection[pname] = Problem(
            name   = mname.name,
            startp = mname.startp,
            n      = mname.n,
            feval  = mname.feval
        )
    return prob_collection

problems : dict[str, str] | None = None
def get_problems() -> dict[str, str]:
    global problems
    if problems is not None:
        return problems
    problems = {}
    for file in listdir(Path("src//problems//problem_files")):
        filename = file.split(".")
        if len(filename) == 2 and filename[1] == "py":
            module = import_module(f"src.problems.problem_files.{filename[0]}")
            problems[module.name] = filename[0]
    return problems

def get_problem_names() -> list[str]:
    probs : dict[str, str] = get_problems()
    return list(probs.keys())

def get_problem_files() -> list[str]:
    probs : dict[str, str] = get_problems()
    return list(probs.values())

def print_problem_names() -> None:
    probs : list[str] = get_problem_names()
    print(*probs, sep = "\n")
