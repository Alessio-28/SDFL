from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np

from ..sdfl.core.typing import ObjectiveFunction, Point


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
            raise TypeError("Starting_point must be a ndarray.")
        if len(starting_point.shape) != 1:
            raise ValueError("starting_point must be a 1-dimensional array.")
        if starting_point.size != n:
            raise ValueError(f"starting_point must of size {n}.")

        self.name = name
        self.starting_point = starting_point
        self.n = n
        self.feval = feval


_TEST_FUNCTION_DIR: Path = Path("src/test/problems")
_TEST_FUNCTION_MODULE: str = ".".join(_TEST_FUNCTION_DIR.parts)
_problems: dict[str, Problem] | None = None


def import_problem(file: str) -> Problem:
    try:
        module: ModuleType = import_module(file)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module {file} not found.")

    if (
        type(getattr(module, "name", None)) != str
        or type(getattr(module, "starting_point", None)) != np.ndarray
        or type(getattr(module, "n", None)) != int
        or not callable(getattr(module, "feval", None))
    ):
        raise AttributeError(f"Module {file} does not have the required variables or functions.")

    return Problem(
        name=module.name,
        starting_point=module.starting_point,
        n=module.n,
        feval=module.feval,
    )


def _get_problems() -> dict[str, Problem]:
    global _problems
    if _problems is not None:
        return _problems

    if not _TEST_FUNCTION_DIR.is_dir():
        raise FileNotFoundError(f"{_TEST_FUNCTION_DIR} not found.")

    _problems = {
        problem.name: problem
        for f in _TEST_FUNCTION_DIR.glob("*.py")
        if f.is_file()
        and f.stem != "__init__"
        and (problem := import_problem(f"{_TEST_FUNCTION_MODULE}.{f.stem}"))
    }

    return _problems


def get_problem(problem_name: str) -> Problem:
    return deepcopy(_get_problems()[problem_name])


def get_problem_names() -> list[str]:
    return list(_get_problems().keys())


def print_problem_names() -> None:
    problems: list[str] = get_problem_names()
    print(*problems, sep="\n")
