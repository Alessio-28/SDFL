import json

import numpy as np
import numpy.typing as npt

from ..sdfl.core import parameters
from ..test import problem_manager
from . import constants, sdfl_data

DATA_JSON: str = "data.json"


def export_data(data_dict: constants.SDFLArgsDict) -> None:
    with open(DATA_JSON, "w") as p:
        json.dump(data_dict, p, indent=4, separators=(",", ": "))


def import_data() -> constants.SDFLArgsDict:
    try:
        with open(DATA_JSON, "r") as p:
            data_dict = json.load(p)
    except OSError:
        return create_default_data_json()

    _validate_data_json(data_dict)
    return data_dict


def create_default_data_json() -> constants.SDFLArgsDict:
    export_data(constants.DEFAULT_PROBLEM_DATA)
    return deepcopy(constants.DEFAULT_PROBLEM_DATA)


def dict_to_SDFLData(
    p: problem_manager.Problem,
    data_dict: constants.SDFLArgsDict,
    starting_step: npt.NDArray[np.float64] | None = None,
) -> sdfl_data.SDFLData:
    data = sdfl_data.SDFLData(
        problem=p,
        max_eval=data_dict["max_eval"],
        min_step=data_dict["min_step"],
        params=parameters.Parameters(
            theta=data_dict["params"]["theta"],
            gamma=data_dict["params"]["gamma"],
            c=data_dict["params"]["c"],
            eta=data_dict["params"]["eta"],
            epsilon=data_dict["params"]["epsilon"],
        ),
        starting_step=starting_step,
    )

    return data


# fmt: off
def _validate_data_json(data_dict: constants.SDFLArgsDict) -> None:
    err_str: str = f"Invalid {DATA_JSON} file."

    if (
        set(constants.DEFAULT_PROBLEM_DATA.keys()) != set(data_dict.keys())
        or set(constants.DEFAULT_PROBLEM_DATA["params"].keys()) != set(data_dict["params"].keys())
    ):
        raise ValueError(err_str)

    def is_integer(elem):
        return np.issubdtype(type(elem), np.integer)

    def is_integer_or_float(elem):
        return is_integer(elem) or np.issubdtype(type(elem), np.floating)

    if (
        not is_integer(data_dict["max_eval"])
        or not is_integer_or_float(data_dict["min_step"])
        or not is_integer_or_float(data_dict["params"]["theta"])
        or not is_integer_or_float(data_dict["params"]["gamma"])
        or not is_integer_or_float(data_dict["params"]["c"])
        or not is_integer_or_float(data_dict["params"]["eta"])
        or not is_integer_or_float(data_dict["params"]["epsilon"])
    ):
        raise ValueError(err_str)
