import json
from ..sdfl.core.parameters import Parameters
from .sdfl_data_manager import *

DATA_JSON: str = "data.json"

def export_data(data: SDFLData) -> None:
    data_dict = SDFLData.to_dict(data)
    with open(DATA_JSON, "w") as p:
        json.dump(data_dict, p, indent = 4, separators = (",", ": "))

def import_data() -> SDFLData:
    try:
        with open(DATA_JSON, "r") as p:
            data_dict = json.load(p)
        data = SDFLData.to_SDFLData(data_dict)
        return data
    except OSError:
        return create_default_data_json()

def create_default_data_json() -> SDFLData:
    data: SDFLData = SDFLData(
        DEFAULT_STARTING_POINT,
        DEFAULT_STARTING_STEP,
        DEFAULT_MAX_EVAL,
        DEFAULT_MIN_STEP,
        Parameters(
            DEFAULT_THETA,
            DEFAULT_GAMMA,
            DEFAULT_C,
            DEFAULT_ETA,
            DEFAULT_EPSILON
        )
    )
    export_data(data)
    return data
