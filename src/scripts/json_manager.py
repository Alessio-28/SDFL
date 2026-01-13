import json
from ..sdfl.core.parameters import Parameters
from .sdfl_data_manager import *

DATA_JSON: str = "data.json"

# DATA_JSON_SCHEMA: str = '''{
#     "starting_point": [
#         2.0,
#         2.0
#     ],
#     "starting_step": [
#         1.0,
#         1.0
#     ],
#     "max_eval": 10000,
#     "min_step": 1e-08,
#     "theta": 0.5,
#     "gamma": 2.5,
#     "c": 1.0,
#     "eta": 1.0,
#     "epsilon": 1.0
# }'''

def export_data(data: SDFLData) -> None:
    data_dict = SDFLData.to_dict(data)
    with open(DATA_JSON, "w") as p:
        json.dump(data_dict, p, indent=4, separators=(",", ": "))

def import_data() -> SDFLData:
    try:
        with open(DATA_JSON, "r") as p:
            data_dict = json.load(p)
        data = SDFLData.to_SDFLData(data_dict)
        return data
    except ValueError:
        raise ValueError(f"{DATA_JSON} file invalid.")
    except OSError:
        return create_default_data_json()


def create_default_data_json() -> SDFLData:
    data: SDFLData = SDFLData(
        starting_point=DEFAULT_STARTING_POINT,
        starting_step=DEFAULT_STARTING_STEP,
        max_eval=DEFAULT_MAX_EVAL,
        min_step=DEFAULT_MIN_STEP,
        params=Parameters(
            theta=DEFAULT_THETA,
            gamma=DEFAULT_GAMMA,
            c=DEFAULT_C,
            eta=DEFAULT_ETA,
            epsilon=DEFAULT_EPSILON
        )
    )
    export_data(data)
    return data
