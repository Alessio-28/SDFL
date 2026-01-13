import numpy as np
import numpy.typing as npt

from ..sdfl.core.typing import Point

_N: int = 2
DEFAULT_STARTING_POINT: Point = np.array([1] * _N, dtype=np.float64)
DEFAULT_STARTING_STEP: npt.NDArray[np.float64] = np.array([1] * _N, dtype=np.float64)
DEFAULT_MAX_EVAL: int = 10_000
DEFAULT_MIN_STEP: np.float64 = np.float64(1e-8)
DEFAULT_THETA: np.float64 = np.float64(0.5)
DEFAULT_GAMMA: np.float64 = np.float64(2.5)
DEFAULT_C: np.float64 = np.float64(1)
DEFAULT_ETA: np.float64 = np.float64(1)
DEFAULT_EPSILON: np.float64 = np.float64(1)

KEY_STARTING_POINT: str = "starting_point"
KEY_STARTING_STEP: str = "starting_step"
KEY_MAX_EVAL: str = "max_eval"
KEY_MIN_STEP: str = "min_step"
KEY_THETA: str = "theta"
KEY_GAMMA: str = "gamma"
KEY_C: str = "c"
KEY_ETA: str = "eta"
KEY_EPSILON: str = "epsilon"
