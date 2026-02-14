import numpy as np

DEFAULT_MAX_EVAL: int = 5000
DEFAULT_MIN_STEP: np.float64 = np.float64(1e-6)
DEFAULT_THETA: np.float64 = np.float64(0.5)
DEFAULT_GAMMA: np.float64 = np.float64(2 + 1e-4)
DEFAULT_C: np.float64 = np.float64(1e-3)
DEFAULT_ETA: np.float64 = np.float64(1e-5)
DEFAULT_EPSILON: np.float64 = np.float64(0.1)

KEY_MAX_EVAL: str = "max_eval"
KEY_MIN_STEP: str = "min_step"
KEY_THETA: str = "theta"
KEY_GAMMA: str = "gamma"
KEY_C: str = "c"
KEY_ETA: str = "eta"
KEY_EPSILON: str = "epsilon"
