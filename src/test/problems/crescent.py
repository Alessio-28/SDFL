import numpy as np
import numpy.typing as npt

name: str = "crescent"
starting_point: npt.NDArray[np.float64] = np.array([-1.5, 2], dtype=np.float64)
n: int = starting_point.size


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f = x[0] ** 2 + (x[1] - 1) ** 2 - 1
    return np.abs(f) + x[1]
