import numpy as np
import numpy.typing as npt

name: str = "maxl"
n: int = 20
starting_point: npt.NDArray[np.float64] = 1 + np.arange(n, dtype=np.float64)
starting_point[n // 2 :] *= -1


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    return np.max(np.abs(x))
