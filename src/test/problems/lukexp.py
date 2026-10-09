import numpy as np
import numpy.typing as npt

name: str = "exp"
starting_point: npt.NDArray[np.float64] = np.array([0.5, 0, 0, 0, 0], dtype=np.float64)
n: int = starting_point.size


def feval(x: npt.NDArray[np.float64]) -> np.float64:
    f1 = x[0] + x[1] * _T
    f2 = 1 + x[2:] @ _U
    return np.max(f1 / f2 - _expT)


_T = np.linspace(-1, 1, 21, dtype=np.float64)
_U = np.array([_T, _T**2, _T**3])
_expT = np.exp(_T)
