from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class Parameters:
    theta : np.float64
    gamma : np.float64
    c : np.float64
    eta : np.float64
    epsilon : np.float64

def SDFL(x_0 : npt.NDArray[np.float64], step_0 : npt.NDArray[np.float64], param : Parameters) -> npt.NDArray[np.float64]:
    k : int = 0
    bound : np.float64 = np.float64(1e-10)
    temp : np.float64 = np.float64(0)
    success : bool

    n : int = x_0.size
    x : npt.NDArray[np.float64] = x_0.copy()
    y : npt.NDArray[np.float64]
    while temp > bound:
        y = x.copy()
        for i in range(n):
            success = True

        k += 1
    return np.zeros((1,), dtype = np.float64)
