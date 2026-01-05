from collections.abc import Callable
import numpy as np
import numpy.typing as npt

type Point = npt.NDArray[np.float64]
type ObjectiveFunction = Callable[[Point], np.float64]
