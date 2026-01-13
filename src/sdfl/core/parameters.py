import numpy as np

class Parameters:
    theta: np.float64   # in (0, 1)
    gamma: np.float64   # > 2
    c: np.float64       # > 0
    eta: np.float64     # > 0
    epsilon: np.float64 # > 0

    _bound_coeff: np.float64

    _THETA_LOWER_BOUND: int = 0
    _THETA_UPPER_BOUND: int = 1
    _GAMMA_LOWER_BOUND: int = 2
    _C_LOWER_BOUND: int = 0
    _ETA_LOWER_BOUND: int = 0
    _EPSILON_LOWER_BOUND: int = 0

    def __init__(self: Parameters, theta: np.float64, gamma: np.float64, c: np.float64, eta: np.float64, epsilon: np.float64) -> None:
        if self.validate_parameters_in_range(theta, gamma, c, eta, epsilon):
            str_error: str = (
                "Invalid parameter values: "
                f"{self._THETA_LOWER_BOUND} < theta < {self._THETA_UPPER_BOUND}, "
                f"gamma > {self._GAMMA_LOWER_BOUND}, "
                f"c > {self._C_LOWER_BOUND}, "
                f"eta > {self._ETA_LOWER_BOUND}, "
                f"epsilon > {self._EPSILON_LOWER_BOUND}"
            )
            raise ValueError(str_error)

        self.theta = theta
        self.gamma = gamma
        self.c = c
        self.eta = eta
        self.epsilon = epsilon
        self._bound_coeff = self.compute_bound_coeff()

    def compute_bound_coeff(self: Parameters) -> np.float64:
        return -self.gamma * self.c * self.epsilon

    def compute_bound(self: Parameters, step_size: np.float64) -> np.float64:
        return self._bound_coeff * step_size * step_size

    @staticmethod
    def validate_parameters_in_range(theta: np.float64, gamma: np.float64, c: np.float64, eta: np.float64, epsilon: np.float64) -> bool:
        return (theta <= Parameters._THETA_LOWER_BOUND or
                theta >= Parameters._THETA_UPPER_BOUND or
                gamma <= Parameters._GAMMA_LOWER_BOUND or
                c <= Parameters._C_LOWER_BOUND or
                eta <= Parameters._ETA_LOWER_BOUND or
                epsilon <= Parameters._EPSILON_LOWER_BOUND) # pyright: ignore[reportReturnType]
