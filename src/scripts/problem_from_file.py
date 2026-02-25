from ..test import problem_manager as pm

PROBLEM_PY: str = "problem.py"

def copy_template() -> None:
    template: str = (
        'import numpy as np\n'
            'import numpy.typing as npt\n\n'
            'name: str = ""\n'
            'n: int = 0\n'
            'starting_point: npt.NDArray[np.float64] = []\n\n'
            'def feval(x: npt.NDArray[np.float64]) -> np.float64:\n'
            '\tpass\n'
    )

    with open(f"./{PROBLEM_PY}", "w") as p:
        p.write(template)

def load_problem() -> pm.Problem:
    return pm.import_problem(PROBLEM_PY.split(".")[0])
