import argparse as ap
import numpy as np

from ..test import run_test, problems
from .parameters import import_parameters, export_parameters, DATA_JSON
# from logging import LOG_FILE
from ..sdfl.core import parameters

LOG_FILE = "HI"
def cli() -> None:
    usage: str = "%(prog)s [-h] [-l] [-f F [F ...] [-p P P P P P] [--log [LOG]]]"
    parser: ap.ArgumentParser = ap.ArgumentParser(usage = usage, formatter_class = ap.RawTextHelpFormatter)
    parser.add_argument("-l", "--list-test-functions", action = "store_true", help = "Prints available test functions.")
    parser.add_argument("-f", "--function", nargs = "+", type = str, dest = "F", help = "Function(s) to run.")
    parser.add_argument("--log", const = LOG_FILE, nargs = "?", type = str,  help = f"Enables logging. The name of the log file can be passed to this argument. Default: {LOG_FILE}.")
    help: str = (
        "Parameters must be witten in the following order: theta gamma c eta epsilon.\n"
        f"Parameters can also be written in {DATA_JSON}.\n"
        "Previous used values are saved in that same file.\n"
        f"{parameters._THETA_LOWER_BOUND} < theta < {parameters._THETA_UPPER_BOUND}, "
        f"gamma > {parameters._GAMMA_LOWER_BOUND}, "
        f"c > {parameters._C_LOWER_BOUND}, "
        f"eta > {parameters._ETA_LOWER_BOUND}, "
        f"epsilon > {parameters._EPSILON_LOWER_BOUND}"
    )
    parser.add_argument("-p", "--parameters", nargs = 5, type = np.float64, dest = "P", help = help)

    args: ap.Namespace = parser.parse_args()
    check_arguments(parser, args)

def check_arguments(parser: ap.ArgumentParser, args: ap.Namespace) -> None:
    if args.list_test_functions:
        problems.print_problem_names()
    if args.F:
        (functions, unavailable) = check_input_functions(args.F)
        if len(unavailable) > 0:
            parser.error(f"Function(s) unavailable: {", ".join(unavailable)}\n\t       Run -l to list available test functions.")
        if args.log:
            LOG_FILE = args.log
        params: parameters.Parameters = check_parameters(args.P)
        run_test.setup_tests_and_run(functions, params)
    elif args.P or args.log:
        str_error: str = ""
        if args.P and args.log:
            str_error = "-p and --log require"
        elif args.P:
            str_error = "-p requires"
        else:
            str_error = "--log requires"
        parser.error(str_error + " --function")

def check_input_functions(functions: list[str]) -> tuple[list[str], list[str]]:
    probs: dict[str, str] = problems.get_problems()
    prob_names: list[str] = list(probs.keys())
    unavailable: list[str] = []
    available: list[str] = []
    for f in functions:
        if f in prob_names:
            available.append(probs[f])
        else:
            unavailable.append(f)
    return (available, unavailable)

def check_parameters(params_list: list[np.float64] | None = None) -> parameters.Parameters:
    if params_list:
        params: parameters.Parameters = parameters.Parameters(
            theta   = params_list[0],
            gamma   = params_list[1],
            c       = params_list[2],
            eta     = params_list[3],
            epsilon = params_list[4]
        )
        export_parameters(params)
        return params
    else:
        return import_parameters()
