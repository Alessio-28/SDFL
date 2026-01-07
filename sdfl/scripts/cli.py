import argparse as ap
import numpy as np

from ..test import run_test, problems
from .utils import SDFLData, import_data, export_data, DATA_JSON
from ..sdfl.core import parameters

def cli() -> None:
    usage: str = "%(prog)s [-h] [-l] [-f F [F ...] [-x X [X ...]] [-s S [S ...]] [--max-eval MAX] [--min-step MIN] [--params P P P P P] [-v]]"
    parser: ap.ArgumentParser = ap.ArgumentParser(usage = usage, formatter_class = ap.RawTextHelpFormatter)

    set_parser_list_group(parser)
    set_parser_run_group(parser)

    args: ap.Namespace = parser.parse_args()
    check_arguments(parser, args)

def check_arguments(parser: ap.ArgumentParser, args: ap.Namespace) -> None:
    if args.list_test_functions:
        problems.print_problem_names()
    if args.F:
        (functions, unavailable) = check_input_functions(args.F)
        if len(unavailable) > 0:
            parser.error(f"Function(s) unavailable: {", ".join(unavailable)}\n\t       Run -l to list available test functions.")
        data: SDFLData = check_data(args)
        run_test.setup_tests_and_run(functions, data, verbose = args.verbose)
    elif args.X or args.S or args.MAX or args.MIN or args.P or args.verbose:
        parser.error( "-x, -s, --max-eval, --min-step, --params, and -v require -f")

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

def check_data(args: ap.Namespace) -> SDFLData:
    data = import_data()
    data_dict = SDFLData.to_dict(data)

    if args.X or args.S or args.MAX or args.MIN or args.P:
        if args.X:
            data_dict["starting_point"] = np.array(args.X, dtype = np.float64)
        if args.S:
            data_dict["starting_step"] = np.array(args.S, dtype = np.float64)
        if args.MAX:
            data_dict["limit_eval"] = args.MAX
        if args.MIN:
            data_dict["limit_step"] = args.MIN
        if args.P:
            data_dict["params"] = args.P

        data = SDFLData.to_sdfl_data(data_dict)
        export_data(data)
    return data

def set_parser_list_group(parser: ap.ArgumentParser) -> None:
    list_group = parser.add_argument_group(title = "list")
    list_group.add_argument("-l", "--list-test-functions", action = "store_true", help = "Prints available test functions.")

def set_parser_run_group(parser: ap.ArgumentParser) -> None:
    description: str = (
        f"The following values can also be written in {DATA_JSON} except for -f and -v.\n"
        "Values of then previous run of the program are saved in that same file."
    )
    run_group = parser.add_argument_group(title = "algorithm options", description = description)
    run_group.add_argument("-f", "--function", nargs = "+", type = str, dest = "F", help = "Test function(s) to run.")
    run_group.add_argument("-x", "--point", nargs = "+", type = np.float64, dest = "X", help = "Starting point of the algorigthm. List of values separated by blank spaces.")
    run_group.add_argument("-s", "--step", nargs = "+", type = np.float64, dest = "S", help = "Starting step of the algorigthm. List of values separated by blank spaces.")
    run_group.add_argument("--max-eval", nargs = 1, type = int, dest = "MAX", help = "Max number of function evaluations before SDFL terminates.")
    run_group.add_argument("--min-step", nargs = 1, type = np.float64, dest = "MIN", help = "Minimum step value before SDFL terminates.")

    help: str = (
        "Parameters must be witten in the following order: theta gamma c eta epsilon.\n"
        f"{parameters._THETA_LOWER_BOUND} < theta < {parameters._THETA_UPPER_BOUND}, "
        f"gamma > {parameters._GAMMA_LOWER_BOUND}, "
        f"c > {parameters._C_LOWER_BOUND}, "
        f"eta > {parameters._ETA_LOWER_BOUND}, "
        f"epsilon > {parameters._EPSILON_LOWER_BOUND}"
    )
    run_group.add_argument("--params", nargs = 5, type = np.float64, dest = "P", help = help)
    run_group.add_argument("-v", "--verbose", action = "store_true", help = "Print intermediate results of the algorithm.")
