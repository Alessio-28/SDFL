import argparse as ap
import numpy as np

from ..sdfl.core.parameters import Parameters
from . import constants
from . import json_manager
from . import sdfl_data_manager as sdfl_data
from ..test import run_test, problems

def cli() -> None:
    usage: str = "%(prog)s [-h] [-l] [-j] [-f F [-x X [X ...]] [-s S [S ...]] [--max-eval MAX] [--min-step MIN] [--params P P P P P] [-v]]"
    parser: ap.ArgumentParser = ap.ArgumentParser(usage=usage, formatter_class=ap.RawTextHelpFormatter)

    set_parser_utils_group(parser)
    set_parser_run_group(parser)

    args: ap.Namespace = parser.parse_args()
    check_arguments(parser, args)

def check_arguments(parser: ap.ArgumentParser, args: ap.Namespace) -> None:
    if args.list_test_functions:
        problems.print_problem_names()
    if args.create_json:
        json_manager.create_default_data_json()
    if args.F:
        f = check_input_function(args.F[0])
        if f == None:
            parser.error(f"Function unavailable.\n\t       Run -l to list available test functions.")
        data: sdfl_data.SDFLData = check_data(args)
        # run_test.setup_tests_and_run(functions, data, verbose=args.verbose)
    elif args.X or args.S or args.MAX or args.MIN or args.P or args.verbose:
        parser.error("-x, -s, --max-eval, --min-step, --params, and -v require -f")

def check_input_function(f: str) -> problems.Problem | None:
    probs: dict[str, problems.Problem] = problems.get_problems()
    if f in probs.keys():
        return probs[f]
    else
        return None

def check_data(args: ap.Namespace) -> sdfl_data.SDFLData:
    try:
        data = json_manager.import_data()
        data_dict = sdfl_data.SDFLData.to_dict(data)

        if args.X or args.S or args.MAX or args.MIN or args.P:
            if args.X:
                data_dict[constants.KEY_STARTING_POINT] = np.array(args.X, dtype=np.float64)
            if args.S:
                data_dict[constants.KEY_STARTING_STEP] = np.array(args.S, dtype=np.float64)
            if args.MAX:
                data_dict[constants.KEY_MAX_EVAL] = int(args.MAX[0])
            if args.MIN:
                data_dict[constants.KEY_MIN_STEP] = np.float64(args.MIN[0])
            if args.P:
                data_dict[constants.KEY_THETA] = args.P[0]
                data_dict[constants.KEY_GAMMA] = args.P[1]
                data_dict[constants.KEY_C] = args.P[2]
                data_dict[constants.KEY_ETA] = args.P[3]
                data_dict[constants.KEY_EPSILON] = args.P[4]

            data = sdfl_data.SDFLData.to_SDFLData(data_dict)
    except ValueError:
        exit(f"\nThe content of {json_manager.DATA_JSON} is not valid.\n") # Valid {DATA_JSON} file example:\n{DATA_JSON_SCHEMA}\n")
    json_manager.export_data(data)
    return data

def set_parser_utils_group(parser: ap.ArgumentParser) -> None:
    parser.add_argument("-l", "--list-test-functions", action="store_true", help="Prints available test functions.")
    parser.add_argument("-j", "--create-json", action="store_true", help=f"Creates default {json_manager.DATA_JSON}")

def set_parser_run_group(parser: ap.ArgumentParser) -> None:
    description: str = (
        f"--max-eval, --min-step and --params can also be set in {json_manager.DATA_JSON}.\n"
        "Values of the previous run of the program are saved in that same file.\n"
        "Starting point and starting step must be of the same size."
    )
    run_group = parser.add_argument_group(title="algorithm options", description=description)
    run_group.add_argument("-f", "--function", nargs=1, type=str, dest="F", help="Test function to run.")
    run_group.add_argument("-x", "--point", nargs="+", type=np.float64, dest="X", help="Starting point of the algorigthm. List of values separated by blank spaces.\nIf not used, defualt starting point for the given is used.")
    run_group.add_argument("-s", "--steps", nargs="+", type=np.float64, dest="S", help="Starting step of the algorigthm. List of values separated by blank spaces.\nIf not used, starting steps get initialised appropriately.")
    run_group.add_argument("--max-eval", nargs=1, type=int, dest="MAX", help="Max number of function evaluations before SDFL terminates.")
    run_group.add_argument("--min-step", nargs=1, type=np.float64, dest="MIN", help="Minimum step value before SDFL terminates.")

    help: str = (
        "Parameters must be witten in the following order: theta gamma c eta epsilon.\n"
        f"{Parameters._THETA_LOWER_BOUND} < theta < {Parameters._THETA_UPPER_BOUND}, "
        f"gamma > {Parameters._GAMMA_LOWER_BOUND}, "
        f"c > {Parameters._C_LOWER_BOUND}, "
        f"eta > {Parameters._ETA_LOWER_BOUND}, "
        f"epsilon > {Parameters._EPSILON_LOWER_BOUND}"
    )
    run_group.add_argument("--params", nargs=5, type=np.float64, dest="P", help=help)
    run_group.add_argument("-v", "--verbose", action="store_true", help="Print intermediate results of the algorithm.")
