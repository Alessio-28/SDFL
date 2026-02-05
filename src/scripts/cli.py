import argparse as ap
import numpy as np
import numpy.typing as npt

from . import constants
from . import data_json
from . import sdfl_data
from ..test import run_test, problems
from ..sdfl.core.parameters import Parameters

def check_arguments(parser: ap.ArgumentParser, args: ap.Namespace) -> None:
    if args.list_test_functions:
        problems.print_problem_names()
    if args.create_json:
        data_json.create_default_data_json()

    if args.F:
        try:
            f = check_input_function(args.F[0])
            data: sdfl_data.SDFLData = check_data(args, f)
        except KeyError:
            parser.error(f"Function unavailable.\n\t       Run -l to list available test functions.")
        except ValueError as e:
            parser.error(str(e)) # f"\nThe content of {data_json.DATA_JSON} is not valid.\n")

        try:
            run_test.run(data, verbose=args.verbose)
        except ValueError as ve:
            parser.error(str(ve))
        except RuntimeError as re:
            parser.error(str(re))

    elif args.X or args.S or args.MAX or args.MIN or args.P or args.verbose:
        parser.error("-x, -s, --max-eval, --min-step, --params, and -v require -f")

def check_input_function(f: str) -> problems.Problem:
    return problems.get_problems()[f]

def check_data(args: ap.Namespace, f: problems.Problem) -> sdfl_data.SDFLData:
    starting_step: npt.NDArray[np.float64] | None = None
    data = data_json.import_data()

    if args.X:
        f.starting_point = np.array(args.X, dtype=np.float64)
    if args.S:
        starting_step = np.array(args.S, dtype=np.float64)
    if args.MAX or args.MIN or args.P:
        if args.MAX:
            data[constants.KEY_MAX_EVAL] = int(args.MAX[0])
        if args.MIN:
            data[constants.KEY_MIN_STEP] = np.float64(args.MIN[0])
        if args.P:
            data[constants.KEY_THETA]   = args.P[0]
            data[constants.KEY_GAMMA]   = args.P[1]
            data[constants.KEY_C]       = args.P[2]
            data[constants.KEY_ETA]     = args.P[3]
            data[constants.KEY_EPSILON] = args.P[4]

        data_json.export_data(data)

    return data_json.dict_to_SDFLData(f, data, starting_step)

def cli() -> None:
    usage: str = "%(prog)s [-h] [-l] [-j] [-f F [-x X [X ...]] [-s S [S ...]] [--max-eval MAX] [--min-step MIN] [--params P P P P P] [-v]]"
    parser: ap.ArgumentParser = ap.ArgumentParser(usage=usage, formatter_class=ap.RawTextHelpFormatter)

    set_parser_utils_group(parser)
    set_parser_run_group(parser)

    args: ap.Namespace = parser.parse_args()
    check_arguments(parser, args)

def set_parser_utils_group(parser: ap.ArgumentParser) -> None:
    parser.add_argument("-l", "--list-test-functions", action="store_true", help="Prints available test functions.")
    parser.add_argument("-j", "--create-json", action="store_true", help=f"Creates default {data_json.DATA_JSON}")

def set_parser_run_group(parser: ap.ArgumentParser) -> None:
    description: str = (
        f"--max-eval, --min-step and --params can also be set in {data_json.DATA_JSON}.\n"
        "Values of the previous run of the program are saved in that same file.\n"
        "Starting point and starting step must be of the same size."
    )
    run_group = parser.add_argument_group(title="algorithm options", description=description)
    run_group.add_argument("-f", "--function", nargs=1, type=str, dest="F", help="Test function to run.")
    run_group.add_argument("-x", "--point", nargs="+", type=np.float64, dest="X", help="Starting point of the algorigthm. List of values separated by blank spaces.\nIf not used, defualt starting point for the given function is used.")
    run_group.add_argument("-s", "--steps", nargs="+", type=np.float64, dest="S", help="Starting step of the algorigthm. List of values separated by blank spaces.\nIf not used, starting steps get initialised appropriately.")
    run_group.add_argument("--max-eval", nargs=1, type=int, dest="MAX", help="Max number of function evaluations before SDFL terminates.")
    run_group.add_argument("--min-step", nargs=1, type=np.float64, dest="MIN", help="Minimum step value before SDFL terminates.")

    help: str = (
        "Parameters must be written in the following order: theta gamma c eta epsilon.\n"
        f"{Parameters._THETA_LOWER_BOUND} < theta < {Parameters._THETA_UPPER_BOUND}, "
        f"gamma > {Parameters._GAMMA_LOWER_BOUND}, "
        f"c > {Parameters._C_LOWER_BOUND}, "
        f"eta > {Parameters._ETA_LOWER_BOUND}, "
        f"epsilon > {Parameters._EPSILON_LOWER_BOUND}"
    )
    run_group.add_argument("--params", nargs=5, type=np.float64, dest="P", help=help)
    run_group.add_argument("-v", "--verbose", action="store_true", help="Print intermediate results of the algorithm.")
