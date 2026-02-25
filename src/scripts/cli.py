import argparse as ap
import numpy as np
import numpy.typing as npt

from . import constants
from . import data_json
from . import sdfl_data
from . import problem_from_file
from ..test import run_test, problem_manager as pm
from ..sdfl.core.parameters import Parameters

def check_arguments(parser: ap.ArgumentParser, args: ap.Namespace) -> None:
    if args.list_problems:
        pm.print_problem_names()
    if args.create_json:
        data_json.create_default_data_json()
    if args.create_template_problem:
        problem_from_file.copy_template()

    if args.PROBLEM or args.from_file:
        try:
            if args.PROBLEM:
                p = check_input_problem(args.PROBLEM[0])
            else:
                p = problem_from_file.load_problem()

            data = check_args(args, p)
            run_test.run(data, verbose=args.verbose)
        except (ImportError, KeyError, ValueError, FloatingPointError) as e:
            parser.error(str(e))

    elif args.X or args.S or args.MAX or args.MIN or args.P or args.verbose:
        parser.error("-x, -s, --max-eval, --min-step, --params, and -v require -p or --from-file.")

def check_input_problem(p: str) -> pm.Problem:
    try:
        return pm.get_default_problems()[p]
    except KeyError:
        raise KeyError("Problem not available.\n\t       Run -l to list available problems.")

def check_args(args: ap.Namespace, p: pm.Problem) -> sdfl_data.SDFLData:
    starting_step: npt.NDArray[np.float64] | None = None
    data = data_json.import_data()

    if args.X:
        p.starting_point = np.array(args.X, dtype=np.float64)
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

    return data_json.dict_to_SDFLData(p, data, starting_step)

#############################
#                           #
# Command line parser setup #
#                           #
#############################

def cli() -> None:
    usage: str = "%(prog)s [-h] [-l] [-j] [(-p PROBLEM | --from-file) [-x X [X ...]] [-s S [S ...]] [--max-eval MAX] [--min-step MIN] [--params P P P P P] [-v]]"
    parser: ap.ArgumentParser = ap.ArgumentParser(usage=usage, formatter_class=ap.RawTextHelpFormatter)

    set_parser_utils_group(parser)
    set_parser_run_group(parser)

    args: ap.Namespace = parser.parse_args()
    check_arguments(parser, args)

def set_parser_utils_group(parser: ap.ArgumentParser) -> None:
    parser.add_argument("-l", "--list-problems", action="store_true", help="Prints available problems.")
    parser.add_argument("-j", "--create-json", action="store_true", help=f"Creates default {data_json.DATA_JSON}")
    parser.add_argument("-t", "--create-template-problem", action="store_true", help=f"Creates default {problem_from_file.PROBLEM_PY}")

def set_parser_run_group(parser: ap.ArgumentParser) -> None:
    description: str = (
        f"--max-eval, --min-step and --params can also be set in {data_json.DATA_JSON}.\n"
        "Values of the previous run of the program are saved in that same file.\n"
        "Starting point and starting step must be of the same size."
    )
    params_help: str = (
        "Parameters must be written in the following order: theta gamma c eta epsilon.\n"
        f"{Parameters._THETA_LOWER_BOUND} < theta < {Parameters._THETA_UPPER_BOUND}, "
        f"gamma > {Parameters._GAMMA_LOWER_BOUND}, "
        f"c > {Parameters._C_LOWER_BOUND}, "
        f"eta > {Parameters._ETA_LOWER_BOUND}, "
        f"epsilon > {Parameters._EPSILON_LOWER_BOUND}"
    )

    run_group = parser.add_argument_group(title="algorithm options", description=description)

    problem_group = run_group.add_mutually_exclusive_group()
    problem_group.add_argument("-p", "--problem", nargs=1, type=str, dest="PROBLEM", help="Problem to run.")
    problem_group.add_argument("--from-file", action="store_true", help=f"A problem can be defined in {problem_from_file.PROBLEM_PY} file and then tested by the algorithm.")
    
    run_group.add_argument("-x", "--point", nargs="+", type=np.float64, dest="X", help="Starting point of the algorigthm. List of values separated by blank spaces.\nIf not used, defualt starting point for the given problem is used.")
    run_group.add_argument("-s", "--steps", nargs="+", type=np.float64, dest="S", help="Starting step values of the algorigthm. List of values separated by blank spaces.\nIf not used, starting steps get initialised appropriately.")
    run_group.add_argument("--max-eval", nargs=1, type=int, dest="MAX", help="Maximum number of function evaluations before SDFL terminates.")
    run_group.add_argument("--min-step", nargs=1, type=np.float64, dest="MIN", help="Minimum step value before SDFL terminates.")

    run_group.add_argument("--params", nargs=5, type=np.float64, dest="P", help=params_help)
    run_group.add_argument("-v", "--verbose", action="store_true", help="Print intermediate results of the algorithm.")
