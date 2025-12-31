from argparse import ArgumentParser, RawTextHelpFormatter, Namespace
from json import load, dump

from numpy import float64

from src import setup
from src.problems.problems import print_problem_names, get_problems
from src.sdfl import Parameters

SDFL_LOG : str = "sdfl.log"
PARAMETERS_JSON : str = "parameters.json"

def main() -> None:
    usage : str = "%(prog)s [-h] [-l] [-f F [F ...] [-p P P P P P] [--log [LOG]]]"
    parser : ArgumentParser = ArgumentParser(usage = usage, formatter_class = RawTextHelpFormatter)
    _ = parser.add_argument("-l", "--list-functions", action = "store_true", help = "Prints available functions.")
    _ = parser.add_argument("-f", "--function", nargs = "+", type = str, dest = "F", help = "Function(s) to run.")
    _ = parser.add_argument("--log", const = SDFL_LOG, nargs = "?", type = str,  help = f"Enables logging. Log file name can be passed to this argument. Default: {SDFL_LOG}.")
    help : str = (
        "Parameters must be witten in the following order: theta gamma c eta epsilon.\n"
        f"Parameters can also be written in {PARAMETERS_JSON}.\n"
        "Previous used values are saved in that same file.\n"
        f"{Parameters._THETA_LOWER_BOUND} < theta < {Parameters._THETA_UPPER_BOUND}, " # pyright: ignore[reportPrivateUsage]
        f"gamma > {Parameters._GAMMA_LOWER_BOUND}, "                                   # pyright: ignore[reportPrivateUsage]
        f"c > {Parameters._C_LOWER_BOUND}, "                                           # pyright: ignore[reportPrivateUsage]
        f"eta > {Parameters._ETA_LOWER_BOUND}, "                                       # pyright: ignore[reportPrivateUsage]
        f"epsilon > {Parameters._EPSILON_LOWER_BOUND}"                                 # pyright: ignore[reportPrivateUsage]
    )
    _ = parser.add_argument("-p", "--parameters", nargs = 5, type = float64, dest = "P", help = help)


    args : Namespace = parser.parse_args()
    check_arguments(parser, args)

def check_arguments(parser : ArgumentParser, args : Namespace) -> None:
    if args.list_functions:
        print_problem_names()
    if args.F:
        (functions, unavailable) = check_input_functions(args.F)
        if len(unavailable) > 0:
            parser.error(f"Function(s) unavailable: {", ".join(unavailable)}\n\t       Run -l to list available functions.")
        if args.log:
            setup.LOGGING = True
            setup.LOG_FILE = args.log
        param : Parameters = check_parameters(args.P)
        setup.run(functions, param)
    elif args.P or args.log:
        str_error : str = ""
        if args.P and args.log:
            str_error = "-p and --log require"
        elif args.P:
            str_error = "-p requires"
        else:
            str_error = "--log requires"
        parser.error(str_error + " --function")

def check_input_functions(functions : list[str]) -> tuple[list[str], list[str]]:
    probs : dict[str, str] = get_problems()
    prob_names : list[str] = list(probs.keys())
    unavailable : list[str] = []
    available : list[str] = []
    for f in functions:
        if f in prob_names:
            available.append(probs[f])
        else:
            unavailable.append(f)
    return (available, unavailable)

def check_parameters(parameters : list[float64] | None) -> Parameters:
    if parameters:
        param : Parameters = Parameters(
            theta   = parameters[0],
            gamma   = parameters[1],
            c       = parameters[2],
            eta     = parameters[3],
            epsilon = parameters[4]
        )
        export_parameters(param)
        return param
    else:
        return import_parameters()

def export_parameters(parameters : Parameters) -> None:
    param : dict[str, float64] = {
        "theta"   : parameters.theta,
        "gamma"   : parameters.gamma,
        "c"       : parameters.c,
        "eta"     : parameters.eta,
        "epsilon" : parameters.epsilon,
    }
    with open(f"./{PARAMETERS_JSON}", "w") as p:
        dump(param, p, indent = 4, separators = (",", ": "))

def import_parameters() -> Parameters:
    with open(f"./{PARAMETERS_JSON}") as p:
        data = load(p)
    param : Parameters = Parameters(
        theta   = data["theta"],
        gamma   = data["gamma"],
        c       = data["c"],
        eta     = data["eta"],
        epsilon = data["epsilon"]
    )
    return param

if __name__ == "__main__":
    main()
