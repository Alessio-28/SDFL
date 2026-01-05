import logging
import pathlib
import os

LOG_FILE: pathlib.PurePath = pathlib.PurePath("./log/sdfl.log")
MODE: str = "a"
LEVEL = logging.INFO 

test_log: logging.Logger

def setup_test_logging() -> None:
    os.makedirs(LOG_FILE.parent, exist_ok = True)

    global test_log

    test_fh: logging.FileHandler = logging.FileHandler(filename = LOG_FILE, mode = MODE)
    test_fh.setLevel(LEVEL)

    test_log = logging.getLogger(name = "test")
    test_log.setLevel(LEVEL)
    test_log.addHandler(test_fh)
