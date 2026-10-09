if __name__ == "__main__":
    import os

    from src.scripts.cli import cli

    user_wd: str = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    try:
        cli()
    finally:
        os.chdir(user_wd)
