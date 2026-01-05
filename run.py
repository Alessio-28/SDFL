import os

user_wd: str = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if __name__ == "__main__":
    import sdfl.scripts.cli
    sdfl.scripts.cli.cli()

os.chdir(user_wd)
