import os

user_wd: str = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if __name__ == "__main__":
    import src.scripts.cli as sdfl
    sdfl.cli()

os.chdir(user_wd)
