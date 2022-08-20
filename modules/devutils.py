import os

import pickle


def forced_open(file, mode='r'):
    """
    Wrapper around standard python `open`.
    If the file's leaf directory doesn't exist, it creates all intermediate-level directories.
    """
    the_dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)
    return open(file, mode)

def pickle_dump(obj, path: str):
    with forced_open(path, "wb") as f:
        pickle.dump(obj, f)
        return path
def pickle_load(path: str):
    with forced_open(path, "rb") as f:
        return pickle.load(f)


def var_dump(path: str, obj):
    pickle_dump(obj, f"intermediate_variables_dump/{path}")
def var_load(path: str):
    return pickle_load(f"intermediate_variables_dump/{path}")
