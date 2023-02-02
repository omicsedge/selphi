import os

import numpy as np
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
    with open(path, "rb") as f:
        return pickle.load(f)


def var_dump(path: str, obj):
    pickle_dump(obj, f"intermediate_variables_dump/{path}")
def var_load(path: str):
    return pickle_load(f"intermediate_variables_dump/{path}")

def nptype(array: np.ndarray):
    if len(array.shape) == 0:
        el = array
    elif len(array.shape) == 1:
        el = array[0]
    elif len(array.shape) == 2:
        el = array[0][0]
    elif len(array.shape) == 3:
        el = array[0][0][0]
    else:
        print('nptype: unknown')
        return
    print(f'vals: shape={array.shape}, span=[{array.min()}, {array.max()}]')
    t = type(el)
    try:
        print(f'type: {t}, span=[{np.iinfo(t).min}, {np.iinfo(t).max}]')
    except:
        print(f'type: {t}, span=[{np.finfo(t).min}, {np.finfo(t).max}]')

def dbg(val, name:str):
    if isinstance(val, np.ndarray):
        print(f"{name} ({val.shape}) dtype={val.dtype.type}")
    elif isinstance(val, list):
        if len(val) > 4:
            print(f"{name}: list[{len(val)}] = [{val[0]}, {val[1]}, {val[2]}, ...{val[-1]}]")
        else:
            print(f"{name} = {val}")
    elif isinstance(val, tuple):
        if len(val) > 4:
            print(f"{name}: tuple[{len(val)}] = ({val[0]}, {val[1]}, {val[2]}, ...{val[-1]})")
        else:
            print(f"{name} = {val}")
    else:
        print(f"{name} = {val}")


