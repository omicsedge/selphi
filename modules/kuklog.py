import os
from typing import IO
import time


def open_kuklog(file, mode='w'):
    """
    If the file's leaf directory doesn't exist, it creates all intermediate-level directories.
    """
    the_dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)
    return open(file, mode)

def kuklog_timestamp(comment: str, IO_obj: IO):
    IO_obj.write(f"{time.time() * 1000}\t{comment}\n")

def close_kuklog(IO_obj: IO):
    IO_obj.close()

