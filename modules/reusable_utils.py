from typing import List, Set, TypeVar, Iterable
import os


"""
Independent or trivial utils, not related to the tool by themselves.
"""


def forced_open(file, mode="r"):
    """
    Wrapper around standard python `open`.
    If the file's leaf directory doesn't exist, it creates all intermediate-level directories.
    """
    the_dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)
    return open(file, mode)


T1 = TypeVar("T1")


def omit_duplicates(seq: Iterable[T1]) -> List[T1]:
    """Returns a list of only first occurrences of unique values from the input sequence"""
    seen: Set[T1] = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
