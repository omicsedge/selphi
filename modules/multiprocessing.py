from typing import Dict, List, Literal, Tuple, Union, OrderedDict as OrderedDict_type
import sys
import os


def split_range_into_n_whole_even_subranges(range_len: int, n: int):
    """
    Cuts the slices into the most even ranges
    e.g., cutting 38 into 4 slice ranges yields:
      0-10    |10|
      10-20   |10|
      20-29   |9|
      29-38   |9|
    """

    if not (0 < n <= range_len):
        raise ValueError("`range_len` and `n` have to be positive integers, and `n` has to be no less than `range_len`")
    pass

    min_subrange_len: int = int(range_len/n)
    closest_multiple: int = n*min_subrange_len
    mod: int = range_len - closest_multiple


    slice_idxs: List[int] = []
    slice_idxs.append(0)
    for i in range(1, mod+1):
        slice_idxs.append(slice_idxs[i-1] + min_subrange_len + 1)
    for i in range(mod+1, n+1):
        slice_idxs.append(slice_idxs[i-1] + min_subrange_len)


    return slice_idxs



def mute_stdout():
    sys.stdout = open(os.devnull, 'w')



