from typing import List, Set, Tuple, TypeVar, Union, Iterable, Dict, Literal
from ctypes import c_int8, cdll, c_longlong, CDLL
import time

import numpy as np


LIB_PATH = './modules/vcfgz_reader.lib'


class vcfgz_reader(object):
    lib: CDLL
    table_row_i: int
    total_haploids_n: int

    def __init__(self, ref_panel_path: str):
        print(f"going to load lib at {LIB_PATH}")
        self.lib = cdll.LoadLibrary(LIB_PATH)
        print(f"finished loading lib at {LIB_PATH}")
        self.obj = self.lib.vcfgz_reader_new(ref_panel_path.encode())
        self.table_row_i = 0
        self.total_haploids_n = self.lib.get_total_haploids(self.obj)
        print(f"file initialized with total_haploids = {self.total_haploids_n}")

    def __enter__(self):
        pass # file is opened at initialization

    def close(self):
        self.lib.close_(self.obj)

    def readlines(self,
        lines_n: int,
        refill: Union[np.ndarray, None] = None,
    ) -> Tuple[np.ndarray,int]:
        """
        Note: usually the returned number of lines read would equal to the set number `lines_n=`,
            but may be a lower number if reached the end of the file.

        `if lines_read_n == 0:` is the test "if there is nothing more to read from the file"
        """

        if lines_n <= 0:
            raise ValueError(f"lines_n denotes the number of lines to read, and therefore has to be a positive integer")

        data_c_type = c_int8
        data_np_type = np.int8

        c_lines_n = c_longlong(lines_n)
        c_total_haploids_n = c_longlong(self.total_haploids_n)
        c_int8_2d_array_type = lines_n * (self.total_haploids_n * data_c_type)


        if refill is not None:
            if tuple(refill.shape) != (lines_n, self.total_haploids_n):
                raise ValueError(f"passed array to refill is of wrong shape: {refill.shape}. Expected shape: (lines_n, self.total_haploids_n) == {(lines_n, self.total_haploids_n)}")
            if refill.dtype.type != data_np_type:
                raise ValueError(f"passed array to refill is of wrong dtype: {refill.dtype.type}. Expected type: {data_np_type}")
            array = np.ctypeslib.as_ctypes(refill) # type: ignore
        else:
            array = c_int8_2d_array_type() # initializes with int values = 0

        table_row_i: int = self.lib.readlines(self.obj, array, c_lines_n, c_total_haploids_n) # array is passed by ref
        lines_read_n = table_row_i - self.table_row_i
        self.table_row_i = table_row_i

        nparr: np.ndarray = np.ctypeslib.as_array(array) # type: ignore

        return nparr, lines_read_n

        # return np.ctypeslib.as_array(array), lines_read_n # type: ignore



"""
unit test
"""
if __name__ == '__main__':
    REF_PANEL = "/mnt/science-shared/data/users/adriano/GAIN/ref/20.reference_panel.30x.hg38.3202samples.vcf.gz"

    main_start = time.time()
    f = vcfgz_reader(REF_PANEL)

    segment, lines_read_n = f.readlines(lines_n=2_000_000)
    print(f"{segment}")
    print(f"{segment.shape=}")
    print(f"{segment.sum()=}")
    print(f"--- main time: {(time.time() - main_start)} seconds ---")


