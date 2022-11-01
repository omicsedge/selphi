from typing import Dict, List, Literal, Tuple, Union
from ctypes import c_uint8, cdll, Structure, c_int, Array, c_size_t, c_longlong, c_ulonglong, CDLL
import time
# import ctypes

import numpy as np


REF_PANEL = "/mnt/science-shared/data/users/adriano/GAIN/ref/20.reference_panel.30x.hg38.3202samples.vcf.gz"


class IBDs_extractor(object):
    lib: CDLL
    table_row_i: int

    def __init__(self, ref_panel_path: str):
        print("going to load extract_IBDs_from_ref_panel.lib")
        self.lib = cdll.LoadLibrary('./exe/extract_IBDs_from_ref_panel.lib')
        print("finished loading extract_IBDs_from_ref_panel.lib")
        self.obj = self.lib.IBDs_extractor_new(ref_panel_path.encode())
        self.table_row_i = 0

    def __enter__(self):
        pass # file is opened at initialization

    def close(self):
        self.lib.close_(self.obj)

    def readlines(self, lines_n: int) -> Union[int, None]:
        lines_read_n = self.lib.readlines(self.obj, lines_n)
        if lines_read_n == -1:
            return None
        return lines_read_n

    def readlines2(self,
        lines_n: int,
        haploid_i: int,
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, None]]:
        timestart = time.time()
        c_int_array = c_uint8 * lines_n
        array = c_int_array()
        lines_read_n: int = self.lib.readlines2(self.obj, array, lines_n, haploid_i) # array is passed by ref
        print(f"--- readlines2 exe time : {(time.time() - timestart)} seconds ---")
        timestart = time.time()
        nparr: np.ndarray = np.ctypeslib.as_array(array) # type: ignore
        print(f"--- np array conversion time : {(time.time() - timestart)} seconds ---")
        # print(f"{nparr.shape=}")
        # print(f"{nparr.sum()=}")
        # print(f"{np.where(nparr)=}")
        # print(nparr)
        # for i in array: print(f"{i} ")
        if lines_read_n == -1:
            return np.zeros((0,), dtype=np.uint8), None
        return nparr, lines_read_n

    def readlines3(self,
        lines_n: int,
        total_haploids_n: int,
        refill: Union[np.ndarray, None] = None,
    # ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, None]]:
    ) -> Tuple[np.ndarray,int]:
        """
        Note: usually the returned number of lines read would equal to the set number `lines_n=`,
            but may be a lower number if reached the end of the file.

        `if lines_read_n == 0:` is the test "if there is nothing more to read from the file"
        """
        timestart = time.time()

        if lines_n <= 0:
            raise ValueError(f"lines_n denotes the number of lines to read, and therefore has to be a positive integer")
        if total_haploids_n <= 0:
            raise ValueError(f"total_haploids_n denotes the number of haploids in the vcfgz file, and therefore has to be a positive integer")

        data_c_type = c_uint8
        data_np_type = np.uint8

        c_lines_n = c_longlong(lines_n)
        c_total_haploids_n = c_longlong(total_haploids_n)
        c_uint8_2d_array_type = lines_n * (total_haploids_n * data_c_type)


        if refill is not None:
            if tuple(refill.shape) != (lines_n, total_haploids_n):
                raise ValueError(f"passed array to refill is of wrong shape: {refill.shape}. Expected shape: (lines_n, total_haploids_n) == {(lines_n, total_haploids_n)}")
            if refill.dtype.type != data_np_type:
                raise ValueError(f"passed array to refill is of wrong dtype: {refill.dtype.type}. Expected type: {data_np_type}")
            array = np.ctypeslib.as_ctypes(refill) # type: ignore
            # np.ctypeslib.as_ctypes  
        else:
            array = c_uint8_2d_array_type() # initializes with int values = 0

        # print(f"going to call lib.readlines3() with array at: {array} , of size ({total_haploids_n}*{lines_n})")
        # print(f"of type: {c_uint8_2d_array_type})")
        table_row_i: int = self.lib.readlines3(self.obj, array, c_lines_n, c_total_haploids_n) # array is passed by ref
        lines_read_n = table_row_i - self.table_row_i
        self.table_row_i = table_row_i

        # print(f"readlines3 exe time : {(time.time() - timestart)} s")
        # print(f"--- readlines3 exe time : {(time.time() - timestart)} seconds ---")
        # timestart = time.time()
        nparr: np.ndarray = np.ctypeslib.as_array(array) # type: ignore
        # print(f"--- np array conversion time : {(time.time() - timestart)} seconds ---")

        return nparr, lines_read_n

        # # print(f"{nparr.shape=}")
        # # print(f"{nparr.sum()=}")
        # # print(f"{np.where(nparr)=}")
        # # print(nparr)
        # # for i in array: print(f"{i} ")
        # if lines_read_n == -1:
        #     return np.zeros((0,), dtype=np.uint8), None
        # return nparr, lines_read_n


    def readlines3_selfcontained(self,
        lines_n: int,
        total_haploids_n: int,
        refill: Union[np.ndarray, None] = None,
    # ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, None]]:
    ) -> Tuple[np.ndarray,int]:
        timestart = time.time()

        if lines_n <= 0:
            raise ValueError(f"lines_n denotes the number of lines to read, and therefore has to be a positive integer")
        if total_haploids_n <= 0:
            raise ValueError(f"total_haploids_n denotes the number of haploids in the vcfgz file, and therefore has to be a positive integer")

        data_c_type = c_uint8
        data_np_type = np.uint8

        c_lines_n = c_longlong(lines_n)
        c_total_haploids_n = c_longlong(total_haploids_n)
        c_uint8_2d_array_type = lines_n * (total_haploids_n * data_c_type)


        if refill is not None:
            if tuple(refill.shape) != (lines_n, total_haploids_n):
                raise ValueError(f"passed array to refill is of wrong shape: {refill.shape}. Expected shape: (lines_n, total_haploids_n) == {(lines_n, total_haploids_n)}")
            if refill.dtype.type != data_np_type:
                raise ValueError(f"passed array to refill is of wrong dtype: {refill.dtype.type}. Expected type: {data_np_type}")
            array = np.ctypeslib.as_ctypes(refill) # type: ignore
            # np.ctypeslib.as_ctypes  
        else:
            array = c_uint8_2d_array_type() # initializes with int values = 0

        print(f"going to call lib.readlines3_selfcontained() with array at: {array} , of size ({total_haploids_n}*{lines_n})")
        print(f"of type: {c_uint8_2d_array_type})")
        table_row_i: int = self.lib.readlines3_selfcontained(self.obj, array, c_lines_n, c_total_haploids_n) # array is passed by ref
        lines_read_n = table_row_i - self.table_row_i
        self.table_row_i = table_row_i

        print(f"--- readlines3_selfcontained exe time : {(time.time() - timestart)} seconds ---")
        timestart = time.time()
        nparr: np.ndarray = np.ctypeslib.as_array(array) # type: ignore
        print(f"--- np array conversion time : {(time.time() - timestart)} seconds ---")

        return nparr, lines_read_n


    def readlines3_int(self,
        lines_n: int,
        total_haploids_n: int,
    ) -> Tuple[np.ndarray,int]:
        timestart = time.time()

        c_uint8_2d_array_type = lines_n * (total_haploids_n * c_uint8)
        array = c_uint8_2d_array_type() # initializes with int values = 0
        print(f"going to call lib.readlines3_int() with array at: {array} , of size ({total_haploids_n}*{lines_n})")
        print(f"of type: {c_uint8_2d_array_type})")
        lines_read_n: int = self.lib.readlines3_int(self.obj, array, lines_n, total_haploids_n) # array is passed by ref
        print(f"--- readlines3 exe time : {(time.time() - timestart)} seconds ---")
        timestart = time.time()
        nparr: np.ndarray = np.ctypeslib.as_array(array) # type: ignore
        print(f"--- np array conversion time : {(time.time() - timestart)} seconds ---")

        return nparr, lines_read_n




if __name__ == '__main__':

    # # c_int_array = c_int * 10
    # # a = c_int_array()
    # # print(a)

    # # for i in a: print(i, end=" ")

    # main_start = time.time()
    # f = IBDs_extractor(REF_PANEL)

    # segment, size = f.readlines2(lines_n=2_000_000, haploid_i=6)
    # print(f"{size=}")
    # print(f"{segment.shape=}")
    # print(f"{segment.sum()=}")
    # print(f"--- main time: {(time.time() - main_start)} seconds ---")

    # del f

    # main_start = time.time()
    # f2 = IBDs_extractor(REF_PANEL)

    # segment2, size2 = f2.readlines2(lines_n=2_000_000, haploid_i=6)
    # print(f"{size2=}")
    # print(f"{segment2.shape=}")
    # print(f"{segment2.sum()=}")
    # print(f"--- main time: {(time.time() - main_start)} seconds ---")


    # print(f"{(segment == segment2).all()=}")

    # # read1_start = time.time()
    # # print(f"{f.readlines2(lines_n=10)=}")
    # # print(f"--- read1 time: {(time.time() - read1_start)} seconds ---")

    # # read2_start = time.time()
    # # print(f"{f.readlines2(lines_n=90)=}")
    # # print(f"--- read2 time: {(time.time() - read2_start)} seconds ---")



    main_start = time.time()
    f = IBDs_extractor(REF_PANEL)

    segment, lines_read_n = f.readlines3(lines_n=2_000_000, total_haploids_n=6404)
    # segment, lines_read_n = f.readlines3(lines_n=1_647_105, total_haploids_n=6404)
    # segment, lines_read_n = f.readlines2(lines_n=1_647_105, haploid_i=6)
    print(f"{segment}")
    print(f"{segment.shape=}")
    print(f"{segment.sum()=}")
    print(f"--- main time: {(time.time() - main_start)} seconds ---")

    


