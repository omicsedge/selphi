from typing import List, Set, Tuple, Type, TypeVar, Union, Iterable, Dict, Literal
from ctypes import Array, c_int8, c_int, cdll, c_longlong, CDLL, string_at, wstring_at, c_char_p, create_string_buffer, c_void_p
import time

import numpy as np


LIB_PATH = './lib/vcfgz_reader.lib'


COLS_LEN = [
    4,    #0 #CHROM
    11,   #1 POS
    350,  #2 ID
    350,  #3 REF
    350,  #4 ALT
    10,   #5 QUAL
    100,  #6 FILTER
    1000, #7 INFO
    50,   #8 FORMAT
]


class vcfgz_reader(object):
    lib: CDLL
    table_row_i: int
    total_haploids_n: int

    table_header: str
    table_delimiter: str

    samples: List[str]

    def __init__(self, ref_panel_path: str):
        # print(f"going to load lib at {LIB_PATH}")
        self.lib = cdll.LoadLibrary(LIB_PATH)
        # print(f"finished loading lib at {LIB_PATH}")

        # self.obj = self.lib.vcfgz_reader_new(ref_panel_path.encode())
        vcfgz_reader_constructor_wrapper = self.lib.vcfgz_reader_new
        vcfgz_reader_constructor_wrapper.restype = c_void_p
        # vcfgz_reader_constructor_wrapper.argtypes = [c_char_p]
        self.obj = c_void_p(vcfgz_reader_constructor_wrapper(ref_panel_path.encode()))

        self.table_row_i = 0
        self.total_haploids_n = self.lib.get_total_haploids(self.obj)

        CACHE_LINE_LEN = self.lib.get_cache_line_len(self.obj) # type: int
        cache_line_copy = create_string_buffer(b'\x00' * CACHE_LINE_LEN) # allocate string of NUL characters
        table_header_len = self.lib.fill_table_header(self.obj, cache_line_copy)

        self.table_header = cache_line_copy.value[:table_header_len].decode("utf-8")
        self.samples = self.table_header.split('\t')[9:]
        # print(f"file initialized with total_haploids = {self.total_haploids_n}")

    def __enter__(self):
        pass # file is opened at initialization

    def close(self):
        self.lib.close_(self.obj)


    def read_columns(self,
        lines_n: int,
        CHROM:     Union[np.ndarray, bool] = False,
        POS:       Union[np.ndarray, bool] = False,
        ID:        Union[np.ndarray, bool] = False,
        REF:       Union[np.ndarray, bool] = False,
        ALT:       Union[np.ndarray, bool] = False,
        QUAL:      Union[np.ndarray, bool] = False,
        FILTER:    Union[np.ndarray, bool] = False,
        INFO:      Union[np.ndarray, bool] = False,
        FORMAT:    Union[np.ndarray, bool] = False,
        GENOTYPES: Union[np.ndarray, bool] = False, # a 2d array, cuz may be several columns
    ):
        """
        Reads the opened vcf.gz file, and fills up the arrays specified in arguments.

        Returns 2 values:
         1. number of lines read
         2. list of arrays as per requested columns (inferred from the arguments)


        Specify any subset of possible columns (e.g. GENOTYPES=True, ID=True, QUAL=True),
            and the function will return those arrays as the second value,
        The order in which the columns arrays appear in the returned list is
            in the order as the arguments defined here
            (from example above: return ID, QUAL, GENOTYPES)

        From returned columns arrays, make sure to only use values up to the *number of lines read*.
        You may want to pass the previously returned array for refill
            to prevent unnecessary copying and memory allocation and deallocation.
        For the same reason, this function does not slice the resulting arrays,
            but returns the *number of lines read* integer.

        Note: usually the returned number of lines read would equal to the set number `lines_n=`,
            but may be a lower number if reached the end of the file.

        `if lines_read_n == 0:` is the test "if there is nothing more to read from the file"
        """
        if lines_n < 0:
            raise ValueError(f"lines_n denotes the number of lines to read, and therefore has to be a non-negative integer")

        metadata_cols_names: List[str] = ['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO','FORMAT']
        metadata_cols_input: List[Union[np.ndarray, bool]] = [CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT]
        metadata_cols_py: List[Union[np.ndarray, Literal[False]]] = []
        metadata_cols_c: List[Union[c_void_p, Literal[None]]] = []

        # parsing the input, allocating arrays where necessary
        for i, (col_i_name, col_i) in enumerate(zip(metadata_cols_names, metadata_cols_input)):
            dtype = f'|S{COLS_LEN[i]}'
            if col_i is False:
                metadata_cols_py.append(False)
                metadata_cols_c.append(None)
            elif col_i is True:
                col_i_: np.ndarray = np.char.array(np.zeros(lines_n, dtype=dtype))#type:ignore # char *is* a known member of np
                metadata_cols_py.append(col_i_)
                metadata_cols_c.append(col_i_.ctypes.data_as(c_void_p))#type:ignore
            else: # col_i is np.ndarray
                if tuple(col_i.shape) != (lines_n,):
                    raise ValueError(f"passed array-to-refill for {col_i_name} is of wrong shape: {col_i.shape}. Expected shape: (lines_n,) == {(lines_n, )}")
                if not (col_i.dtype.type == np.bytes_ and col_i.dtype.itemsize == COLS_LEN[i]):
                    raise ValueError(f"passed array-to-refill for {col_i_name} is of wrong dtype: {col_i.dtype}. Expected type: {dtype}")
                metadata_cols_py.append(col_i)
                metadata_cols_c.append(col_i.ctypes.data_as(c_void_p))


        data_c_type = c_int8
        data_np_type = np.int8

        c_lines_n = c_longlong(lines_n)
        c_total_haploids_n = c_longlong(self.total_haploids_n)
        c_int8_2d_array_type = lines_n * (self.total_haploids_n * data_c_type)
        # c_int8_2d_array_type_py = Type[Array[Array[c_int8]]]

        genotype_table_py: Union[np.ndarray, Literal[None]] = None
        genotype_table_c = None

        for i, (col_i_name, col_i) in [(0, ('GENOTYPES', GENOTYPES))]:
            if col_i is False:
                genotype_table_c = None
                genotype_table_py = None
            elif col_i is True:
                genotype_table_c = c_int8_2d_array_type() # initializes with int values = 0
                genotype_table_py = np.ctypeslib.as_array(genotype_table_c) # type: ignore
            else: # col_i is np.ndarray
                if tuple(col_i.shape) != (lines_n, self.total_haploids_n):
                    raise ValueError(f"passed array-to-refill for {col_i_name} is of wrong shape: {col_i.shape}. Expected shape: (lines_n, self.total_haploids_n) == {(lines_n, self.total_haploids_n)}")
                if col_i.dtype.type != data_np_type:
                    raise ValueError(f"passed array-to-refill for {col_i_name} is of wrong dtype: {col_i.dtype.type}. Expected type: {data_np_type}")
                genotype_table_c = np.ctypeslib.as_ctypes(col_i) # type: ignore
                genotype_table_py = col_i

        readcolumns_argumnets = (self.obj, lines_n, 
            *metadata_cols_c, # expands to 9 values
            genotype_table_c,
            self.total_haploids_n,
        )

        table_row_i: int

        if lines_n > 0:
            readcolumns_wrapper = self.lib.readcolumns
            readcolumns_wrapper.restype = c_int
            readcolumns_wrapper.argtypes = [
                c_void_p,
                c_longlong,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
                c_void_p,
                c_longlong,
            ]
            table_row_i = readcolumns_wrapper(self.obj, lines_n,
                *metadata_cols_c, # expands to 9 values
                genotype_table_c,
                self.total_haploids_n,
            )
            # print(f"py: finished lib.readsites() call (table_row_i = {table_row_i})")
        else:
            table_row_i = self.table_row_i

        lines_read_n = table_row_i - self.table_row_i
        self.table_row_i = table_row_i

        return lines_read_n, [col for col in metadata_cols_py if not col is False] + [gntp_tbl for gntp_tbl in [genotype_table_py] if not gntp_tbl is None]



    def read_all_lines(self,
        CHROM:     bool = False,
        POS:       bool = False,
        ID:        bool = False,
        REF:       bool = False,
        ALT:       bool = False,
        QUAL:      bool = False,
        FILTER:    bool = False,
        INFO:      bool = False,
        FORMAT:    bool = False,
        GENOTYPES: bool = False, # a 2d array, cuz may be several columns
        batch_size_lines: int = 10_000,
    ):
        """
        Reads specific columns (or genotypes table) from all the lines in the vcf table.

        `batch_size_lines` - the number of lines to read at a time. Doesn't affect the returned value.

        Returns the list of arrays as per requested columns (inferred from the arguments)

        Specify any subset of possible columns (e.g. GENOTYPES=True, ID=True, QUAL=True),
            and the function will return those arrays,
        The order in which the columns arrays appear in the returned list is
            in the order as the arguments defined here (e.g. return ID, QUAL, GENOTYPES)
        """

        if batch_size_lines <= 0:
            raise ValueError(f"batch_size_lines denotes the number of lines to read at a time, and therefore has to be a positive integer")

        cols_input: List[bool] = [CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT, GENOTYPES]
        cols_batches: List[List[np.ndarray]] = [list() for col in cols_input if col is True]

        n_batches = 0
        while (True):
            start = time.time()
            lines_read_n, metadata_cols = self.read_columns(
                lines_n = batch_size_lines,
                CHROM     = CHROM,
                POS       = POS,
                ID        = ID,
                REF       = REF,
                ALT       = ALT,
                QUAL      = QUAL,
                FILTER    = FILTER,
                INFO      = INFO,
                FORMAT    = FORMAT,
                GENOTYPES = GENOTYPES,
            )

            if lines_read_n == 0:
                break

            n_batches += 1

            for i, col in enumerate(metadata_cols):
                cols_batches[i].append(col[:lines_read_n])

            # print(f"{(round(time.time()-start, 3))} seconds")

        metadata_cols_full: List[np.ndarray]
        if n_batches > 1:
            metadata_cols_full = [np.concatenate(col_batches, axis=0) for col_batches in cols_batches]
        else:
            metadata_cols_full = [col_batches[0] for col_batches in cols_batches]

        return metadata_cols_full



    def readlines(self,
        lines_n: int,
        refill: Union[np.ndarray, None] = None,
    ) -> Tuple[np.ndarray,int]:
        """
        Reads the table of genotypes from the vcf.gz

        Functionally this:
            >>> GT_table, lines_read_n = vcfgz.readlines(1000, refill=GT_table)
        is the same as this:
            >>> lines_read_n, [GT_table] = vcfgz.read_columns(1000, GENOTYPES=GT_table)
        , but the first one (this function) is a little bit faster.

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


