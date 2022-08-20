from typing import List, Literal
import itertools

import pickle
import numpy as np

from modules.devutils import pickle_load


# HAPS: List[Literal[0,1]] = [0,1]

# class hap:
#     def __init__(self, BI: np.ndarray, BJ: np.ndarray) -> None:
#         self.BI = BI
#         self.BJ = BJ


# class test:
#     def __init__(self, hap0: hap, hap1: hap) -> None:
#         self.hap0 = hap0
#         self.hap1 = hap1



base = { 
    '0': {
        'BI': pickle_load('intermediate_variables_dump/HG02330/0/01_BI.pkl'),
        'BJ': pickle_load('intermediate_variables_dump/HG02330/0/01_BJ.pkl'),
    },
    '1': {
        'BI': pickle_load('intermediate_variables_dump/HG02330/1/01_BI.pkl'),
        'BJ': pickle_load('intermediate_variables_dump/HG02330/1/01_BJ.pkl'),
    }
}


opt01 = { 
    '0': {
        'BI': pickle_load('intermediate_variables_dump/opt01/HG02330/0/01_BI.pkl'),
        'BJ': pickle_load('intermediate_variables_dump/opt01/HG02330/0/01_BJ.pkl'),
    },
    '1': {
        'BI': pickle_load('intermediate_variables_dump/opt01/HG02330/1/01_BI.pkl'),
        'BJ': pickle_load('intermediate_variables_dump/opt01/HG02330/1/01_BJ.pkl'),
    }
}


for hap in ['0', '1']:
    for varname in ['BI', 'BJ']:
        base_var : np.ndarray = base[hap][varname]
        opt01_var: np.ndarray = opt01[hap][varname]
        print(f"base.{hap}.{varname} == opt1.{hap}.{varname} = \t{np.array_equal(base_var, opt01_var)}")

# for keys in itertools.product(*[
#     ['0','1'],
#     ['BI', 'BJ']
# ]):
#     print(f"{base[keys[0]][keys[1]]}")


# somelists = [
#     [1, 2, 3],
#     ['a', 'b'],
#     [4, 5]
# ]
# for element in itertools.product(*somelists):
#     print(element)


