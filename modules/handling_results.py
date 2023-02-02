from typing import Dict, List, Literal

import numpy as np
import pickle

from modules.reusable_utils import (
    forced_open,
)




def save_imputation_results(
    full_res_sample: np.ndarray,
    filepath: str,
):
    with forced_open(filepath, 'wb') as f:
        pickle.dump(full_res_sample, f)
    # with open(f'./method_first_draft/saved_dictionary_{str(sample)}_new_method_{chrom}.pkl', 'wb') as f:
    #     pickle.dump(full_res__NEW[sample], f)


def save_imputed_in_stripped_vcf(
    sample_name: str,
    full_res_sample: np.ndarray,
    filepath: str,
):
    assert len(full_res_sample) == 2, "array shape has to be a matrix with the two haploids"
    full_res_sample = np.array(full_res_sample).T
    with forced_open(filepath, 'wb') as f:
        pass
    np.savetxt(filepath, full_res_sample, delimiter="|", fmt="%d", header=sample_name, comments='')


def haploid_imputation_accuracy(
    resultoo_fb: np.ndarray,
    target_full_array: np.ndarray,
    hap: Literal[0, 1],
):
    """Returns number of mismatches of an imputed haploid with the true full haploid"""
    new_x = target_full_array
    y = resultoo_fb
    return int(y.shape - np.sum(new_x == y))


def genotype_imputation_accuracy(
    full_res_sample: np.ndarray,
    final_y: np.ndarray,
):
    """
    Returns number of individual mismatching unphased genotypes of an imputed sample
        with the true sample
    """
    arr__1 = full_res_sample[0].astype(np.int8)
    arr__2 = full_res_sample[1].astype(np.int8)
    final_arr = arr__1 + arr__2

    return int(final_arr.shape - np.sum(final_y == final_arr))
