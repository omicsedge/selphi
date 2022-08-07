import os
from functools import lru_cache
import csv
from typing import Tuple

import numpy as np
import pandas as pd
import zarr
import pickle


def vcf_to_zarr_target_full(
    vcf_file="./data/20.reference_panel.30x.hg38.HG00096.vcf.gz",
    output_name="./data/new_sample/target_full_array.zip",
    full_id_list_path="./data/new_sample/full_id_list_full.txt",
):
    """ """
    target_full_array = pd.read_csv(vcf_file, sep="\t", comment="#", header=None)
    target_full_array.drop([0, 1, 3, 4, 5, 6, 7, 8], axis=1, inplace=True)
    concat_target_full = pd.concat(
        [
            target_full_array.applymap(
                lambda x: str(x).replace("/", "|").split("|")[0]
            ),
            target_full_array.applymap(
                lambda x: str(x).replace("/", "|").split("|")[-1]
            ),
        ],
        axis=1,
    )
    concat_target_full
    concat_target_full.drop([2], axis=1, inplace=True)
    concat_target_full.columns = [0, 1]

    full_id_list = list(target_full_array[2])
    with open(full_id_list_path, "wb") as fp:  # Pickling
        pickle.dump(full_id_list, fp)

    zarr.save(output_name, concat_target_full.to_numpy().astype(np.int8))


def vcf_to_target_chip(
    full_id_list,
    vcf_file="./data/HG00096_euro_15k_unphased.vcf.gz",
    output_name="./data/new_sample/target_chip_array.zip",
    chip_id_list_path="./data/new_sample/chip_id_list_full.txt",
):
    target_chip_array = pd.read_csv(vcf_file, sep="\t", header=None, comment="#")
    target_chip_array.drop([0, 1, 3, 4, 5, 6, 7, 8], axis=1, inplace=True)
    target_chip_array = target_chip_array[target_chip_array[2].isin(full_id_list)]
    chip_id_list = list(target_chip_array[2])

    with open(chip_id_list_path, "wb") as fp:  # Pickling
        pickle.dump(chip_id_list, fp)

    target_chip_array.drop([2], 1, inplace=True)
    concat_target = pd.concat(
        [
            target_chip_array.applymap(
                lambda x: str(x).replace("/", "|").split("|")[0]
            ),
            target_chip_array.applymap(
                lambda x: str(x).replace("/", "|").split("|")[-1]
            ),
        ],
        axis=1,
    )
    zarr.save(output_name, concat_target.to_numpy().astype(np.int8))


def remove_sample_from_ref(
    indexes,
    ref_numpy_arr,
):
    """ """
    ref_numpy_arr = np.delete(ref_numpy_arr, indexes, axis=1)

    return ref_numpy_arr

@lru_cache(100)
def _load_samples_list(samples_list_path: str):
    with open(samples_list_path) as f:
        return f.read().splitlines()

def get_sample_index(
    sample_name,
    samples_txt_path="./data/samples_HG00096_REMOVED.txt",
) -> Tuple[int, int]:
    """
    kukrefactored
    Returns a tuple of two values:
     - the index of the given sample name from the samples list of size `n` from a given file,
     - the same index + n
    """
    samples_list = _load_samples_list(os.path.abspath(samples_txt_path))
    try:
        index = samples_list.index(sample_name)
    except ValueError as e:
        raise ValueError(f'given sample name {sample_name} is missing from the samples list at "{samples_txt_path}"') from e
    return (index, len(samples_list) + index)

def remove_all_samples(
    ref_numpy_arr,
    ref_samples="./data/SI_data/samples.txt",
    samples_tobe_removed="./data/beagle_data/imputed_samples.txt"
) -> np.ndarray:
    indexes = []
    with open(samples_tobe_removed, mode="r") as f:
        lines = f.readlines()
        samples = [line.strip() for line in lines]
    
    for sample in samples:
        indexes.extend(get_sample_index(sample,ref_samples))
    ref_numpy_arr = np.delete(ref_numpy_arr, indexes, axis=1)
    return ref_numpy_arr
