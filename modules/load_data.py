from functools import lru_cache
from typing import Dict, List, Literal, OrderedDict as OrderedDictType
import os
from collections import OrderedDict


import pandas as pd
import numpy as np
import zarr
import pickle


from modules.data_utils import get_sample_index, remove_all_samples, load_samples_list
from modules.devutils import pickle_load, var_load
from modules.vcfgz_reader import vcfgz_reader






######################################
#                                    #
#            GLOBAL DATA             #
#                                    #
######################################


def get_chipsites_refpan_path(chipsites_input_path: str, full_refpanel_path: str):
    """
    Gets the path to the reference panel dataset only with sites from the input vcf.gz dataset
    """
    input_variants_hash_file=f"{chipsites_input_path}.chip-variants.tsv.md5"
    input_variants_hash=""

    with open(input_variants_hash_file) as f:
        input_variants_hash = f.readline().strip('\n')

    return f"{full_refpanel_path}.chip.{input_variants_hash}.vcf.gz"



def load_sequences_metadata(
    chip_sites_indicies_path: str,
    chip_BPs_path: str,
    fullseq_sites_n_path: str,
    chip_sites_n_path: str,
):
    """
    `chip_sites_indicies_path` file contains the list of indicies of the input sample variants,
      as they appear in the indicies of the reference panel variants

    `chip_BPs_path` contains the list of BP positions of the variants in the input sample

    `fullseq_sites_n_path` - the number of the variants in the reference panel

    `chip_sites_n_path` - the number of the variants in the chip sites
    """

    chip_sites_indicies: List[int] = pickle_load(chip_sites_indicies_path)
    chip_BPs: List[int] = pickle_load(chip_BPs_path)
    fullseq_sites_n: int = pickle_load(fullseq_sites_n_path)
    chip_sites_n: int = pickle_load(chip_sites_n_path)

    return chip_sites_indicies, chip_BPs, fullseq_sites_n, chip_sites_n


def load_and_interpolate_genetic_map(
    genetic_map_path: str,
    chip_BPs: List[int],
):
    """
    Loads the genetic map, linearly interpolates the cM coordinates for chip BP positions.

    Takes:
     - `genetic_map_path` - path to the genetic map in the "PLINK-format"
     - `chip_BPs` - list of BP positions (e.g. `792461`)

    Returns two lists:
        - `chip_BP_positions` - chip (input) BP positions
        - `chip_cM_coordinates` - corresponding cM coordinates for the chip positions, inferred from the genetic map
    """

    genetic_map = pd.read_csv(genetic_map_path, sep=" ", comment="#",header=None,usecols=[0,2,3])
    genetic_map.columns = ['chr','cM','pos'] # type: ignore
    genetic_map.set_index('pos',inplace=True)
    genetic_map = genetic_map.join(pd.DataFrame(chip_BPs).set_index(0),how='outer')
    genetic_map['cM'] = genetic_map['cM'].interpolate('index').fillna(method='bfill')
    genetic_map['chr'] = genetic_map.chr.fillna(20.0).astype(int)
    genetic_map = genetic_map.reset_index().rename({'index':'pos'},axis=1)
    genetic_map = genetic_map[genetic_map['pos'].isin(chip_BPs)].reset_index(drop=True)
    chip_cM_coordinates = genetic_map.cM # type: ignore
    chip_cM_coordinates: List[float] = list(chip_cM_coordinates)

    assert len(chip_cM_coordinates) == len(chip_BPs)
    return chip_cM_coordinates


def load_reference_panels(
    full_dataset_file: str,
    chip_sites_dataset_file: str,
    ref_samples_list_file: str,
    samples_tobe_removed_list_file: str,
):
    """
    Loads/creates 3 datasets:
     - the full packed original full-sequenced dataset (includes true test samples)
     - the same dataset, but without test samples (i.e. full-sequenced reference panel)
     - the reference panel without test samples (i.e. reference panel with chip sites only)
    """
    #1
    ref_panel_full_array_full_packed: np.ndarray = zarr.load(full_dataset_file) # type: ignore # enforce the type
    #2
    ref_panel_full_array = remove_all_samples(ref_panel_full_array_full_packed, ref_samples_list_file, samples_tobe_removed_list_file)
    #3
    ref_panel_chip_array: np.ndarray = zarr.load(chip_sites_dataset_file) # type: ignore # enforce the type

    assert ref_panel_full_array.shape[0] == ref_panel_full_array_full_packed.shape[0] # number of sites
    assert ref_panel_full_array.shape[1] <= ref_panel_full_array_full_packed.shape[1] # number of samples
    return ref_panel_full_array_full_packed, ref_panel_full_array, ref_panel_chip_array


def load_chip_reference_panel_combined(
    chip_sites_refpanel_file: str,
    chip_sites_n: int,
    input_samples: np.ndarray,
    sample_i: int,
    sample_name: str,
):
    """
    Loads and creates a chip-sites "combined" reference panel:
     - the reference panel without test samples, but with 2 additional haploids from the input sample
    """
    chip_sites_refpanel = vcfgz_reader(chip_sites_refpanel_file)
    _, [chip_sites_refpanel_array] = chip_sites_refpanel.read_columns(lines_n=chip_sites_n, GENOTYPES=True)

    combined_ref_panel_chip: np.ndarray = np.concatenate(
        (chip_sites_refpanel_array, input_samples[:, sample_i*2 : sample_i*2 + 2]),
        axis=1,
    )

    return combined_ref_panel_chip


