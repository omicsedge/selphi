from functools import lru_cache
from typing import Dict, List, Literal
import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange

from modules.data_utils import get_sample_index, remove_all_samples
from modules.hmm_utils import setFwdValues, setBwdValues
from modules.utils import (
    interpolate_parallel,
    interpolate_parallel_packed,
    BidiBurrowsWheelerLibrary,
    get_std,
    skip_duplicates,
    forced_open
)





######################################
#                                    #
#            GLOBAL DATA             #
#                                    #
######################################



def load_test_samples_names(
    path:str = "./data/beagle_data/imputed_samples.txt"
):
    """
    
    """
    with open(path, mode="r") as f:
        lines = f.readlines()
    sample_names = skip_duplicates([line.strip() for line in lines])

    return sample_names
    print(len(samples))


def load_variants_ids(
    chip_id_list_file: str,
    full_id_list_file: str,
    original_indicies_file: str,
):
    """
    `chip_id_list` contains ids available in the input sample

    `full_id_list` contains ids available in samples from the reference panel

    all ids in `chip_id_list` are included in `full_id_list`, while `original_indicies`
    is the list of indicies of ids in the `chip_id_list` as they appear in `full_id_list`
    """
    with open(chip_id_list_file, "rb") as fp:   # Unpickling
        chip_id_list: List[str] = pickle.load(fp)
    with open(full_id_list_file, "rb") as fp:   # Unpickling
        full_id_list: List[str] = pickle.load(fp)
    with open(original_indicies_file, "rb") as fp:   # Unpickling
        original_indicies: List[int] = pickle.load(fp)

    assert len(chip_id_list) == len(original_indicies)
    for i in range(len(original_indicies)):
        oi = original_indicies[i]
        assert chip_id_list[i] == full_id_list[oi]

    return chip_id_list, full_id_list, original_indicies


def load_and_interpolate_genetic_map(
    genetic_map_path: str,
    chip_id_list: List[str],
):
    """
    Loads the genetic map, linearly interpolates the cM coordinates for chip BP positions.

    Takes:
     - `genetic_map_path` - path to the genetic map in the "PLINK-format"
     - `chip_id_list` - list of chip variant ids (e.g. `'1-792461-G-A'`)

    Returns two lists:
        - `chip_BP_positions` - chip (input) BP positions
        - `chip_cM_coordinates` - corresponding cM coordinates for the chip positions, inferred from the genetic map
    """
    chip_BP_positions = [int(x.split('-')[1]) for x in chip_id_list]

    genetic_map = pd.read_csv(genetic_map_path, sep=" ", comment="#",header=None,usecols=[0,2,3])
    genetic_map.columns = ['chr','cM','pos']
    genetic_map.set_index('pos',inplace=True)
    genetic_map = genetic_map.join(pd.DataFrame(chip_BP_positions).set_index(0),how='outer')
    genetic_map['cM'] = genetic_map['cM'].interpolate('index').fillna(method='bfill')
    genetic_map['chr'] = genetic_map.chr.fillna(20.0).astype(int)
    genetic_map = genetic_map.reset_index().rename({'index':'pos'},axis=1)
    genetic_map = genetic_map[genetic_map['pos'].isin(chip_BP_positions)].reset_index(drop=True)
    chip_cM_coordinates = genetic_map.cM
    chip_cM_coordinates: List[float] = list(chip_cM_coordinates)

    assert len(chip_cM_coordinates) == len(chip_BP_positions)
    return chip_BP_positions, chip_cM_coordinates


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


def load_chip_reference_panel(
    chip_sites_dataset_file: str,
):
    """
    Loads the reference panel without test samples (i.e. reference panel with chip sites only)
    """
    ref_panel_chip_array: np.ndarray = zarr.load(chip_sites_dataset_file) # type: ignore # enforce the type

    return ref_panel_chip_array

@lru_cache(1)
def load_full_reference_panels(
    full_dataset_file: str,
    ref_samples_list_file: str,
    samples_tobe_removed_list_file: str,
):
    """
    Loads/creates 2 datasets:
     - the full packed original full-sequenced dataset (includes true test samples)
     - the same dataset, but without test samples (i.e. full-sequenced reference panel)
    """
    #1
    ref_panel_full_array_full_packed: np.ndarray = zarr.load(full_dataset_file) # type: ignore # enforce the type
    #2
    ref_panel_full_array = remove_all_samples(ref_panel_full_array_full_packed, ref_samples_list_file, samples_tobe_removed_list_file)

    assert ref_panel_full_array.shape[0] == ref_panel_full_array_full_packed.shape[0] # number of sites
    assert ref_panel_full_array.shape[1] <= ref_panel_full_array_full_packed.shape[1] # number of samples
    return ref_panel_full_array_full_packed, ref_panel_full_array




######################################
#                                    #
#        SAMPLE-SPECIFIC DATA        #
#                                    #
######################################


def load_sample(
    sample: str,
    chr_length: int,
    original_indicies: List[int],
    ref_panel_full_array_full_packed: np.ndarray,
    ref_panel_chip_array: np.ndarray,
):
    """
    Loading two datasets of the test sample: 
     - the true full-sequenced sample,
     - the unimputed one (chip sites only)

    And creates a combined array:
     - the reference panel haploids + the test sample (2 haploids) (chip sites only)

    """
    sample_index = get_sample_index(sample, samples_txt_path=f"./data/SI_data/samples.txt")
    target_full_array = np.zeros((chr_length,2))
    target_full_array[:,0] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[0]])[:chr_length]
    target_full_array[:,1] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[1]])[:chr_length]

    target_chip_array: np.ndarray = target_full_array[original_indicies,:]
    combined_ref_panel_chip: np.ndarray = np.concatenate([ref_panel_chip_array,target_chip_array],axis=1)

    print("Concatenated shape (the reference panel haploids + the test sample (chip sites only)): ", combined_ref_panel_chip.shape)

    return target_full_array, target_chip_array, combined_ref_panel_chip





def load_sample_chipsites(
    sample: str,
    chr_length: int,
    original_indicies: List[int],
    ref_panel_full_array_full_packed: np.ndarray,
    ref_panel_chip_array: np.ndarray,
    target_full_array: np.ndarray,
):
    """


    """

    target_chip_array: np.ndarray = target_full_array[original_indicies,:]
    combined_ref_panel_chip: np.ndarray = np.concatenate([ref_panel_chip_array,target_chip_array],axis=1)

    print("Concatenated shape (the reference panel haploids + the test sample (chip sites only)): ", combined_ref_panel_chip.shape)

    return target_chip_array, combined_ref_panel_chip


def load_sample_fullsequence(
    sample: str,
    chr_length: int,
    original_indicies: List[int],
    ref_panel_full_array_full_packed: np.ndarray,
    ref_panel_chip_array: np.ndarray,
):
    """


    """
    sample_index = get_sample_index(sample, samples_txt_path=f"./data/SI_data/samples.txt")
    target_full_array = np.zeros((chr_length,2))
    target_full_array[:,0] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[0]])[:chr_length]
    target_full_array[:,1] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[1]])[:chr_length]

    return target_full_array



