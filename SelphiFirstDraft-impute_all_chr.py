from typing import Dict, List, Literal, Tuple
import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange
from load_data import (
    load_test_samples_names,
    load_variants_ids,
    load_and_interpolate_genetic_map,
    load_reference_panels,

    load_sample,
)

from imputation_lib import (
    BiDiPBWT,
    create_composite_ref_panel,
    calculate_haploid_frequencies,
    calculate_haploid_count_threshold,
    apply_filters_to_composite_panel_mask,
    form_haploid_ids_lists,
    run_hmm,
    run_hmm_variable_N,
)

from handling_results import (
    haploid_imputation_accuracy,
    save_imputation_results,
    genotype_imputation_accuracy,
)




# PARAMS
chrom = "chr1"
SI_data_folder = f"./new_data/SI_data/{chrom}"





samples = load_test_samples_names()

chip_id_list, full_id_list, original_indicies = load_variants_ids(
    chip_id_list_file       = f"{SI_data_folder}/chip_id_list.txt",
    full_id_list_file       = f"{SI_data_folder}/full_id_list_full.txt",
    original_indicies_file  = f"{SI_data_folder}/original_indicies_full.txt",
)
num_obs = len(chip_id_list)
chr_length = len(full_id_list)

chip_BP_positions, chip_cM_coordinates = load_and_interpolate_genetic_map(
    genetic_map_path = f"/home/nikita/s3-sd-platform-dev/efs/service/genome-files-processing/resource_files/genetic_map_plink/plink.{chrom}.GRCh38.map",
    chip_id_list = chip_id_list,
)


ref_panel_full_array_full_packed, ref_panel_full_array, ref_panel_chip_array = load_reference_panels(
    full_dataset_file = f"{SI_data_folder}/reference_panel.30x.hg38_chr1_noinfo_full_nppacked.zip",
    chip_sites_dataset_file = f"{SI_data_folder}/reference_panel.30x.hg38_{chrom}_noinfo_chip_variants_292samples_removed.zip",
    ref_samples_list_file = "./new_data/SI_data/samples.txt",
    samples_tobe_removed_list_file = "./data/beagle_data/imputed_samples.txt",
)



CHUNK_SIZE = num_obs # number of chip variants
HAPS: List[Literal[0,1]] = [0,1]


import time
start_time = time.time()

for sample in samples[12:13]:
    print(sample)

    target_full_array, target_chip_array, combined_ref_panel_chip = load_sample(
        sample, 
        chr_length,
        original_indicies,
        ref_panel_full_array_full_packed,
        ref_panel_chip_array,
    )

    full_res_: Dict[str, np.ndarray] = {}

    for hap in HAPS:

        BI, BJ = BiDiPBWT(
            combined_ref_panel_chip,
            ref_panel_chip_array,
            hap,
        )
        matches, composite_, comp_matches_hybrid = create_composite_ref_panel(
            BI,
            BJ,
            fl=13,
        )
        haps_freqs_array_norm_dict = calculate_haploid_frequencies(
            matches,
            comp_matches_hybrid,
            CHUNK_SIZE,
        )
        nc_thresh = calculate_haploid_count_threshold(
            matches,
            composite_,
            comp_matches_hybrid,
            haps_freqs_array_norm_dict,
            CHUNK_SIZE,
        )
        composite_ = apply_filters_to_composite_panel_mask(
            matches,
            composite_,
            comp_matches_hybrid,
            BI,
            BJ,

            # new filters
            haps_freqs_array_norm_dict,
            nc_thresh,

            CHUNK_SIZE,
        )
        """
        Adriano is working here
        """
        ordered_matches_test__, length_matches_normalized = form_haploid_ids_lists(
            matches,
            composite_,
        )
        resultoo_fb = run_hmm(
            original_indicies, # common
            ref_panel_full_array, # common
            num_obs, # common 
            ordered_matches_test__, # haploid-specific
            chip_cM_coordinates,  # common
            BI, BJ, # unused
            # THIS VARIABLE IS USELESS, YOU CAN REFACTOR AND REMOVE
            length_matches_normalized, # haploid-specific
            chr_length, # common
            num_hid=matches.shape[0], # haploid-specific
        )

        if resultoo_fb is None:
            raise RuntimeError(f"run_hmm returned `None` instead of a numpy array")

        print("DONE")

        print(f"haploid {hap} imputation mismatch: ", haploid_imputation_accuracy(
            resultoo_fb, target_full_array, hap
        ))

        full_res_.setdefault(sample, []).append(resultoo_fb.copy()) # type: ignore # Dict.setdefault returns a list

        if hap == 1:
            print("saving results")
            save_imputation_results(
                full_res_[sample],
                f'./method_first_draft/saved_dictionary_{str(sample)}_new_method_{chrom}.pkl',
            )

            print("Full results")
            print(f"sample {sample} imputation mismatch:", genotype_imputation_accuracy(
                full_res_[sample],
                target_full_array,
            ))


print("--- Total time: %s seconds ---" % (time.time() - start_time))

