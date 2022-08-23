from modules.devutils import nptype, pickle_dump, var_dump, var_load
from modules.kuklog import open_kuklog, kuklog_timestamp, close_kuklog
KUKLOG = open_kuklog("checkpoints.tsv")

from typing import Dict, List, Literal, Tuple
import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange
from modules.load_data import (
    load_test_samples_names,
    load_variants_ids,
    load_and_interpolate_genetic_map,
    load_reference_panels,
    load_chip_reference_panel,

    load_sample,

    load_full_reference_panels,
)

from modules.imputation_lib import (
    BiDiPBWT,
    create_composite_ref_panel,
    calculate_haploid_frequencies,
    calculate_haploid_count_threshold,
    apply_filters_to_composite_panel_mask,
    form_haploid_ids_lists,
    run_hmm,
    run_hmm_variable_N,
)

from modules.handling_results import (
    haploid_imputation_accuracy,
    save_imputation_results,
    genotype_imputation_accuracy,
)



# PARAMS
chrom = "chr20"
SI_data_folder = f"./data/SI_data/{chrom}"

kuklog_timestamp("loaded libraries", KUKLOG)


samples = load_test_samples_names()
chip_id_list, full_id_list, original_indicies = load_variants_ids(
    chip_id_list_file       = f"{SI_data_folder}/chip_id_list.txt",
    full_id_list_file       = f"{SI_data_folder}/full_id_list_full.txt",
    original_indicies_file  = f"{SI_data_folder}/original_indicies_full.txt",
)
kuklog_timestamp("loaded datasets_metadata", KUKLOG)
num_obs = len(chip_id_list)
chr_length = len(full_id_list)

chip_BP_positions, chip_cM_coordinates = load_and_interpolate_genetic_map(
    genetic_map_path = f"/home/nikita/s3-sd-platform-dev/efs/service/genome-files-processing/resource_files/genetic_map_plink/plink.{chrom}.GRCh38.map",
    chip_id_list = chip_id_list,
)
kuklog_timestamp("loaded and interpolated genetic map", KUKLOG)

ref_panel_full_array_full_packed, ref_panel_full_array, ref_panel_chip_array = load_reference_panels(
    full_dataset_file = f"{SI_data_folder}/reference_panel.30x.hg38_{chrom}_noinfo_full_nppacked.zip",
    chip_sites_dataset_file = f"{SI_data_folder}/reference_panel.30x.hg38_{chrom}_noinfo_chip_variants_292samples_removed.zip",
    ref_samples_list_file = "./data/SI_data/samples.txt",
    samples_tobe_removed_list_file = "./data/beagle_data/imputed_samples.txt",
)
kuklog_timestamp("loaded reference panels", KUKLOG)


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
    # var_dump(f"{sample}/00_target_full_array.pkl", target_full_array)
    # var_dump(f"{sample}/00_target_chip_array.pkl", target_chip_array)
    # var_dump(f"{sample}/00_combined_ref_panel_chip.pkl", combined_ref_panel_chip)
    kuklog_timestamp("loaded sample", KUKLOG)

    full_res_: Dict[str, np.ndarray] = {}

    for hap in HAPS:

        kuklog_timestamp_func = lambda s: kuklog_timestamp(f"hap{hap}: BiDiPBWT: {s}", KUKLOG)
        BI, BJ = BiDiPBWT(
            combined_ref_panel_chip,
            ref_panel_chip_array,
            hap,
            kuklog_timestamp_func,
        )
        # var_dump(f"{sample}/{hap}/01_BI.pkl", BI)
        # var_dump(f"{sample}/{hap}/01_BJ.pkl", BJ)
        kuklog_timestamp(f"hap{hap}: ran BiDiPBWT", KUKLOG)
        matches, composite_, comp_matches_hybrid = create_composite_ref_panel(
            BI,
            BJ,
            fl=13,
        )
        # var_dump(f"{sample}/{hap}/02_matches.pkl", matches)
        # var_dump(f"{sample}/{hap}/02_composite_.pkl", composite_)
        # var_dump(f"{sample}/{hap}/02_comp_matches_hybrid.pkl", comp_matches_hybrid)
        kuklog_timestamp(f"hap{hap}: created first composite ref panel", KUKLOG)
        haps_freqs_array_norm_dict = calculate_haploid_frequencies(
            matches,
            comp_matches_hybrid,
            CHUNK_SIZE,
        )
        # var_dump(f"{sample}/{hap}/03_haps_freqs_array_norm_dict.pkl", haps_freqs_array_norm_dict)
        kuklog_timestamp(f"hap{hap}: calculated 1st filter: haploid frequencies", KUKLOG)
        nc_thresh = calculate_haploid_count_threshold(
            matches,
            composite_,
            comp_matches_hybrid,
            haps_freqs_array_norm_dict,
            CHUNK_SIZE,
        )
        # var_dump(f"{sample}/{hap}/04_nc_thresh.pkl", nc_thresh)
        kuklog_timestamp(f"hap{hap}: calculated 2nd filter: haploid count threshold", KUKLOG)
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
        # var_dump(f"{sample}/{hap}/05_composite_.pkl", composite_)
        kuklog_timestamp(f"hap{hap}: applied the 2 filters", KUKLOG)
        """
        Adriano is working here
        """
        ordered_matches_test__ = form_haploid_ids_lists(
            matches,
            composite_,
        )
        # var_dump(f"{sample}/{hap}/06_ordered_matches_test__.pkl", ordered_matches_test__)
        kuklog_timestamp(f"hap{hap}: form haploid ids list", KUKLOG)

        kuklog_timestamp_func = lambda s: kuklog_timestamp(f"hap{hap}: f-b & imputation: {s}", KUKLOG)
        resultoo_fb = run_hmm(
            original_indicies, # common
            ref_panel_full_array, # common
            num_obs, # common 
            ordered_matches_test__, # haploid-specific
            chip_cM_coordinates,  # common
            BI, BJ, # unused
            chr_length, # common
            num_hid=matches.shape[0], # haploid-specific
            kuklog_timestamp_func=kuklog_timestamp_func,
        )
        # var_dump(f"opts01-04/{sample}/{hap}/07_resultoo_fb.pkl", resultoo_fb)
        kuklog_timestamp(f"hap{hap}: ran forward backward and imputation", KUKLOG)

        if resultoo_fb is None:
            raise RuntimeError(f"run_hmm returned `None` instead of a numpy array")

        print("DONE")

        print(f"haploid {hap} imputation mismatch: ", haploid_imputation_accuracy(
            resultoo_fb, target_full_array, hap
        ))

        full_res_.setdefault(sample, []).append(resultoo_fb.copy()) # type: ignore # Dict.setdefault returns a list

        kuklog_timestamp(f"hap{hap}: calc'ed accuracy", KUKLOG)
        if hap == 1:
            print("saving results")
            save_imputation_results(
                full_res_[sample],
                f'./results/pickled_selphi_results/saved_dictionary_{str(sample)}_new_method_{chrom}.pkl',
            )

            print("Full results")
            print(f"sample {sample} imputation mismatch:", genotype_imputation_accuracy(
                full_res_[sample],
                target_full_array,
            ))
            kuklog_timestamp(f"saved the results", KUKLOG)



print("--- Total time: %s seconds ---" % (time.time() - start_time))

close_kuklog(KUKLOG)
