from modules.devutils import forced_open, pickle_dump, pickle_load, nptype
# from modules.devutils import var_load # separate line so i can comment/uncomment var_load lines conveniently
# from modules.devutils import var_dump # separate line so i can comment/uncomment var_dump lines conveniently
from modules.kuklog import open_kuklog, kuklog_timestamp, close_kuklog
KUKLOG = open_kuklog("checkpoints.tsv")

from typing import Dict, List, Literal, Tuple, Union, OrderedDict as OrderedDict_type
from collections import OrderedDict

import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange

from modules.utils import vcf_haploid_order_to_internal

from modules.load_data import (
    load_test_samples_indices,
    load_variants_ids,
    load_and_interpolate_genetic_map,
    load_chip_reference_panel,
    load_chip_reference_panel_combined,
)

from modules.imputation_lib import (
    BiDiPBWT,
    create_composite_ref_panel,
    create_composite_ref_panel_np,
    run_BiDiPBWT_and_cr8_composite_refpanel,
    calculate_haploid_frequencies,
    calculate_haploid_count_threshold,
    apply_filters_to_composite_panel_mask,
    form_haploid_ids_lists,
    run_hmm,
)

from modules.handling_results import (
    haploid_imputation_accuracy,
    save_imputation_results,
    genotype_imputation_accuracy,
)






# PARAMS
chrom = "chr20"
SI_data_folder = f"./data/SI_data/{chrom}"

full_full_ref_panel_vcfgz_path = f'./data/full_ref_panel/{chrom}/reference_panel.30x.hg38_{chrom}_noinfo.vcf.gz'

kuklog_timestamp("loaded libraries", KUKLOG)
print("loaded libraries")


test_samples_index, refpan_samples_index = load_test_samples_indices(
    test_samples_names_path = "./data/beagle_data/imputed_samples.txt",
    all_samples_names_path = "./data/SI_data/samples.txt",
)
samples = list(test_samples_index.keys())
refpan_samples_indices = list(refpan_samples_index.values())
refpan_haploid_indices = []
for i in refpan_samples_indices:
    refpan_haploid_indices.extend([2*i, 2*i + 1])

chip_id_list, full_id_list, original_indicies = load_variants_ids(
    chip_id_list_file       = f"{SI_data_folder}/chip_id_list.txt",
    full_id_list_file       = f"{SI_data_folder}/full_id_list_full.txt",
    original_indicies_file  = f"{SI_data_folder}/original_indicies_full.txt",
)
kuklog_timestamp("loaded datasets_metadata", KUKLOG)
print("loaded datasets_metadata")
num_obs = len(chip_id_list) # number of chip sites
chr_length = len(full_id_list)

chip_BP_positions, chip_cM_coordinates = load_and_interpolate_genetic_map(
    genetic_map_path = f"/home/nikita/s3-sd-platform-dev/efs/service/genome-files-processing/resource_files/genetic_map_plink/plink.{chrom}.GRCh38.map",
    chip_id_list = chip_id_list,
)
kuklog_timestamp("loaded and interpolated genetic map", KUKLOG)
print("loaded and interpolated genetic map")

num_hid = len(refpan_haploid_indices) # number of reference panel haploids
internal_haploid_order = [vcf_haploid_order_to_internal(i, int(num_hid/2)) for i in range(num_hid)]
# full_ref_panel_chip_array, combined_ref_panel_chip = load_chip_reference_panel(
#     chip_sites_dataset_file = f"./data/full_ref_panel/{chrom}/reference_panel.30x.hg38_{chrom}.chip-sites.zip",
#     refpan_haploid_indices = refpan_haploid_indices,
# )
# kuklog_timestamp("loaded reference panels", KUKLOG)
# print("loaded reference panels")


import sys
test_sample_from = int(sys.argv[1])
test_sample_to   = int(sys.argv[2])
ARG_ERROR_MSG = f"you should pass 2 integers that will serve as slice of the `samples` list.\n"+\
f" They have to be between 0 and {len(samples)} in strictly ascending order"
assert 0 <= test_sample_from < test_sample_to <= len(samples), ARG_ERROR_MSG


mismatch_log_f = forced_open(f"mismatch-{chrom}_vcforder_samples_{test_sample_from}-{test_sample_to}.tsv", 'w')
mismatch_log_f.write('\t'.join(['sample#', 'sample_name', 'mismatch_hap0', 'mismatch_hap1', 'mismatch_gntp', 'accuracy_gntp_perc']) + '\n')
mismatch_log_f.flush()




CHUNK_SIZE = num_obs # number of chip variants
HAPS: List[Literal[0,1]] = [0,1]


import time
start_time = time.time()

last_sample = samples[test_sample_to-1]
for the_sample in samples[test_sample_from:test_sample_to]:

    ordered_matches_list: List[Dict[int, List[np.int64]]] = []
    imputing_samples: OrderedDict_type[str, OrderedDict_type[Literal[0,1], Literal[True]]] = OrderedDict()

    full_res_: Dict[str, np.ndarray] = {}


    # for sample in samples[test_sample_from:test_sample_to]:
    for sample in [the_sample]:
        sample_global_index = test_samples_index[sample]
        imputing_samples[sample] = OrderedDict()

        start_sample_time = time.time()
        print(f"{sample} (#{sample_global_index})")

        # # set the two last haploids in the "combined" ref panel to the input sample's two haploids
        # combined_ref_panel_chip[:,-2:] = full_ref_panel_chip_array[:,2*sample_global_index : 2*sample_global_index+2]
        combined_ref_panel_chip = load_chip_reference_panel_combined(
            chip_sites_dataset_file = f"./data/full_ref_panel/{chrom}/reference_panel.30x.hg38_{chrom}.chip-sites.zip",
            refpan_haploid_indices = refpan_haploid_indices,
            sample_global_index=sample_global_index,
        )
        kuklog_timestamp("loaded reference panels", KUKLOG)
        print("loaded reference panels")

        # # var_dump(f"opts01-05_dev1.1/{sample}/00_target_full_array.pkl", target_full_array)
        # # var_dump(f"opts01-05_dev1.1/{sample}/00_target_chip_array.pkl", target_chip_array)
        # # var_dump(f"opts01-05_dev1.3/{sample}/00_combined_ref_panel_chip.pkl", combined_ref_panel_chip)
        # # combined_ref_panel_chip_true = var_load(f"opts01-04_dev1/{sample}/00_combined_ref_panel_chip.pkl")
        # # print(f"{(combined_ref_panel_chip == combined_ref_panel_chip_true).all()=}")
        # # print(f"{np.where(combined_ref_panel_chip != combined_ref_panel_chip_true)[0].shape=}")
        # # exit(0)
        kuklog_timestamp(f"loaded sample {sample} and formed the combined ref panel", KUKLOG)
        print(f"loaded sample {sample} and formed the combined ref panel")

        for hap in HAPS:
            imputing_samples[sample][hap] = True

            kuklog_timestamp_func = lambda s: kuklog_timestamp(f"hap{hap}: BiDiPBWT: {s}", KUKLOG)
            BI, BJ = BiDiPBWT(
                combined_ref_panel_chip, #type:ignore # it's not unbound since we del it only after the last iteration
                num_hid,
                hap,
                kuklog_timestamp_func,
            )
            if hap == HAPS[-1]:
                del combined_ref_panel_chip #type:ignore # it's not unbound since we del it only after the last iteration
            # var_dump(f"opts01-05_dev1.3/{sample}/{hap}/01_BI.pkl", BI)
            # var_dump(f"opts01-05_dev1.3/{sample}/{hap}/01_BJ.pkl", BJ)
            # BI_true = var_load(f"opts01-04_dev1/{sample}/{hap}/01_BI.pkl")
            # BJ_true = var_load(f"opts01-04_dev1/{sample}/{hap}/01_BJ.pkl")
            # print(f"{(BI == BI_true).all()=}")
            # print(f"{np.where(BI != BI_true)[0].shape=}")
            # print(f"{(BJ == BJ_true).all()=}")
            # print(f"{np.where(BJ != BJ_true)[0].shape=}")
            # exit(0)
            kuklog_timestamp(f"hap{hap}: ran BiDiPBWT", KUKLOG) #1
            matches, composite_, comp_matches_hybrid = create_composite_ref_panel_np(
                BI,
                BJ,
                fl=13,
            )
            # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/02_matches.pkl", matches)
            # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/02_composite_.pkl", composite_)
            # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/02_comp_matches_hybrid.pkl", comp_matches_hybrid)
            # matches = var_load(f"opts01-04_dev1/{sample}/{hap}/02_matches.pkl")
            # composite_ = var_load(f"opts01-04_dev1/{sample}/{hap}/02_composite_.pkl")
            # comp_matches_hybrid = var_load(f"opts01-04_dev1/{sample}/{hap}/02_comp_matches_hybrid.pkl")

            kuklog_timestamp(f"hap{hap}: created first composite ref panel", KUKLOG) #2
            haps_freqs_array_norm_dict = calculate_haploid_frequencies(
                matches, # X
                comp_matches_hybrid,
                CHUNK_SIZE,
            )
            # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/03_haps_freqs_array_norm_dict.pkl", haps_freqs_array_norm_dict)
            # haps_freqs_array_norm_dict = var_load(f"opts01-04_dev1/{sample}/{hap}/03_haps_freqs_array_norm_dict.pkl")
            kuklog_timestamp(f"hap{hap}: calculated 1st filter: haploid frequencies", KUKLOG) #3
            nc_thresh = calculate_haploid_count_threshold(
                matches, # likely X cuz "hybrid" is "comp_matches_hybrid"
                composite_, # X
                comp_matches_hybrid,
                haps_freqs_array_norm_dict,
                CHUNK_SIZE,
            )
            del comp_matches_hybrid
            del composite_
            # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/04_nc_thresh.pkl", nc_thresh)
            # # nc_thresh = var_load(f"opts01-04_dev1/{sample}/{hap}/04_nc_thresh.pkl")
            kuklog_timestamp(f"hap{hap}: calculated 2nd filter: haploid count threshold", KUKLOG) #4
            composite_ = apply_filters_to_composite_panel_mask(
                matches,
                # composite_, # X
                # comp_matches_hybrid, # X
                BI,
                BJ,

                # new filters
                haps_freqs_array_norm_dict,
                nc_thresh,

                CHUNK_SIZE,
            )
            # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/05_composite_.pkl", composite_)
            del BI
            del BJ
            del haps_freqs_array_norm_dict
            del nc_thresh
            # # composite_ = var_load(f"opts01-04_dev1/{sample}/{hap}/05_composite_.pkl")
            kuklog_timestamp(f"hap{hap}: applied the 2 filters", KUKLOG) #5
            """
            Adriano is working here
            """
            ordered_matches_test__ = form_haploid_ids_lists(
                matches,
                composite_,
            )
            # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/06_ordered_matches_test__.pkl", ordered_matches_test__)
            del matches
            del composite_
            # # # ordered_matches_test__ = var_load(f"opts01-04_dev1/{sample}/{hap}/06_ordered_matches_test__.pkl")
            # ordered_matches_test___true = var_load(f"opts01-04_dev1/{sample}/{hap}/06_ordered_matches_test__.pkl")
            # print(f"{all([(ordered_matches_test__[i] == ordered_matches_test___true[i]) for i in range(num_obs)])=}")
            # print(f"{sum([(ordered_matches_test__[i] != ordered_matches_test___true[i]) for i in range(num_obs)])=}")
            # # print(f"{np.where(ordered_matches_test__ != ordered_matches_test___true)[0].shape=}")
            # # exit(0)
            # del ordered_matches_test___true
            kuklog_timestamp(f"hap{hap}: form haploid ids list", KUKLOG)

            ordered_matches_list.append(ordered_matches_test__)
            del ordered_matches_test__


        # # del combined_ref_panel_chip
        print(f"--- {sample} time: {(time.time() - start_sample_time)} seconds ---")

    # # var_dump(f"stage2/06_ordered_matches_list_1sample_12.pkl", ordered_matches_list)
    # ordered_matches_list = var_load(f"stage2/06_ordered_matches_list_1sample_12.pkl")
    # # del full_ref_panel_chip_array
    # # ordered_matches_list = var_load(f"opts01-05_dev1.3.1/06_ordered_matches_list_1sample_12.pkl")
    # ordered_matches_list = var_load(f"opts01-05_dev1.3.1/06_ordered_matches_list_10samples_12-21.pkl")
    # # ordered_matches_list = [ordered_matches_list[0]]
    # # exit(0)


    imputing_samples_haploids_indices: List[int] = []
    for sample, haps in imputing_samples.items():
        for hap in haps.keys():
            imputing_samples_haploids_indices.append(2*test_samples_index[sample] + hap)



    kuklog_timestamp_func = lambda s: kuklog_timestamp(f"f-b: {s}", KUKLOG)
    resultsoo_fb, target_full_array = run_hmm(
        original_indicies, # common
        full_full_ref_panel_vcfgz_path, # common
        num_obs, # common 
        ordered_matches_list, # haploid-specific
        chip_cM_coordinates,  # common
        np.array(None), np.array(None), # unused
        chr_length, # common
        num_hid=num_hid, # common
        imputing_samples_haploids_indices = imputing_samples_haploids_indices,
        internal_haploid_order = internal_haploid_order,
        reference_haploids_indices = refpan_haploid_indices,
        kuklog_timestamp_func=kuklog_timestamp_func,
    )

    # var_dump(f"opts01-04_dev1/{sample}/{hap}/07_resultoo_fb.pkl", resultoo_fb)
    # resultoo_fb_true = var_load(f"opts01-04_dev1/{sample}/{hap}/07_resultoo_fb.pkl")
    kuklog_timestamp(f"ran forward backward and imputation", KUKLOG)
    # print(f"{(resultoo_fb == resultoo_fb_true).all()=}")
    # print(f"{np.where(resultoo_fb != resultoo_fb_true)=}")
    # if not (resultoo_fb == resultoo_fb_true).all():
    #     raise RuntimeError(f"`run_hmm` result is different from original, check what's wrong there")
    # var_dump(f"opts01-04_dev1/{sample}/{hap}/07_post.pkl", post)
    # smpls2take = post.astype(bool).sum(axis=1)
    # var_dump(f"opts01-04_dev1/{sample}/{hap}/07_smpls2take.pkl", smpls2take)

    if resultsoo_fb is None or len(resultsoo_fb) == 0:
        raise RuntimeError(f"run_hmm returned `None` instead of a numpy array")

    print("DONE")

    i = 0
    for sample, haps in imputing_samples.items():
        true_target_full_sample: List[np.ndarray] = []
        mismatch = {
            0: 0,
            1: 0,
            'genotype': 0,
        }
        for hap in haps.keys():
            resultoo_fb = resultsoo_fb[i]
            true_target_full: np.ndarray = target_full_array[:,i]
            true_target_full_sample.append(true_target_full)

            # print(f"true_target_full: "
            # f"[{true_target_full[0]}, {true_target_full[1]}, {true_target_full[2]}, ..., {true_target_full[-1]}] "
            # f"shape={true_target_full.shape}, dtype={true_target_full.dtype} sum={true_target_full.sum()}")
            # print(f"resultoo_fb:      "
            # f"[{resultoo_fb[0]}, {resultoo_fb[1]}, {resultoo_fb[2]}, ..., {resultoo_fb[-1]}] "
            # f"shape={resultoo_fb.shape}, dtype={resultoo_fb.dtype} sum={resultoo_fb.sum()}")

            # resultoo_fb_true = var_load(f"opts01-04_dev1/{sample}/{hap}/07_resultoo_fb.pkl")
            # diff = np.where(resultoo_fb_true != resultoo_fb)[0]
            # # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/07_resultoo_fb.pkl", resultoo_fb)
            # # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/07_res_diff.pkl", diff)
            # print(f"{diff=} ({diff.shape})")

            # print(f"{(resultoo_fb_true == resultoo_fb).all()=}")


            mismatch[hap] = haploid_imputation_accuracy(resultoo_fb, true_target_full, hap)
            print(f"haploid {hap} imputation mismatch: ", mismatch[hap])

            full_res_.setdefault(sample, []).append(resultoo_fb) # type: ignore # Dict.setdefault returns a list

            kuklog_timestamp(f"hap{hap}: calc'ed accuracy", KUKLOG)
            if hap == 1:
                print("saving results")
                save_imputation_results(
                    full_res_[sample],
                    f'./results/pickled_selphi_results/saved_dictionary_{str(sample)}_new_method_{chrom}.pkl',
                )

                print("Full results")
                mismatch['genotype'] = genotype_imputation_accuracy(
                    full_res_[sample],
                    true_target_full_sample[0] + true_target_full_sample[1],
                )
                print(f"sample {sample} imputation mismatch:", mismatch['genotype'])

                mismatch_log_f.write('\t'.join([
                    str(samples.index(sample)),
                    sample,
                    str(mismatch[0]),
                    str(mismatch[1]),
                    str(mismatch['genotype']),
                    str(round(100 - (mismatch['genotype']/len(true_target_full_sample[0]))*100, 6)),
                ]) + '\n')
                mismatch_log_f.flush()

            i += 1


    del resultsoo_fb
    del target_full_array

    kuklog_timestamp(f"saved the results", KUKLOG)




print("===== Total time: %s seconds =====" % (time.time() - start_time))

mismatch_log_f.close()

close_kuklog(KUKLOG)
