from modules.devutils import forced_open, pickle_dump, pickle_load, nptype, var_load
# from modules.devutils import var_load # separate line so i can comment/uncomment var_load lines conveniently
# from modules.devutils import var_dump # separate line so i can comment/uncomment var_dump lines conveniently
from modules.kuklog import open_kuklog, kuklog_timestamp, close_kuklog
from modules.vcfgz_reader import vcfgz_reader


from typing import Dict, List, Literal, Tuple, Union, OrderedDict as OrderedDict_type
from collections import OrderedDict
# import dill as pickle
from joblib import Parallel, delayed
import sys
import subprocess
from multiprocessing import Pool
from datetime import datetime as dt


import os
import pandas as pd
import numpy as np
import zarr
from tqdm import trange


from modules.utils import vcf_haploid_order_to_internal

from modules.load_data import (
    get_chipsites_refpan_path,
    load_test_samples_indices,
    load_sequences_metadata,
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
    save_imputed_in_stripped_vcf,
    genotype_imputation_accuracy,
)

from modules.multiprocessing import mute_stdout, split_range_into_n_whole_even_subranges








def selphi(
    input_vcfgz_path: str,
    chip_refpanel_vcfgz_path: str,
    full_refpanel_vcfgz_path: str,

    genetic_map_path: str,

    output_path: str,
    cores: int,
):
    KUKLOG = open_kuklog("checkpoints.tsv")


    kuklog_timestamp("loaded libraries", KUKLOG)
    print("loaded libraries")


    chip_sites_indicies, chip_BPs, fullseq_sites_n, chip_sites_n = load_sequences_metadata(
        chip_sites_indicies_path=f'{chip_refpanel_vcfgz_path}.chip_sites_indicies.pkl',
        chip_BPs_path=f'{input_vcfgz_path}.chip_BPs.pkl',
        fullseq_sites_n_path=f'{full_refpanel_vcfgz_path}.fullseq_sites_n.pkl',
        chip_sites_n_path=f'{chip_refpanel_vcfgz_path}.chip_sites_n.pkl',
    )

    kuklog_timestamp("loaded datasets_metadata", KUKLOG)
    print("loaded datasets_metadata")



    chip_cM_coordinates = load_and_interpolate_genetic_map(
        # genetic_map_path = f"/home/nikita/s3-sd-platform-dev/efs/service/genome-files-processing/resource_files/genetic_map_plink/plink.{chrom}.GRCh38.map",
        genetic_map_path = genetic_map_path,
        chip_BPs = chip_BPs,
    )
    kuklog_timestamp("loaded and interpolated genetic map", KUKLOG)
    print("loaded and interpolated genetic map")




    # import sys
    # test_sample_from = int(sys.argv[1])
    # test_sample_to   = int(sys.argv[2])
    # ARG_ERROR_MSG = f"you should pass 2 integers that will serve as slice of the `samples` list.\n"+\
    # f" They have to be between 0 and {len(samples)} in strictly ascending order"
    # assert 0 <= test_sample_from < test_sample_to <= len(samples), ARG_ERROR_MSG


    # mismatch_log_f = forced_open(f"mismatch-{chrom}_vcforder_samples_{test_sample_from}-{test_sample_to}.tsv", 'w')
    # mismatch_log_f.write('\t'.join(['sample#', 'sample_name', 'mismatch_hap0', 'mismatch_hap1', 'mismatch_gntp', 'accuracy_gntp_perc']) + '\n')
    # mismatch_log_f.flush()



    CHUNK_SIZE = chip_sites_n # number of chip variants
    HAPS: List[Literal[0,1]] = [0,1]


    import time
    start_time = time.time()

    input_vcfgz = vcfgz_reader(input_vcfgz_path)
    lines_read_n, [input_samples] = input_vcfgz.read_columns(lines_n=chip_sites_n, GENOTYPES=True)
    assert lines_read_n == chip_sites_n

    input_samples_names = input_vcfgz.samples
    n_input_samples = len(input_vcfgz.samples)

    input_vcfgz.close()


    chip_sites_refpanel = vcfgz_reader(chip_refpanel_vcfgz_path)
    refpan_haps_n = len(chip_sites_refpanel.samples)*2
    chip_sites_refpanel.close()


    def impute_input_vcf_sample(samples: List[int]):
        imputed_sample_files: List[str] = []


        for sample_i in samples:
            sample_name = input_samples_names[sample_i]
            # sample == HG02330

            ordered_matches_list: List[Dict[int, List[np.int64]]] = []
            imputing_samples: OrderedDict_type[str, OrderedDict_type[Literal[0,1], Literal[True]]] = OrderedDict()

            full_res_: Dict[str, np.ndarray] = {}


            # for sample in samples[test_sample_from:test_sample_to]:
            for sample_i, sample_name in [(sample_i, sample_name)]:
                # sample_global_index = test_samples_index[sample]
                imputing_samples[sample_name] = OrderedDict()

                start_sample_time = time.time()
                print(f"{dt.now().strftime('%c')}\t{sample_name}")

                combined_ref_panel_chip = np.array([]) # tmp
                # # set the two last haploids in the "combined" ref panel to the input sample's two haploids
                # combined_ref_panel_chip[:,-2:] = full_ref_panel_chip_array[:,2*sample_global_index : 2*sample_global_index+2]
                combined_ref_panel_chip = load_chip_reference_panel_combined(
                    chip_sites_refpanel_file = chip_refpanel_vcfgz_path,
                    chip_sites_n = chip_sites_n,
                    input_samples = input_samples,
                    sample_i = sample_i,
                    sample_name = sample_name,
                )
                assert refpan_haps_n == combined_ref_panel_chip.shape[1] - 2 # number of reference panel haploids
                # kuklog_timestamp("loaded reference panels", KUKLOG)
                # print("loaded reference panels")

                # # var_dump(f"opts01-05_dev1.1/{sample}/00_target_full_array.pkl", target_full_array)
                # # var_dump(f"opts01-05_dev1.1/{sample}/00_target_chip_array.pkl", target_chip_array)
                # # var_dump(f"opts01-05_dev1.3/{sample}/00_combined_ref_panel_chip.pkl", combined_ref_panel_chip)
                # combined_ref_panel_chip_true = var_load(f"opts01-04_dev1/{sample_name}/00_combined_ref_panel_chip.pkl")
                # print(f"{(combined_ref_panel_chip == combined_ref_panel_chip_true).all()=}")
                # print(f"{np.where(combined_ref_panel_chip != combined_ref_panel_chip_true)[0].shape=}")
                # exit(0)
                # kuklog_timestamp(f"loaded sample {sample_name} and formed the combined ref panel", KUKLOG)
                # print(f"loaded sample {sample_name} and formed the combined ref panel")

                for hap in HAPS:
                    imputing_samples[sample_name][hap] = True

                    # kuklog_timestamp_func = lambda s: kuklog_timestamp(f"hap{hap}: BiDiPBWT: {s}", KUKLOG)
                    BI, BJ = BiDiPBWT(
                        combined_ref_panel_chip, #type:ignore # it's not unbound since we del it only after the last iteration
                        refpan_haps_n,
                        hap,
                        lambda s: None,
                    )
                    if hap == HAPS[-1]:
                        del combined_ref_panel_chip #type:ignore # it's not unbound since we del it only after the last iteration
                    # var_dump(f"opts01-05_dev1.3/{sample}/{hap}/01_BI.pkl", BI)
                    # var_dump(f"opts01-05_dev1.3/{sample}/{hap}/01_BJ.pkl", BJ)
                    # BI_true = var_load(f"opts01-04_dev1/{sample_name}/{hap}/01_BI.pkl")
                    # BJ_true = var_load(f"opts01-04_dev1/{sample_name}/{hap}/01_BJ.pkl")
                    # print(f"{(BI == BI_true).all()=}")
                    # print(f"{np.where(BI != BI_true)[0].shape=}")
                    # print(f"{(BJ == BJ_true).all()=}")
                    # print(f"{np.where(BJ != BJ_true)[0].shape=}")
                    # exit(0)
                    # kuklog_timestamp(f"hap{hap}: ran BiDiPBWT", KUKLOG) #1
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

                    # kuklog_timestamp(f"hap{hap}: created first composite ref panel", KUKLOG) #2
                    haps_freqs_array_norm_dict = calculate_haploid_frequencies(
                        matches, # X
                        comp_matches_hybrid,
                        CHUNK_SIZE,
                    )
                    # var_dump(f"opts01-05_dev1.3.1/{sample}/{hap}/03_haps_freqs_array_norm_dict.pkl", haps_freqs_array_norm_dict)
                    # haps_freqs_array_norm_dict = var_load(f"opts01-04_dev1/{sample}/{hap}/03_haps_freqs_array_norm_dict.pkl")
                    # kuklog_timestamp(f"hap{hap}: calculated 1st filter: haploid frequencies", KUKLOG) #3
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
                    # kuklog_timestamp(f"hap{hap}: calculated 2nd filter: haploid count threshold", KUKLOG) #4
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
                    # kuklog_timestamp(f"hap{hap}: applied the 2 filters", KUKLOG) #5
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
                    # kuklog_timestamp(f"hap{hap}: form haploid ids list", KUKLOG)

                    ordered_matches_list.append(ordered_matches_test__)
                    del ordered_matches_test__


                # # # del combined_ref_panel_chip
                # print(f"--- {sample_name} time: {(time.time() - start_sample_time)} seconds ---")

            # # var_dump(f"stage2/06_ordered_matches_list_1sample_12.pkl", ordered_matches_list)
            # ordered_matches_list = var_load(f"stage2/06_ordered_matches_list_1sample_12.pkl")
            # # del full_ref_panel_chip_array
            # # ordered_matches_list = var_load(f"opts01-05_dev1.3.1/06_ordered_matches_list_1sample_12.pkl")
            # ordered_matches_list = var_load(f"opts01-05_dev1.3.1/06_ordered_matches_list_10samples_12-21.pkl")
            # # ordered_matches_list = [ordered_matches_list[0]]
            # # exit(0)


            # imputing_samples_haploids_indices: List[int] = []
            # for sample, haps in imputing_samples.items():
            #     for hap in haps.keys():
            #         imputing_samples_haploids_indices.append(2*test_samples_index[sample] + hap)



            # kuklog_timestamp_func = lambda s: kuklog_timestamp(f"f-b: {s}", KUKLOG)
            resultsoo_fb = run_hmm(
                chip_sites_indicies, # common
                full_refpanel_vcfgz_path,  # common
                chip_sites_n, # common
                ordered_matches_list, # haploid-specific
                chip_cM_coordinates,  # common
                np.array(None), np.array(None), # unused
                fullseq_sites_n, # common
                num_hid=refpan_haps_n, # common
                kuklog_timestamp_func=lambda s : None,
            )

            # # var_dump(f"opts01-04_dev1/{sample}/{hap}/07_resultoo_fb.pkl", resultoo_fb)
            # # resultoo_fb_true = var_load(f"opts01-04_dev1/{sample}/{hap}/07_resultoo_fb.pkl")
            # kuklog_timestamp(f"ran forward backward and imputation", KUKLOG)
            # # print(f"{(resultoo_fb == resultoo_fb_true).all()=}")
            # # print(f"{np.where(resultoo_fb != resultoo_fb_true)=}")
            # # if not (resultoo_fb == resultoo_fb_true).all():
            # #     raise RuntimeError(f"`run_hmm` result is different from original, check what's wrong there")
            # # var_dump(f"opts01-04_dev1/{sample}/{hap}/07_post.pkl", post)
            # # smpls2take = post.astype(bool).sum(axis=1)
            # # var_dump(f"opts01-04_dev1/{sample}/{hap}/07_smpls2take.pkl", smpls2take)

            if resultsoo_fb is None or len(resultsoo_fb) == 0:
                raise RuntimeError(f"run_hmm returned `None` instead of a numpy array")

            # print("DONE")

            imputed_sample_file = f'{output_path}.sample{sample_i:04}.tsv'

            for hap in HAPS:
                resultoo_fb = resultsoo_fb[hap]

                full_res_.setdefault(sample_name, []).append(resultoo_fb) # type: ignore # Dict.setdefault returns a list

                if hap == 1:
                    # print("saving results")

                    save_imputed_in_stripped_vcf(
                        sample_name,
                        full_res_[sample_name],
                        imputed_sample_file,
                    )

            del resultsoo_fb

            imputed_sample_files.append(imputed_sample_file)


        assert len(imputed_sample_files) == len(samples)


        return imputed_sample_files





    if cores > n_input_samples:
        cores = n_input_samples
    samples_segments_points = split_range_into_n_whole_even_subranges(n_input_samples, cores)
    samples_segments: List[List[int]] = []

    assert len(samples_segments_points) == cores+1, f"got {len(samples_segments_points)=} and {cores=}"

    samples_idxs = list(range(n_input_samples))

    for i in range(1, len(samples_segments_points)):
        samples_segments.append(samples_idxs[samples_segments_points[i-1]:samples_segments_points[i]])


    # samples_segments = [[12],[13],[14],[15],[16],[17],[18],[19]]
    # samples_segments = [[12],[13],[14],[15]]
    # cores = 4

    assert len(samples_segments) == cores > 0, f"got {len(samples_segments)=} and {cores=}"

    total_samples_to_impute = sum([len(seg) for seg in samples_segments])

    print(f"going to impute {total_samples_to_impute} samples using {cores} cores ...")


    # # sequential
    # for samples in samples_segments:
    #     impute_input_vcf_sample(samples)

    # in parallel
    imputed_sample_files_lists: List[List[str]] = Parallel(n_jobs=cores)(
        delayed(impute_input_vcf_sample)(samples) for samples in samples_segments
    ) # type: ignore # it will not return None given above 
    # if cores == 1:
    #     pl = Pool(cores)
    # else:
    #     pl = Pool(cores, initializer=mute_stdout)
    # imputed_sample_files_lists = pl.map(impute_input_vcf_sample, samples_segments)


    # flatten the list of lists into a single list
    imputed_sample_files: List[str] = [f for sublist in imputed_sample_files_lists for f in sublist]

    subprocess.check_call([formvcf_script,
        f'{full_refpanel_vcfgz_path}.vcfheader.txt',
        f'{full_refpanel_vcfgz_path}.cols1-9.tsv',
        output_path,

        *imputed_sample_files,
    ])

    subprocess.check_call(['rm',
        *imputed_sample_files,
    ])



    print("===== Total time: %s seconds =====" % (time.time() - start_time))

    # mismatch_log_f.close()

    close_kuklog(KUKLOG)

    return imputed_sample_files












if __name__ == '__main__':

    thedir = os.path.dirname(os.path.abspath(__file__))
    formvcf_script = f"{thedir}/scripts/form_vcf_from_imputed.sh"

    chipsites_input = sys.argv[1]
    full_refpanel_vcfgz_path = sys.argv[2]
    chip_refpanel_vcfgz_path = get_chipsites_refpan_path(chipsites_input, full_refpanel_vcfgz_path)
    genetic_map_path = sys.argv[3]
    output_path = sys.argv[4]
    cores = int(sys.argv[5])



    selphi(
        input_vcfgz_path=chipsites_input,
        chip_refpanel_vcfgz_path=chip_refpanel_vcfgz_path,
        full_refpanel_vcfgz_path=full_refpanel_vcfgz_path,
        genetic_map_path=genetic_map_path,

        output_path=output_path,

        cores = cores,
    )








