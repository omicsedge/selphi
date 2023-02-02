from typing import Dict, List, Literal, Tuple, Union, OrderedDict as OrderedDict_type
from collections import OrderedDict
# import dill as pickle
from joblib import Parallel, delayed
import sys
import subprocess
from datetime import datetime as dt
import time


import os
import numpy as np


from modules.load_data import (
    get_chipsites_refpan_path,
    load_sequences_metadata,
    load_and_interpolate_genetic_map,
    load_chip_reference_panel_combined,
)

from modules.imputation_lib import (
    BiDiPBWT,
    create_composite_ref_panel_np,
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
from modules.devutils import forced_open, pickle_dump, pickle_load, nptype, var_load
from modules.kuklog import open_kuklog, kuklog_timestamp, close_kuklog
from modules.vcfgz_reader import vcfgz_reader








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
        genetic_map_path = genetic_map_path,
        chip_BPs = chip_BPs,
    )
    kuklog_timestamp("loaded and interpolated genetic map", KUKLOG)
    print("loaded and interpolated genetic map")




    CHUNK_SIZE = chip_sites_n # number of chip variants
    HAPS: List[Literal[0,1]] = [0,1]


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

            ordered_matches_list: List[Dict[int, List[np.int64]]] = []
            imputing_samples: OrderedDict_type[str, OrderedDict_type[Literal[0,1], Literal[True]]] = OrderedDict()

            full_res_: Dict[str, np.ndarray] = {}


            for sample_i, sample_name in [(sample_i, sample_name)]:
                imputing_samples[sample_name] = OrderedDict()

                start_sample_time = time.time()
                print(f"{dt.now().strftime('%c')}\t{sample_name}")

                combined_ref_panel_chip = load_chip_reference_panel_combined(
                    chip_sites_refpanel_file = chip_refpanel_vcfgz_path,
                    chip_sites_n = chip_sites_n,
                    input_samples = input_samples,
                    sample_i = sample_i,
                    sample_name = sample_name,
                )
                assert refpan_haps_n == combined_ref_panel_chip.shape[1] - 2 # number of reference panel haploids
                # kuklog_timestamp("loaded reference panels", KUKLOG)

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

                    # kuklog_timestamp(f"hap{hap}: ran BiDiPBWT", KUKLOG) #1
                    matches, composite_, comp_matches_hybrid = create_composite_ref_panel_np(
                        BI,
                        BJ,
                        fl=13,
                    )
  
                    # kuklog_timestamp(f"hap{hap}: created first composite ref panel", KUKLOG) #2
                    haps_freqs_array_norm_dict = calculate_haploid_frequencies(
                        matches,
                        comp_matches_hybrid,
                        CHUNK_SIZE,
                    )
                    # kuklog_timestamp(f"hap{hap}: calculated 1st filter: haploid frequencies", KUKLOG) #3
                    nc_thresh = calculate_haploid_count_threshold(
                        matches,
                        composite_,
                        comp_matches_hybrid,
                        haps_freqs_array_norm_dict,
                        CHUNK_SIZE,
                    )
                    del comp_matches_hybrid
                    del composite_
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
                    del BI
                    del BJ
                    del haps_freqs_array_norm_dict
                    del nc_thresh
                    # kuklog_timestamp(f"hap{hap}: applied the 2 filters", KUKLOG) #5
                    ordered_matches_test__ = form_haploid_ids_lists(
                        matches,
                        composite_,
                    )
                    del matches
                    del composite_
                    # kuklog_timestamp(f"hap{hap}: form haploid ids list", KUKLOG)

                    ordered_matches_list.append(ordered_matches_test__)
                    del ordered_matches_test__


                # # # del combined_ref_panel_chip
                # print(f"--- {sample_name} time: {(time.time() - start_sample_time)} seconds ---")


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

            if resultsoo_fb is None or len(resultsoo_fb) == 0:
                raise RuntimeError(f"run_hmm returned `None` instead of a numpy array")

            imputed_sample_file = f'{output_path}.sample{sample_i:04}.tsv'

            for hap in HAPS:
                resultoo_fb = resultsoo_fb[hap]

                full_res_.setdefault(sample_name, []).append(resultoo_fb) # type: ignore # Dict.setdefault returns a list

                if hap == 1:

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



    assert len(samples_segments) == cores > 0, f"got {len(samples_segments)=} and {cores=}"

    # samples_segments = [[12],[13],[14],[15],[16],[17],[18],[19],[20],[21]]
    # cores = 10
    samples_segments = [[12]]
    cores = 1


    total_samples_to_impute = sum([len(seg) for seg in samples_segments])

    print(f"going to impute {total_samples_to_impute} samples using {cores} cores ...")


    # # sequential
    # for samples in samples_segments:
    #     impute_input_vcf_sample(samples)

    # in parallel
    imputed_sample_files_lists: List[List[str]] = Parallel(n_jobs=cores)(
        delayed(impute_input_vcf_sample)(samples) for samples in samples_segments
    ) # type: ignore # it will not return None given above 


    # flatten the list of lists into a single list
    imputed_sample_files: List[str] = [f for sublist in imputed_sample_files_lists for f in sublist]

    # join imputed samples into a single vcf
    subprocess.check_call([formvcf_script,
        f'{full_refpanel_vcfgz_path}.vcfheader.txt',
        f'{full_refpanel_vcfgz_path}.cols1-9.tsv',
        output_path,

        *imputed_sample_files,
    ])

    # remove individual single-column files with imputed samples 
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








