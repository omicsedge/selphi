import os
from typing import List, Set, Tuple, TypeVar, Union, Iterable, Dict, Literal
import sys

import numpy as np

from modules.load_data import get_chipsites_refpan_path
from modules.vcfgz_reader import vcfgz_reader
from modules.devutils import pickle_dump






def extract_chipsites_indicies_in_fullsequece(
    chipsites_input: str = f'./data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.chipsites-292-input-samples.vcf.gz',
    full_refpanel_vcfgz_path: str = f'./data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.fullseq-2910-refpan-samples.vcf.gz',
    chip_refpanel_vcfgz_path: str = f'./data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.chipsites-2910-refpan-samples.vcf.gz',

    batch_size: int = 20_000, # does not affect the result, only the memory usage during the process
):
    """
    assumption:
     - input samples sites are a subset of the sites in the reference panel, ordered in the same way
    """

    chip_sites_indicies_path = f'{chip_refpanel_vcfgz_path}.chip_sites_indicies.pkl'
    chip_BPs_path = f'{chipsites_input}.chip_BPs.pkl'
    fullseq_sites_n_path = f'{full_refpanel_vcfgz_path}.fullseq_sites_n.pkl'
    chip_sites_n_path = f'{chip_refpanel_vcfgz_path}.chip_sites_n.pkl'



    if all(map(os.path.isfile, [
        chip_sites_indicies_path, 
        chip_BPs_path,
        fullseq_sites_n_path,
        chip_sites_n_path,
    ])):
        print(f" - \"{chip_sites_indicies_path}\" already exists")
        print(f" - \"{chip_BPs_path}\" already exists")
        print(f" - \"{fullseq_sites_n_path}\" already exists")
        print(f" - \"{chip_sites_n_path}\" already exists")

    else:
        print(f"creating:")
        print(f" - \"{chip_sites_indicies_path}\" already exists")
        print(f" - \"{chip_BPs_path}\" already exists")
        print(f" - \"{fullseq_sites_n_path}\" already exists")
        print(f" - \"{chip_sites_n_path}\" already exists")

        fullrefpanel = vcfgz_reader(full_refpanel_vcfgz_path)
        chiprefpanel = vcfgz_reader(chip_refpanel_vcfgz_path)


        F_ID_col, C_ID_col = True, True


        def readbatch_full(F_ID_col: Union[np.ndarray, Literal[True]]):
            if F_ID_col is not True:
                F_ID_col[:] = b''
            return fullrefpanel.read_columns(lines_n=batch_size, ID=F_ID_col)

        def readbatch_chip(C_ID_col: Union[np.ndarray, Literal[True]]):
            if C_ID_col is not True:
                C_ID_col[:] = b''
            return chiprefpanel.read_columns(lines_n=batch_size, POS=True, ID=C_ID_col)


        # read first batches
        F_lines_read_n, [F_ID_col] = readbatch_full(F_ID_col)
        C_lines_read_n, [C_POS_col, C_ID_col] = readbatch_chip(C_ID_col)

        C_POS_list = C_POS_col[:C_lines_read_n]



        chip_sites_indicies: List[int] = []

        i_C: int = -1
        i_F: int = -1

        i_C_batch: int = -1 # variant idx within the chip-sites batch (input sample)
        i_F_batch: int = -1 # variant idx within the full-sequence batch (ref panel)

        v_C: str = C_ID_col[0]
        v_F: str = F_ID_col[0]


        # loop through all of the chip sites, and loop through the ref panel sites,
        #   while collecting indicies of chip sites in the reference panel
        while (C_lines_read_n != 0):

            for i_C_batch in range(C_lines_read_n):
                i_C += 1
                v_C = C_ID_col[i_C_batch]

                while (v_F != v_C):
                    i_F_batch += 1
                    i_F += 1

                    if i_F_batch >= F_lines_read_n:
                        F_lines_read_n, [F_ID_col] = readbatch_full(F_ID_col)
                        i_F_batch = 0
                        if F_lines_read_n == 0:
                            break

                    v_F = F_ID_col[i_F_batch]

                if F_lines_read_n == 0:
                    break
                chip_sites_indicies.append(i_F)

            C_lines_read_n, [C_POS_col, C_ID_col] = readbatch_chip(C_ID_col)
            C_POS_list = np.concatenate((C_POS_list, C_POS_col[:C_lines_read_n]), axis=0)

        C_POS_list_int = np.array(C_POS_list).astype(np.int64)

        chip_sites_n: int = i_C + 1



        # loop through the rest of the reference panel variants
        while (True):
            i_F_batch += 1
            i_F += 1

            if i_F_batch >= F_lines_read_n:
                F_lines_read_n, [F_ID_col] = readbatch_full(F_ID_col)
                i_F_batch = 0
                if F_lines_read_n == 0:
                    break

        fullseq_sites_n: int = i_F



        pickle_dump(chip_sites_indicies, chip_sites_indicies_path)
        pickle_dump(list(C_POS_list_int), chip_BPs_path)

        pickle_dump(fullseq_sites_n, fullseq_sites_n_path)
        pickle_dump(chip_sites_n, chip_sites_n_path)







if __name__ == "__main__":
    # chrom = "chr20"
    # chipsites_input = f'{dir}/reference_panel.30x.hg38_{chrom}_noinfo.chipsites-292-input-samples.vcf.gz'
    # full_refpanel_vcfgz_path = f'{dir}/reference_panel.30x.hg38_{chrom}_noinfo.fullseq-2910-refpan-samples.vcf.gz'
    # chip_refpanel_vcfgz_path = f'{dir}/reference_panel.30x.hg38_{chrom}_noinfo.chipsites-2910-refpan-samples.vcf.gz'

    chipsites_input=sys.argv[1]
    full_refpanel_vcfgz_path=sys.argv[2]

    chip_refpanel_vcfgz_path = get_chipsites_refpan_path(chipsites_input, full_refpanel_vcfgz_path)


    batch_size = 20_000


    extract_chipsites_indicies_in_fullsequece(
        chipsites_input=chipsites_input,
        full_refpanel_vcfgz_path=full_refpanel_vcfgz_path,
        chip_refpanel_vcfgz_path=chip_refpanel_vcfgz_path,
        batch_size=batch_size,
    )






