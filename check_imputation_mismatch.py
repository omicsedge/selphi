from typing import Dict, List
import sys

import pandas as pd
import numpy as np

from modules.vcfgz_reader import vcfgz_reader


try:
    imputed_vcfgz = sys.argv[1]
    true_vcfgz = sys.argv[2]
except IndexError:
    print(
        ValueError("""you have to supply 2 paths to 2 vcf/vcf.gz files as arguments""")
    )
    exit(10)


imputed = vcfgz_reader(imputed_vcfgz)
true = vcfgz_reader(true_vcfgz)

if tuple(imputed.samples) != tuple(true.samples):
    raise ValueError("the two vcf datasets have to have the same samples")

n_samples = len(imputed.samples)


batch_size = 50_000

accuracy_table: List[Dict[str, int]] = []  # actually counts mismatches
for sample in imputed.samples:
    accuracy_table.append(
        dict(
            {
                "hap0": 0,
                "hap1": 0,
                "genotype": 0,
            }
        )
    )


imputed_GT = True
true_GT = True
lines_read_n1 = 1
lines_read_n2 = 1


batch_from = 0
batch_to = 0
while lines_read_n1 != 0 and lines_read_n2 != 0:
    lines_read_n1, [imputed_GT] = imputed.read_columns(
        lines_n=batch_size, GENOTYPES=imputed_GT
    )
    lines_read_n2, [true_GT] = true.read_columns(lines_n=batch_size, GENOTYPES=true_GT)
    if lines_read_n1 == 0 or lines_read_n2 == 0:
        break
    batch_to = batch_from + lines_read_n1
    print(f"batch: [{batch_from}, {batch_to})")

    assert (
        imputed_GT.shape[1] == true_GT.shape[1]
    ), "number of samples in the two datasets has to be the same"
    assert (
        lines_read_n1 == lines_read_n2
    ), "number of sites in the two datasets has to be the same"

    imputed_GT_hap0: np.ndarray = imputed_GT[:lines_read_n1, ::2]
    imputed_GT_hap1: np.ndarray = imputed_GT[:lines_read_n1, 1::2]
    true_GT_hap0: np.ndarray = true_GT[:lines_read_n2, ::2]
    true_GT_hap1: np.ndarray = true_GT[:lines_read_n2, 1::2]

    imputed_GT_GT: np.ndarray = imputed_GT_hap0 + imputed_GT_hap1
    true_GT_GT: np.ndarray = true_GT_hap0 + true_GT_hap1

    hap0_mismatch: np.ndarray = imputed_GT_hap0 != true_GT_hap0
    hap1_mismatch: np.ndarray = imputed_GT_hap1 != true_GT_hap1
    GT_mismatch: np.ndarray = imputed_GT_GT != true_GT_GT

    for i in range(n_samples):
        accuracy_table[i]["hap0"] += hap0_mismatch[:, i].sum()
        accuracy_table[i]["hap1"] += hap1_mismatch[:, i].sum()

        accuracy_table[i]["genotype"] += GT_mismatch[:, i].sum()

    batch_from = batch_to


df = pd.DataFrame(accuracy_table)
df["sample"] = imputed.samples
df[["sample", "hap0", "hap1", "genotype"]].to_csv(
    f"{imputed_vcfgz}.mismatches.tsv",
    sep="\t",
)
