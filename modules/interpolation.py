from typing import List, Tuple
from pathlib import Path
from uuid import uuid4
from math import ceil
import numpy as np
from scipy import sparse

from joblib import Parallel, delayed

from .array2vcf import VcfWriter
from .sparse_ref_panel import SparseReferencePanel
from .utils import tqdm_joblib


def _interpolate_hap(
    ref_pair: Tuple[int],
    chip_pair: Tuple[int],
    weight_matrix: sparse.csr_matrix,
    is_last: bool,
    ref_haplotypes: SparseReferencePanel,
) -> np.ndarray:
    """
    Given haplotype weights calculated by hmm at beginning and end of interval,
    interpolate weights for each variant in the reference panel in that interval.
    Call the haplotype as alt if the sum of weights for alt alleles in the
    reference panel is higher than the sum of weights for ref alleles in the
    reference panel.

    return: array of haplotype calls, True = alt, False = ref

    Note: returning the alt_probs matrix instead of comparing it to 0.5 would be
    analagous to the AP value calculated by Beagle.
    """
    start_row = weight_matrix[chip_pair[0], :]
    end_row = weight_matrix[chip_pair[1], :]
    ref_haps = np.union1d(start_row.indices, end_row.indices)
    assert ref_haps.size > 0
    # Interpolate weights and calculate alt totals
    weights = np.linspace(
        start_row[:, ref_haps].toarray()[0],
        end_row[:, ref_haps].toarray()[0],
        num=ref_pair[1] - ref_pair[0] + int(is_last),
        endpoint=False,
    )
    return (
        weights
        * ref_haplotypes[ref_pair[0] : ref_pair[1] + int(is_last), ref_haps].toarray()
    ).sum(axis=1) / weights.sum(axis=1)


def _interpolate_interval(
    index_pair: List[int],
    ordered_weights: np.ndarray,
    original_chip_indices: np.ndarray,
    original_ref_indices: np.ndarray,
    ref_haplotypes: SparseReferencePanel,
) -> np.ndarray:
    """
    Calculate indices for an interval and then interpolate all haplotypes
    for that interval.

    return: columns of haplotype calls with one variant per row
    """
    ref_pair = tuple(original_ref_indices[index_pair])
    chip_pair = tuple(original_chip_indices[index_pair])
    is_last = index_pair[1] == original_ref_indices.size - 1
    return np.column_stack(
        [
            _interpolate_hap(
                ref_pair, chip_pair, weight_matrix, is_last, ref_haplotypes
            )
            for weight_matrix in ordered_weights
        ]
    )


def interpolate(
    index_pairs: List[Tuple[int]],
    ordered_weights: np.ndarray,
    original_chip_indices: np.ndarray,
    original_ref_indices: np.ndarray,
    ref_haplotypes: SparseReferencePanel,
    writer: VcfWriter,
) -> Path:
    """
    Interpolate all intervals in a list, stacking the intervals into one matrix.
    Allows for easy parallel processing of chunks of a chromosome.
    Write chunk to file and return Path.
    """
    alt_probs = np.row_stack(
        [
            _interpolate_interval(
                list(index_pair),
                ordered_weights,
                original_chip_indices,
                original_ref_indices,
                ref_haplotypes,
            )
            for index_pair in index_pairs
        ]
    )
    start = original_ref_indices[index_pairs[0][0]]
    stop = original_ref_indices[index_pairs[-1][1]]
    in_target = original_ref_indices[np.trim_zeros(np.unique(index_pairs), "f")] - start
    if stop == original_ref_indices[-1]:
        stop += 1
        in_target = in_target[:-1]
    return writer.write_variants(ref_haplotypes.ids[start:stop], in_target, alt_probs)


def interpolate_genotypes(
    ref_haplotypes: SparseReferencePanel,
    target_samples: List[str],
    ordered_weights: np.ndarray,
    wgs_idx: np.ndarray,
    target_idx: np.ndarray,
    output_path: Path,
    tmpdir: Path,
    threads: int = 1,
) -> str:
    # Prepare to write results to VCF
    writer = VcfWriter(target_samples, ref_haplotypes.chromosome, tmpdir)
    writer.write_header()

    # Prepare intervals
    ref_order = np.argsort(wgs_idx)
    original_ref_indices = np.concatenate(
        ([0], wgs_idx[ref_order], [ref_haplotypes.n_variants - 1])
    )
    # Weights matrix has extra rows at 0 and -1
    original_chip_indices = np.concatenate(
        ([0], np.argsort(target_idx[ref_order]) + 1, [target_idx.size + 1])
    )
    intervals = list(
        zip(range(original_ref_indices.size - 1), range(1, original_ref_indices.size))
    )
    # For best performance, threads should not try to read
    # from the same ref panel chunks at the same time
    chunk_size = ceil(
        (original_ref_indices.size - 1) / max(ref_haplotypes.n_chunks // 10, 1)
    )

    with tqdm_joblib(
        total=len(intervals) // chunk_size + 1,
        desc=" [interpolate] Interpolating genotypes at missing variants",
        ncols=75,
        bar_format="{desc}:\t\t{percentage:3.0f}% in {elapsed}",
        colour="#808080",
    ):
        filelist = Parallel(n_jobs=threads,)(
            delayed(interpolate)(
                intervals[start : start + chunk_size],
                ordered_weights,
                original_chip_indices,
                original_ref_indices,
                ref_haplotypes,
                writer,
            )
            for start in range(0, len(intervals), chunk_size)
        )

    # make sure path exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If the provided output is a directory, generate a random filename
    # this will avoid that multiple VCF creation in the same folder are overwritten
    if output_path.is_dir():
        output_path = output_path.joinpath(str(uuid4()))

    return writer.complete_vcf(filelist, output_path)
