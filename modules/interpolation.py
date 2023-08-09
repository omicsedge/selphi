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


class Interpolator:
    """Class to interpolate allele probabilities"""

    def __init__(
        self,
        ref_haplotypes: SparseReferencePanel,
        target_samples: List[str],
        wgs_idx: np.ndarray,
        target_idx: np.ndarray,
        tmpdir: Path,
        threads: int = 1,
    ):
        self.ref_haplotypes = ref_haplotypes
        self.target_samples = target_samples

        ref_order = np.argsort(wgs_idx)
        self.wgs_idx = wgs_idx[ref_order]
        self.target_idx = target_idx[ref_order]
        del ref_order

        self.tmpdir = tmpdir
        self.threads = threads

        self.chunk_size = ceil(
            (self.wgs_idx.size + 1) / max(ref_haplotypes.n_chunks // 10, 1)
        )
        self.original_chip_indices = np.concatenate(
            ([0], np.argsort(self.target_idx) + 1, [self.target_idx.size + 1])
        )
        self.original_ref_indices = np.concatenate(
            ([0], self.wgs_idx, [ref_haplotypes.n_variants - 1])
        )
        self.intervals = list(
            zip(
                range(self.original_ref_indices.size - 1),
                range(1, self.original_ref_indices.size),
            )
        )
        self.chunks = [
            self.intervals[start : start + self.chunk_size]
            for start in range(0, len(self.intervals), self.chunk_size)
        ]
        self.breakpoints = [(chunk[0][0], chunk[-1][1] + 1) for chunk in self.chunks]

        for start, _ in self.breakpoints:
            tmpdir.joinpath("weights", str(start)).mkdir(parents=True, exist_ok=True)
        self.writer = VcfWriter(
            self.target_samples, self.ref_haplotypes.chromosome, self.tmpdir
        )

    def _interpolate_hap(
        self,
        ref_pair: Tuple[int],
        chip_pair: Tuple[int],
        weight_matrix: sparse.csr_matrix,
        start: int,
        is_last: bool,
    ) -> sparse.coo_matrix:
        """
        Given haplotype weights calculated by hmm at beginning and end of interval,
        interpolate weights for each variant in the reference panel in that interval.
        Call the haplotype as alt if the sum of weights for alt alleles in the
        reference panel is higher than the sum of weights for ref alleles in the
        reference panel.

        return: array of floats, prob of alt allele
        """
        start_row = weight_matrix[chip_pair[0] - start, :]
        end_row = weight_matrix[chip_pair[1] - start, :]
        ref_haps = np.union1d(start_row.indices, end_row.indices)
        assert ref_haps.size > 0
        # Interpolate weights and calculate alt totals
        weights = np.linspace(
            start_row[:, ref_haps].toarray()[0],
            end_row[:, ref_haps].toarray()[0],
            num=ref_pair[1] - ref_pair[0] + int(is_last),
            endpoint=False,
        )
        return sparse.coo_matrix(
            (
                weights
                * self.ref_haplotypes[
                    ref_pair[0] : ref_pair[1] + int(is_last), ref_haps
                ].toarray()
            ).sum(axis=1)
            / weights.sum(axis=1),
            dtype=np.float_,
        )

    def _interpolate_interval(
        self,
        index_pair: List[int],
        start: int,
        ordered_weights: np.ndarray,
    ) -> sparse.csc_matrix:
        """
        Calculate indices for an interval and then interpolate all haplotypes
        for that interval.

        return: columns of haplotype calls with one variant per row
        """
        ref_pair = tuple(self.original_ref_indices[index_pair])
        chip_pair = tuple(self.original_chip_indices[index_pair])
        is_last = index_pair[1] == self.original_ref_indices.size - 1
        return sparse.vstack(
            [
                self._interpolate_hap(
                    ref_pair, chip_pair, weight_matrix, start, is_last
                )
                for weight_matrix in ordered_weights
            ]
        ).tocsc()

    def interpolate(
        self,
        index_pairs: List[Tuple[int]],
    ) -> Path:
        """
        Interpolate all intervals in a list, stacking the intervals into one matrix.
        Allows for easy parallel processing of chunks of a chromosome.
        Write chunk to file and return Path.
        """
        start_idx = index_pairs[0][0]
        ordered_weights = np.array(
            [
                sparse.load_npz(
                    self.tmpdir.joinpath(
                        "weights", str(start_idx), f"{sample}_{hap}.npz"
                    )
                )
                for sample in self.target_samples
                for hap in (0, 1)
            ],
            dtype=object,
        )
        alt_probs = sparse.hstack(
            [
                self._interpolate_interval(list(index_pair), start_idx, ordered_weights)
                for index_pair in index_pairs
            ]
        )
        del ordered_weights
        start = self.original_ref_indices[start_idx]
        stop = self.original_ref_indices[index_pairs[-1][1]]
        in_target = (
            self.original_ref_indices[np.trim_zeros(np.unique(index_pairs), "f")]
            - start
        )
        if stop == self.original_ref_indices[-1]:
            stop += 1
            in_target = in_target[:-1]
        return self.writer.write_variants(
            self.ref_haplotypes.ids[start:stop],
            in_target,
            alt_probs.transpose(),
        )

    def interpolate_genotypes(self, output_path: Path) -> str:
        # Prepare to write results to VCF
        self.writer.write_header()

        with tqdm_joblib(
            total=len(self.intervals) // self.chunk_size + 1,
            desc=" [interpolate] Interpolating genotypes at missing variants",
            ncols=75,
            bar_format="{desc}:\t\t{percentage:3.0f}% in {elapsed}",
        ):
            filelist = Parallel(n_jobs=self.threads)(
                delayed(self.interpolate)(chunk) for chunk in self.chunks
            )

        # make sure path exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If the provided output is a directory, generate a random filename
        # this will avoid that multiple VCF creation in the same folder are overwritten
        if output_path.is_dir():
            output_path = output_path.joinpath(str(uuid4()))

        return self.writer.complete_vcf(filelist, output_path)
