from typing import List, Tuple
from pathlib import Path
from uuid import uuid4
from math import ceil
import shutil
import psutil
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
        targets_path: Path,
        wgs_idx: np.ndarray,
        target_idx: np.ndarray,
        tmpdir: Path,
        version: str,
        threads: int = 1,
    ):
        self.ref_haplotypes = ref_haplotypes
        self.target_samples = target_samples
        self.targets_path = targets_path

        ref_order = np.argsort(wgs_idx)
        self.wgs_idx = wgs_idx[ref_order]
        self.target_idx = target_idx[ref_order]
        del ref_order

        self.tmpdir = tmpdir
        self.version = version
        self.threads = self._calculate_optimal_interpolation_cores(threads)

        self.chunk_size = self._calculate_optimal_chunk_size(
            ref_haplotypes, self.wgs_idx.size
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
            self.target_samples,
            self.targets_path,
            self.ref_haplotypes.chromosome,
            self.tmpdir,
            self.version,
        )

    def _calculate_optimal_chunk_size(
        self, ref_haplotypes: SparseReferencePanel, wgs_idx_size: int
    ) -> int:
        """
        Calculate optimal chunk size based on available memory and data characteristics.
        Larger chunks when memory allows for better performance.
        """
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Base chunk size calculation (original logic)
        base_chunk_size = ceil(
            (wgs_idx_size + 1) / max(ref_haplotypes.n_chunks // 10, 1)
        )
        
        # Estimate memory usage per chunk (rough approximation)
        # Consider: sparse matrices, weight matrices, intermediate calculations
        estimated_memory_per_chunk_gb = (
            ref_haplotypes.n_haps * base_chunk_size * 8  # float64 weights
            + ref_haplotypes.chunk_size * ref_haplotypes.n_haps * 1  # sparse boolean
        ) / (1024**3)
        
        # Calculate maximum safe chunk size based on available memory
        # Use 60% of available memory to leave room for other operations
        max_safe_chunks = max(1, int(available_memory_gb * 0.6 / estimated_memory_per_chunk_gb))
        
        # Adaptive scaling: increase chunk size if memory allows
        if available_memory_gb > 8:  # If we have plenty of memory (>8GB)
            scaling_factor = min(3.0, available_memory_gb / 8)  # Scale up to 3x
            optimal_chunk_size = int(base_chunk_size * scaling_factor)
        elif available_memory_gb > 4:  # Moderate memory (4-8GB)
            optimal_chunk_size = int(base_chunk_size * 1.5)
        else:  # Limited memory (<4GB)
            optimal_chunk_size = base_chunk_size
        
        # Ensure we don't exceed memory constraints
        optimal_chunk_size = min(optimal_chunk_size, base_chunk_size * max_safe_chunks)
        
        # Ensure minimum chunk size for efficiency
        optimal_chunk_size = max(optimal_chunk_size, 1)
        
        return optimal_chunk_size

    def _calculate_optimal_interpolation_cores(self, requested_cores: int) -> int:
        """
        Calculate optimal number of cores for interpolation tasks based on system resources.
        
        Args:
            requested_cores: Number of cores requested by user
        
        Returns:
            Optimal number of cores for interpolation
        """
        # Get system information
        cpu_count = psutil.cpu_count(logical=True)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Interpolation can benefit from more cores but is less memory-intensive than HMM
        # Use logical cores since interpolation has good parallelization
        if available_memory_gb > 8:
            # Plenty of memory: use most cores
            optimal_cores = min(cpu_count, max(1, int(cpu_count * 0.8)))
        elif available_memory_gb > 4:
            # Moderate memory: use fewer cores
            optimal_cores = min(cpu_count, max(1, int(cpu_count * 0.6)))
        else:
            # Limited memory: conservative approach
            optimal_cores = min(cpu_count, max(1, int(cpu_count * 0.4)))
        
        # Respect user's request but don't exceed system capabilities
        if requested_cores > 0:
            optimal_cores = min(requested_cores, optimal_cores)
        
        # Ensure at least 1 core
        return max(1, optimal_cores)

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
        
        # More efficient memory usage for weight interpolation
        start_weights = start_row[:, ref_haps].toarray()[0]
        end_weights = end_row[:, ref_haps].toarray()[0]
        
        num_variants = ref_pair[1] - ref_pair[0] + int(is_last)
        weights = np.linspace(
            start_weights,
            end_weights,
            num=num_variants,
            endpoint=False,
        )
        
        # Get reference haplotypes slice more efficiently
        ref_slice = self.ref_haplotypes[
            ref_pair[0] : ref_pair[1] + int(is_last), ref_haps
        ]
        
        # Compute probabilities with better memory management
        ref_data = ref_slice.toarray()
        weighted_sum = (weights * ref_data).sum(axis=1)
        weight_totals = weights.sum(axis=1)
        
        # Avoid division by zero
        probabilities = np.divide(
            weighted_sum, 
            weight_totals, 
            out=np.zeros_like(weighted_sum), 
            where=weight_totals!=0
        )
        
        # Cleanup intermediate arrays
        del start_weights, end_weights, ref_data, weighted_sum, weight_totals
        
        return sparse.coo_matrix(probabilities, dtype=np.float64)

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
        
        # Pre-allocate memory for better performance
        n_samples_haps = len(self.target_samples) * 2
        ordered_weights = np.empty(n_samples_haps, dtype=object)
        
        # Load weights with explicit memory management
        for i, (sample, hap) in enumerate(
            [(sample, hap) for sample in self.target_samples for hap in (0, 1)]
        ):
            ordered_weights[i] = sparse.load_npz(
                self.tmpdir.joinpath(
                    "weights", str(start_idx), f"{sample}_{hap}.npz"
                )
            )
        
        # Pre-calculate total number of variants for efficient matrix allocation
        total_variants = sum(
            self.original_ref_indices[pair[1]] - self.original_ref_indices[pair[0]]
            + (1 if pair[1] == self.original_ref_indices.size - 1 else 0)
            for pair in index_pairs
        )
        
        # Use list comprehension with explicit memory cleanup
        interval_results = []
        for index_pair in index_pairs:
            result = self._interpolate_interval(list(index_pair), start_idx, ordered_weights)
            interval_results.append(result)
        
        # Stack results efficiently
        alt_probs = sparse.hstack(interval_results)
        
        # Explicit cleanup
        del ordered_weights, interval_results
        start = self.original_ref_indices[start_idx]
        stop = self.original_ref_indices[index_pairs[-1][1]]
        in_target = (
            self.original_ref_indices[np.trim_zeros(np.unique(index_pairs), "f")]
            - start
        )
        if stop == self.original_ref_indices[-1]:
            stop += 1
            in_target = in_target[:-1]
        chunk_path = self.writer.write_variants(
            self.ref_haplotypes.ids[start:stop],
            self.ref_haplotypes.original_ids[start:stop],
            in_target,
            alt_probs.transpose(),
        )
        shutil.rmtree(
            self.tmpdir.joinpath("weights", str(start_idx)), ignore_errors=True
        )
        return chunk_path

    def interpolate_genotypes(self, output_path: Path) -> str:
        # Prepare to write results to VCF
        self.writer.write_header(self.ref_haplotypes.contig_field)

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
