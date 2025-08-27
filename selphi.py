import os
import shutil
import sys
import subprocess
from hashlib import blake2b
from pathlib import Path
import argparse
import logging
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError

import numpy as np
import cyvcf2

from modules.imputation_lib import calculate_weights
from modules.interpolation import Interpolator
from modules.load_data import load_and_interpolate_genetic_map
from modules.pbwt import get_pbwt_matches
from modules.sparse_ref_panel import SparseReferencePanel
from modules.utils import add_suffix, get_version, timestamp, tqdm_joblib

logger = logging.getLogger("Selphi")
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

version = get_version()


def selphi(
    targets_path: Path,
    ref_base_path: Path,
    genetic_map_path: Path,
    output_path: Path,
    pbwt_path: Path,
    tmpdir: Path,
    match_length: int = 5,
    est_ne: int = 1000000,
    cores: int = 1,
    adjust_cores: bool = True,
):
    # Check if required files are available
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing target vcf/bcf: {targets_path}")
    if not add_suffix(ref_base_path, ".pbwt").exists():
        raise FileNotFoundError(
            f"Missing pbwt reference files: {add_suffix(ref_base_path, '.pbwt')}. "
            "Use --prepare_reference to create required reference files."
        )
    if not add_suffix(ref_base_path, ".srp").exists():
        raise FileNotFoundError(
            f"Missing pbwt reference files: {add_suffix(ref_base_path, '.srp')}. "
            "Use --prepare_reference to create required reference files."
        )
    if not genetic_map_path.exists():
        raise FileNotFoundError(f"Missing genetic map file: {genetic_map_path}")

    start_time = datetime.now()

    # Load sparse reference panel
    ref_panel = SparseReferencePanel(
        str(add_suffix(ref_base_path, ".srp")), cache_size=2
    )
    # Confirm that reference panel files match
    n_pbwt_samples = add_suffix(ref_base_path, ".samples").read_text().count("\n")
    n_pbwt_vars = add_suffix(ref_base_path, ".sites").read_text().count("\n")
    if n_pbwt_samples != ref_panel.n_samples or n_pbwt_vars != ref_panel.n_variants:
        raise ValueError(
            f"pbwt reference panel has {n_pbwt_samples} samples and {n_pbwt_vars} "
            f"variants while sparse reference panel has {ref_panel.n_samples} samples "
            f"and {ref_panel.n_variants} variants"
        )

    # Load target samples and variants
    if not (
        add_suffix(targets_path, ".tbi").exists()
        or add_suffix(targets_path, ".csi").exists()
    ):
        logger.info(f"Indexing input file: {targets_path}")
        subprocess.run(
            f"bcftools index {targets_path} --threads {cores}", check=True, shell=True
        )

    vcf_obj = cyvcf2.VCF(targets_path)
    target_samples = vcf_obj.samples
    if not target_samples:
        raise IndexError(f"No samples found in target vcf: {targets_path}")
    if ref_panel.variants[0][2] not in ref_panel.ids[0]:
        # hash alleles
        target_markers = np.array(
            [
                (
                    variant.CHROM,
                    variant.POS,
                    blake2b(variant.REF.encode(), digest_size=8).hexdigest(),
                    blake2b(variant.ALT[0].encode(), digest_size=8).hexdigest(),
                )
                for variant in vcf_obj
            ],
            dtype=ref_panel.variant_dtypes,
        )
    else:
        target_markers = np.array(
            [
                (variant.CHROM, variant.POS, variant.REF, variant.ALT[0])
                for variant in vcf_obj
            ],
            dtype=ref_panel.variant_dtypes,
        )
    del vcf_obj
    if target_markers.size == 0:
        raise IndexError(f"No variants found in target vcf: {targets_path}")
    if target_markers[0][0] != ref_panel.chromosome:
        raise KeyError(
            f"Reference panel chromosome {ref_panel.chromosome} does not match "
            f"target samples chromosome {target_markers[0][0]}"
        )
    shared, wgs_idx, target_idx = np.intersect1d(
        ref_panel.variants, target_markers, return_indices=True
    )
    del shared
    target_haps = [(sample, hap) for sample in target_samples for hap in (0, 1)]

    logger.info(
        "Stats:\n"
        f"Reference samples:\t\t{int(ref_panel.n_haps / 2)}\n"
        f"Target samples:\t\t\t{len(target_samples)}\n\n"
        f"Reference markers:\t\t{ref_panel.n_variants}\n"
        f"Target markers:\t\t\t{target_markers.size}\n\n"
        f"Chromosome:\t{ref_panel.chromosome}\n"
        f"Shared markers:\t{wgs_idx.size}\n\n"
        "Process:"
    )

    # Make filters to select variants for pbwt
    wgs_filter = np.zeros_like(ref_panel.variants, dtype=int)
    wgs_filter[wgs_idx] = 1
    target_filter = np.zeros_like(target_markers, dtype=int)
    target_filter[target_idx] = 1
    # Find matches to reference panel with pbwt
    pbwt_result_path: Path = get_pbwt_matches(
        pbwt_path,
        targets_path,
        ref_base_path,
        tmpdir.joinpath("matches"),
        target_samples,
        wgs_filter,
        target_filter,
        logger,
        match_length,
        cores,
    )
    expected_shape = (ref_panel.n_haps, wgs_idx.size)
    del wgs_filter
    del target_filter

    # Get coordinates for overlapping variants
    chip_cM_coordinates: np.ndarray = load_and_interpolate_genetic_map(
        genetic_map_path=genetic_map_path,
        chip_BPs=[variant[1] for variant in target_markers[np.sort(target_idx)]],
    )
    del target_markers

    # set up interpolation intervals
    interpolator = Interpolator(
        ref_panel,
        target_samples,
        targets_path,
        wgs_idx,
        target_idx,
        tmpdir,
        version,
        cores,
    )

    # Calculate HMM weights from matches
    # Reduce cores if not enough memory
    hmm_cores = min(cores, len(target_haps))
    while hmm_cores > 0:
        try:
            with tqdm_joblib(
                total=len(target_haps),
                desc=" [core] Calculating weights with HMM",
                ncols=75,
                bar_format="{desc}:\t\t\t\t\t{percentage:3.0f}% in {elapsed}",
            ):
                _ = Parallel(n_jobs=hmm_cores)(
                    delayed(calculate_weights)(
                        target_hap,
                        chip_cM_coordinates,
                        pbwt_result_path,
                        expected_shape,
                        tmpdir.joinpath("weights"),
                        interpolator.breakpoints,
                        logger,
                        est_ne,
                    )
                    for target_hap in target_haps
                )
            break
        except TerminatedWorkerError as exc:
            hmm_cores -= 1
            if not hmm_cores or not adjust_cores:
                raise TerminatedWorkerError from exc
            logger.info(
                f"HMM exceeded available memory, trying again with {hmm_cores} threads"
            )

    if pbwt_result_path != tmpdir:
        shutil.rmtree(pbwt_result_path, ignore_errors=True)
    del chip_cM_coordinates

    # Interpolate genotypes
    output_file = interpolator.interpolate_genotypes(output_path)

    logger.info(f"\n{timestamp()}: Saved imputed genotypes to {output_file}")
    logger.info(
        "===== Total time: %d seconds =====" % (datetime.now() - start_time).seconds
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"Selphi v{version}", description="PBWT for genotype imputation"
    )
    parser.add_argument(
        "--refpanel",
        type=str,
        required=True,
        help="location of reference panel files (required: %(required)s)",
    )
    parser.add_argument(
        "--target", type=str, help="path to vcf/bcf containing target samples"
    )
    parser.add_argument("--map", type=str, help="path to genetic map in plink format")
    parser.add_argument(
        "--outvcf", type=str, help="path to output imputed data to compressed vcf"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="number of cores available (default: %(default)s)",
    )
    parser.add_argument(
        "--prepare_reference",
        action="store_true",
        help="convert reference panel to pbwt and srp formats",
    )
    parser.add_argument(
        "--ref_source_vcf",
        type=str,
        help="location of vcf/bcf containing reference panel",
    )
    parser.add_argument(
        "--ref_source_xsi",
        type=str,
        help="location of xsi files containing reference panel",
    )
    parser.add_argument("--pbwt_path", type=str, help="path to pbwt library")
    parser.add_argument(
        "--tmp_path", type=str, help="location to create temporary directory"
    )
    parser.add_argument(
        "--match_length",
        type=int,
        default=5,
        help="minimum pbwt match length (default: %(default)s)",
    )
    parser.add_argument(
        "--est_ne",
        type=int,
        default=1000000,
        help="Ne for calculating recombination probability (default: %(default)s)",
    )
    parser.add_argument(
        "--no_core_reduction",
        action="store_true",
        help="Exit if memory limit is exceeded instead of reducing cores (default: %(default)s)",
    )
    args = parser.parse_args()
    cmd = " \ \n".join([f"  --{k} {v}" for k, v in vars(args).items() if v is not None])

    thedir = os.path.dirname(os.path.abspath(__file__))

    logger.info(
        (
            f"\nSelphi v{version}\n"
            "Copyright (C) 2023 Selfdecode\nStart time: "
            f"{datetime.now(timezone.utc).strftime('%-I:%M:%S %p %Z on %-d %b %Y')}\n\n"
            f"Command line: {sys.argv[0]} \ \n{cmd}\n"
        )
    )

    pbwt_path = Path(args.pbwt_path or thedir).resolve()
    if pbwt_path.is_dir():
        pbwt_path = pbwt_path.joinpath("pbwt")
        if pbwt_path.is_dir():
            pbwt_path = pbwt_path.joinpath("pbwt")
    if not pbwt_path.exists():
        raise FileNotFoundError(f"Could not locate pbwt library: {pbwt_path}")

    ref_base_path = Path(args.refpanel).resolve()

    # Prepare reference panel if indicated
    if args.prepare_reference:
        if args.ref_source_vcf:
            ref_source_path = Path(args.ref_source_vcf).resolve()
            source_type = "vcf"
        elif args.ref_source_xsi:
            ref_source_path = Path(args.ref_source_xsi).resolve()
            source_type = "xsi"
        else:
            raise KeyError(
                "Reference panel path (--ref_source_vcf or --ref_source_xsi) "
                "required with --prepare_reference"
            )

        if not ref_source_path.exists():
            raise FileNotFoundError(
                f"Could not locate reference source file: {ref_source_path}"
            )
        # Create .pbwt and .srp files
        logger.info("Preparing reference panel files")
        if source_type == "vcf":
            subprocess.run(
                [
                    pbwt_path,
                    "-checkpoint",
                    "100000",
                    "-readVcfGT",
                    ref_source_path,
                    "-writeAll",
                    ref_base_path,
                ],
                check=True,
            )
            SparseReferencePanel(str(add_suffix(ref_base_path, ".srp"))).from_bcf(
                str(ref_source_path), threads=args.cores
            )
        else:
            subprocess.run(
                f"xsqueezeit -x -Ov -f {ref_source_path} | "
                f"{pbwt_path} -checkpoint 100000 -readVcfGT - -writeAll {ref_base_path}",
                check=True,
                shell=True,
            )
            SparseReferencePanel(str(add_suffix(ref_base_path, ".srp"))).from_xsi(
                str(ref_source_path), threads=args.cores
            )
        logger.info(f"\n{timestamp()}: Reference panel files saved to {ref_base_path}")
        if not args.target:
            sys.exit(0)

    if not args.target:
        raise KeyError("Path to target vcf/bcf required for imputation")
    targets_path = Path(args.target).resolve()
    if not args.map:
        raise KeyError("Path to genetic map required for imputation")
    map_path = Path(args.map).resolve()
    output_path = Path(args.outvcf).resolve() if args.outvcf else Path(thedir)

    with TemporaryDirectory(dir=args.tmp_path) as tempdir:
        selphi(
            targets_path=targets_path,
            ref_base_path=ref_base_path,
            genetic_map_path=map_path,
            output_path=output_path,
            pbwt_path=pbwt_path,
            tmpdir=Path(tempdir).resolve(),
            match_length=args.match_length,
            est_ne=args.est_ne,
            cores=args.cores,
            adjust_cores=not args.no_core_reduction,
        )
