import os
import sys
import subprocess
from pathlib import Path
import argparse
import logging
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import cyvcf2

from modules.imputation_lib import calculate_weights
from modules.interpolation import interpolate_genotypes
from modules.load_data import load_and_interpolate_genetic_map
from modules.pbwt import get_pbwt_matches
from modules.sparse_ref_panel import SparseReferencePanel

logger = logging.getLogger("Selphi")
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def selphi(
    targets_path: Path,
    ref_base_path: Path,
    genetic_map_path: Path,
    output_path: Path,
    pbwt_path: Path,
    tmpdir: Path,
    match_length: int = 5,
    cores: int = 1,
):
    # Check if required files are available
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing target vcf/bcf: {targets_path}")
    if not ref_base_path.with_suffix(".pbwt").exists():
        raise FileNotFoundError(
            f"Missing pbwt reference files: {ref_base_path.with_suffix('.pbwt')}. "
            "Use --prepare_reference to create required reference files."
        )
    if not ref_base_path.with_suffix(".srp").exists():
        raise FileNotFoundError(
            f"Missing pbwt reference files: {ref_base_path.with_suffix('.srp')}. "
            "Use --prepare_reference to create required reference files."
        )
    if not genetic_map_path.exists():
        raise FileNotFoundError(f"Missing genetic map file: {genetic_map_path}")

    start_time = datetime.now()

    # Get number of reference samples and markers
    with open(ref_base_path.with_suffix(".samples"), "r") as fin:
        n_ref_samples = len(list(fin.readlines()))
    with open(ref_base_path.with_suffix(".sites"), "r") as fin:
        n_ref_markers = len(list(fin.readlines()))

    # Load target samples
    vcf_obj = cyvcf2.VCF(targets_path)
    target_samples = vcf_obj.samples
    for variant in vcf_obj:
        chrom = variant.CHROM
        break
    n_target_markers = len(list(vcf_obj)) + 1  # because we already looked at one
    del vcf_obj
    target_haps = [(sample, hap) for sample in target_samples for hap in (0, 1)]

    logger.info(
        "Stats:\n"
        f"Reference samples:\t\t{n_ref_samples}\n"
        f"Target samples:\t\t\t{len(target_samples)}\n\n"
        f"Reference markers:\t\t{n_ref_markers}\n"
        f"Target markers:\t\t\t{n_target_markers}\n\n"
        f"Chromosome: {chrom}\n\n"
        "Process:"
    )

    # Find matches to reference panel with pbwt
    pbwt_result_path: Path = get_pbwt_matches(
        pbwt_path,
        targets_path,
        ref_base_path,
        tmpdir,
        target_samples,
        logger,
        match_length,
        cores,
    )
    # Load overlapping variants
    variant_dtypes = np.dtype(
        [("chr", "<U21"), ("pos", int), ("ref", "<U21"), ("alt", "<U21")]
    )
    chip_variants = np.loadtxt(
        pbwt_result_path.joinpath("variants.txt"), dtype=variant_dtypes
    )
    logger.info(
        f"Loaded {chip_variants.size} variants found in both "
        "reference panel and target samples"
    )
    chip_BPs = [variant[1] for variant in chip_variants]
    chip_cM_coordinates: np.ndarray = load_and_interpolate_genetic_map(
        genetic_map_path=genetic_map_path,
        chip_BPs=chip_BPs,
    )
    logger.info("Loaded and interpolated genetic map")

    # Calculate HMM weights from matches
    ordered_weights = np.asarray(
        Parallel(n_jobs=cores)(
            delayed(calculate_weights)(
                target_hap, chip_cM_coordinates, pbwt_result_path
            )
            for target_hap in tqdm(
                target_haps,
                desc="Calculating weights with HMM",
                ncols=75,
                bar_format="{desc}:\t\t{percentage:3.0f}% {bar}\t{elapsed}",
                colour="#808080",
            )
        ),
        dtype=object,
    )

    # Interpolate genotypes
    output_file = interpolate_genotypes(
        ref_base_path.with_suffix(".srp"),
        chip_variants,
        target_samples,
        ordered_weights,
        output_path,
        cores,
    )
    logger.info(f"Saved imputed genotypes to {output_file}")
    logger.info(
        "===== Total time: %.2f seconds =====" % (datetime.now() - start_time).seconds
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Selphi (version 1.0)", description="PBWT for genotype imputation"
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
        "--match_length", type=int, default=5, help="minimum pbwt match length"
    )
    args = parser.parse_args()
    cmd = " \ \n".join([f"  --{k} {v}" for k, v in vars(args).items() if v is not None])

    thedir = os.path.dirname(os.path.abspath(__file__))

    logger.info(
        (
            "\nSelphi (version 1.0)\n"
            "Copyright (C) 2023 Selfdecode\nStart time: "
            f"{datetime.now(timezone.utc).strftime('%-I:%M %p %Z on %-d %b %Y')}\n\n"
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

    ref_suffixes = "".join(Path(args.refpanel).suffixes)
    ref_base_path = Path(args.refpanel.replace(ref_suffixes, "")).resolve()

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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            SparseReferencePanel(str(ref_base_path.with_suffix(".srp"))).from_bcf(
                str(ref_source_path), threads=args.cores
            )
        else:
            subprocess.run(
                f"xsqueezeit -x -Ov -f {ref_source_path} | "
                f"{pbwt_path} -checkpoint 100000 -readVcfGT - -writeAll {ref_base_path}",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            SparseReferencePanel(str(ref_base_path.with_suffix(".srp"))).from_xsi(
                str(ref_source_path), threads=args.cores
            )
        logger.info(f"Reference panel files saved to {ref_base_path}")
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
            cores=args.cores,
        )
