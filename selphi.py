import os
import sys
import subprocess
from pathlib import Path
import time
import argparse
import logging
from tempfile import TemporaryDirectory
from joblib import Parallel, delayed

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

    start_time = time.time()

    # Load target samples
    vcf_obj = cyvcf2.VCF(targets_path)
    target_samples = vcf_obj.samples
    del vcf_obj
    target_haps = [(sample, hap) for sample in target_samples for hap in (0, 1)]

    # Find matches to reference panel with pbwt
    logger.info(f"Matching {len(target_samples)} sample(s) to reference panel")
    pbwt_result_path: Path = get_pbwt_matches(
        pbwt_path,
        targets_path,
        ref_base_path,
        tmpdir,
        target_samples,
        match_length,
        cores,
    )
    logger.info("Time elapsed: %s seconds" % (time.time() - start_time))
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
    logger.info("Calculating weights from reference matches")
    ordered_weights = np.asarray(
        Parallel(n_jobs=cores, verbose=2)(
            delayed(calculate_weights)(
                target_hap, chip_cM_coordinates, pbwt_result_path
            )
            for target_hap in target_haps
        ),
        dtype=object,
    )

    # Interpolate genotypes
    logger.info("Interpolating genotypes at missing variants")
    output_file = interpolate_genotypes(
        ref_base_path.with_suffix(".srp"),
        chip_variants,
        target_samples,
        ordered_weights,
        output_path,
        cores,
    )
    logger.info(f"Saved imputed genotypes to {output_file}")
    logger.info("===== Total time: %s seconds =====" % (time.time() - start_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Selphi", description="PBWT for genotype imputation"
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
        "--pbwt_path", type=str, default=".", help="path to pbwt library"
    )
    parser.add_argument(
        "--tmp_path", type=str, help="location to create temporary directory"
    )
    parser.add_argument(
        "--match_length", type=int, default=5, help="minimum pbwt match length"
    )
    args = parser.parse_args()

    thedir = os.path.dirname(os.path.abspath(__file__))

    pbwt_path = Path(args.pbwt_path).resolve()
    if pbwt_path.is_dir():
        pbwt_path = pbwt_path.joinpath("pbwt")
    if not pbwt_path.exists():
        raise FileNotFoundError(f"Could not locate pbwt library: {pbwt_path}")

    ref_path = Path(args.refpanel)
    ref_suffixes = ref_path.suffixes
    suffix_length = len(ref_suffixes) + sum([len(ext) for ext in ref_suffixes])
    if suffix_length > 0:
        ref_base_path = Path(args.refpanel[:-suffix_length]).resolve()
    else:
        ref_base_path = ref_path.resolve()

    # Prepare reference panel if indicated
    if args.prepare_reference:
        if not args.ref_source_vcf:
            raise KeyError(
                "Reference panel vcf/bcf path (--ref_source_vcf) "
                "required with --prepare_reference"
            )
        ref_source_path = Path(args.ref_source_vcf).resolve()
        if not ref_source_path.exists():
            raise FileNotFoundError(
                f"Could not locate reference vcf/bcf: {ref_source_path}"
            )
        # Create .pbwt files
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
        # Create srp file
        SparseReferencePanel(str(ref_base_path.with_suffix(".srp"))).from_bcf(
            str(ref_source_path), threads=args.cores
        )
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
