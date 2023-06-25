import subprocess
from pathlib import Path
import shutil
from typing import List
from uuid import uuid4
from math import ceil
from logging import Logger

from joblib import Parallel, delayed
import numpy as np

from modules.utils import timestamp


def _run_subset_pbwt(
    pbwt_path: Path,
    samples: List[str],
    tmpdir: Path,
    match_length: int = 5,
) -> Path:
    subtmpdir = Path(tmpdir).joinpath(str(uuid4()))
    subtmpdir.mkdir(parents=True, exist_ok=True)
    subtmpdir.joinpath("samples.txt").write_text("\n".join(samples) + "\n")
    subprocess.run(
        [
            pbwt_path,
            "-readAll",
            tmpdir.joinpath("filtered_target"),
            "-selectSamples",
            subtmpdir.joinpath("samples.txt"),
            "-referenceMatch",
            tmpdir.joinpath("filtered_reference"),
            str(match_length),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        cwd=subtmpdir,
    )
    return subtmpdir


def get_pbwt_matches(
    pbwt_path: Path,
    targets_path: Path,
    ref_base_path: Path,
    tmpdir: Path,
    target_samples: List[str],
    ref_filter: np.ndarray,
    target_filter: np.ndarray,
    logger: Logger,
    match_length: int = 5,
    cores: int = 1,
) -> Path:
    """
    Use selfdecode/pbwt library to get minimum length matches
    between target haplotypes and reference panel haplotypes
    and save the matches in sparse matrix npz format.
    """
    logger.info(f"{timestamp()}: Filtering reference panel to match target variants")
    np.savetxt(tmpdir.joinpath("ref_filter.txt"), ref_filter, fmt="%u")
    subprocess.run(
        [
            pbwt_path,
            "-readAll",
            ref_base_path,
            "-filterSites",
            tmpdir.joinpath("ref_filter.txt"),
            "-writeAll",
            tmpdir.joinpath("filtered_reference"),
        ],
        check=True,
        cwd=tmpdir,
    )
    logger.info(f"{timestamp()}: Filtering target samples to match reference variants")
    np.savetxt(tmpdir.joinpath("target_filter.txt"), target_filter, fmt="%u")
    subprocess.run(
        [
            pbwt_path,
            "-readVcfGT",
            targets_path,
            "-filterSites",
            tmpdir.joinpath("target_filter.txt"),
            "-writeAll",
            tmpdir.joinpath("filtered_target"),
        ],
        check=True,
        cwd=tmpdir,
    )
    n_samples = len(target_samples)
    if cores == 1 or n_samples == 1:
        # Process all samples at once and return
        logger.info(f"{timestamp()}: Matching {n_samples} sample(s) to reference panel")
        subprocess.run(
            [
                pbwt_path,
                "-readAll",
                tmpdir.joinpath("filtered_target"),
                "-referenceMatch",
                tmpdir.joinpath("filtered_reference"),
                str(match_length),
            ],
            check=True,
            cwd=tmpdir,
        )
        return tmpdir

    # Run pbwt in parallel if multiple samples and multiple cores
    batch_size = ceil(n_samples / cores)
    # Filter reference pbwt so we only have to do it once
    logger.info(
        f"{timestamp()}: Matching {n_samples} sample(s) to reference panel "
        f"in batches of {batch_size} samples"
    )
    _ = Parallel(n_jobs=cores)(
        delayed(_run_subset_pbwt)(
            pbwt_path,
            target_samples[start : start + batch_size],
            tmpdir,
            match_length,
        )
        for start in range(0, n_samples, batch_size)
    )
    # Consolidate files
    for file in tmpdir.glob("*/*.npz"):
        shutil.move(str(file), str(tmpdir))
    logger.info(f"{timestamp()}: Matched all samples to reference panel")
    return tmpdir
