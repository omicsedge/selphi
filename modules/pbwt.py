import subprocess
from pathlib import Path
import shutil
from typing import List
from tempfile import mkdtemp
from uuid import uuid4
from math import ceil
import hashlib

from joblib import Parallel, delayed


def _md5sum(filepath: Path) -> str:
    with open(filepath, "rb") as fin:
        return hashlib.md5(fin.read()).hexdigest()


def _run_subset_pbwt(
    pbwt_path: Path,
    targets_path: Path,
    samples: List[str],
    ref_base_path: Path,
    tmpdir: Path,
    match_length: int = 5,
) -> str:
    tmpdir = Path(tmpdir).joinpath(str(uuid4()))
    with open(tmpdir.joinpath("samples.txt"), "w") as fout:
        fout.write("\n".join(samples))
    subprocess.run(
        [
            pbwt_path,
            "-readVcfGT",
            str(targets_path),
            "-selectSamples",
            str(tmpdir.joinpath("samples.txt")),
            "-referenceMatch",
            str(ref_base_path),
            match_length,
        ],
        check=True,
        cwd=tmpdir,
    )
    return tmpdir


def get_pbwt_matches(
    pbwt_path: Path,
    targets_path: Path,
    ref_base_path: Path,
    target_samples: List[str],
    match_length: int = 5,
    cores: int = 1,
) -> Path:
    """
    Use selfdecode/pbwt library to get minimum length matches
    between target haplotypes and reference panel haplotypes
    and save the matches in sparse matrix npz format.
    """
    tmpdir = Path(mkdtemp())
    n_samples = len(target_samples)
    if cores == 1 or n_samples == 1:
        # Process all samples at once and return
        subprocess.run(
            [
                pbwt_path,
                "-readVcfGT",
                str(targets_path),
                "-referenceMatch",
                str(ref_base_path),
                match_length,
            ],
            check=True,
            cwd=tmpdir,
        )
        return tmpdir

    # Run pbwt in parallel if multiple samples and multiple cores
    batch_size = ceil(n_samples // cores)
    # Filter reference pbwt so we only have to do it once
    subprocess.run(
        [
            pbwt_path,
            "-readVcfGT",
            str(targets_path),
            "-writeSites",
            str(tmpdir.joinpath("chipsites.txt")),
        ],
        check=True,
        cwd=tmpdir,
    )
    subprocess.run(
        [
            pbwt_path,
            "-readAll",
            str(ref_base_path),
            "-selectSites",
            str(tmpdir.joinpath("chipsites.txt")),
            "-writeAll",
            str(tmpdir.joinpath("filtered_reference")),
        ],
        check=True,
        cwd=tmpdir,
    )
    dirlist: List[Path] = Parallel(n_jobs=cores,)(
        delayed(_run_subset_pbwt)(
            pbwt_path,
            targets_path,
            target_samples[start : start + batch_size],
            str(tmpdir.joinpath("filtered_reference")),
            tmpdir,
            match_length,
        )
        for start in range(0, n_samples, batch_size)
    )
    # Make sure all batch variants files are the same
    checksums = [_md5sum(dir_.joinpath("variants.txt")) for dir_ in dirlist]
    assert len(set(checksums)) == 1
    # Consolidate files
    shutil.move(str(dirlist[0].joinpath("variants.txt")), str(tmpdir))
    for file in tmpdir.glob("*/*.npz"):
        shutil.move(str(file), str(tmpdir))
    return tmpdir
