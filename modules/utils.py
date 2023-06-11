import subprocess
from io import StringIO
import pandas as pd
import numpy as np


def get_std(avg_length: np.ndarray, a: int = 25) -> np.ndarray:
    # convert average length into number on x axis
    rmin_ = 10000
    rmax_ = 1
    tmin_ = 0
    tmax_ = 1
    avg_length = ((avg_length - rmin_) / (rmax_ - rmin_)) * (tmax_ - tmin_) + tmin_

    std_not_normed = a * avg_length ** (a - 1.0)

    rmin_ = 0
    rmax_ = a * 1 ** (a - 1.0)
    tmin_ = 0.2
    tmax_ = 3
    return ((std_not_normed - rmin_) / (rmax_ - rmin_)) * (tmax_ - tmin_) + tmin_


def get_variants_bcf(input_bcfgz_path):
    """
    return a list of variants in the provided bcf
    """
    cmd_str = f'bcftools view {input_bcfgz_path} | grep -v "##" '
    result = subprocess.run(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
    )
    s = str(result.stdout, "utf-8")
    data = StringIO(s)
    df = pd.read_table(
        data,
        usecols=[
            "ID",
        ],
    )
    return list(df["ID"])


def get_samples_bcf(input_bcfgz_path):
    """
    return a list of samples in the provided bcf
    """
    cmd_str = f"bcftools query -l {input_bcfgz_path}"
    result = subprocess.run(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
    )
    s = str(result.stdout, "utf-8")
    data = StringIO(s)
    df = pd.read_table(data, header=None)
    return list(df[0])


def get_samples_xsi(xsi_file):
    """
    return a list of samples in the provided xsi files
    """
    cmd_str = f"xsqueezeit -x -r 0 -f {xsi_file} | bcftools query -l"
    result = subprocess.run(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
    )
    s = str(result.stdout, "utf-8")
    data = StringIO(s)
    df = pd.read_table(data, header=None)
    return list(df[0])
