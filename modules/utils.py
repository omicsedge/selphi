from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numba import njit
import joblib
from tqdm import tqdm


@njit
def get_std(
    avg_length: np.ndarray, min_length: int, max_length: int, a: int = 25
) -> np.ndarray:
    # convert average length into number on x axis
    rmin = max_length
    rmax = min_length
    avg_length = (avg_length - rmin) / (rmax - rmin)

    std_not_normed = a * avg_length ** (a - 1.0)

    rmax_ = a * 1 ** (a - 1.0)
    tmin_ = 0.2
    tmax_ = 3
    return (std_not_normed / rmax_) * (tmax_ - tmin_) + tmin_


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%-I:%M:%S %p")


@contextmanager
def tqdm_joblib(*args, **kwargs):
    """
    Context manager to patch joblib to report into tqdm progress bar
    given as argument from https://github.com/louisabraham/tqdm_joblib
    """
    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_version() -> str:
    """Retrieve version from pyproject.toml"""
    lines = Path(__file__).parents[1].joinpath("pyproject.toml").read_text().splitlines()
    for line in lines:
        if line.startswith("version"):
            return line.split('"')[-2]
    else:
        raise Warning("Could not determine version")
