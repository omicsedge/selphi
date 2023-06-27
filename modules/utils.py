from datetime import datetime, timezone
from contextlib import contextmanager

import numpy as np
import joblib
from tqdm import tqdm


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
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

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
