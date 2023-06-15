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
