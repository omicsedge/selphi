from typing import List, Tuple, Union
import numpy as np
import zarr
import matplotlib.pyplot as plt
from tqdm import trange





print("don't run this file directly")
print("load the necessary data from the Selphi notebook, then run the stuff below")




full_BP_list = [int(x.split('-')[1]) for x in full_id_list] 
full_BP_arr = np.array(full_BP_list)

EUROFIN_sites = np.genfromtxt('/mnt/science-shared/data/hmm-phasing/files/chrBP_list_EUROFINS.txt', dtype=np.int64, delimiter='\t')[:,1]



chip_BP_positions_arr = np.array(chip_BP_positions)
chip_BP_positions_set = set(chip_BP_positions_arr)
EUROFIN_sites_set = set(EUROFIN_sites)

chip_BP_positions_set.difference(EUROFIN_sites_set)

EUROFIN_sites_set.difference(chip_BP_positions_set)





full_ref_panel_array = np.zeros((chr_length, ref_panel_full_array_full_packed.shape[1]), dtype=np.uint8)

for i in trange(0,ref_panel_full_array_full_packed.shape[1]): 
    full_ref_panel_array[:,i] = np.unpackbits(ref_panel_full_array_full_packed[:,i])[:chr_length] 

original_indicies_arr = np.array(original_indicies)


# CALCULATE NUMBER OF UNIQ HAPLOTYPES IN EACH INTERVAL BETWEEN ADJACENT CHIP SITES #
num_of_unique_haplotypes = np.zeros(original_indicies_arr.shape[0] + 1, dtype=np.int64)
oi_from = -1
oi_to = -1
for i in trange(0,len(original_indicies_arr)):
    oi = original_indicies_arr[i]
    oi_to = oi
    num_of_unique_haplotypes[i] = np.unique(full_ref_panel_array[oi_from+1:oi_to], axis=1).shape[1]
    oi_from = oi+1
num_of_unique_haplotypes[-1] = np.unique(full_ref_panel_array[oi_from+1:], axis=1).shape[1]


# CALCULATE ALLELE FREQUENCIES IN THE REFERENCE PANEL #
full_ref_panel_array_MAF: np.ndarray = np.mean(full_ref_panel_array, axis=1) # type: ignore



# CALCULATE NUMBER OF RARE ALLELES (<5% freq) #
num_of_MAFs_below_5perc = np.zeros(original_indicies_arr.shape[0] + 1, dtype=np.int64)
oi_from = -1
oi_to = -1
for i in trange(0,len(original_indicies_arr)):
    oi = original_indicies_arr[i]
    oi_to = oi
    num_of_MAFs_below_5perc[i] = (full_ref_panel_array_MAF[oi_from+1:oi_to] < 0.05).sum()
    oi_from = oi+1
num_of_MAFs_below_5perc[-1] = (full_ref_panel_array_MAF[oi_from+1:] < 0.05).sum()



# CALCULATE NUMBER OF RARE ALLELES (<1% freq) #
num_of_MAFs_below_1perc = np.zeros(original_indicies_arr.shape[0] + 1, dtype=np.int64)
oi_from = -1
oi_to = -1
for i in trange(0,len(original_indicies_arr)):
    oi = original_indicies_arr[i]
    oi_to = oi
    num_of_MAFs_below_1perc[i] = (full_ref_panel_array_MAF[oi_from+1:oi_to] < 0.01).sum()
    oi_from = oi+1
num_of_MAFs_below_1perc[-1] = (full_ref_panel_array_MAF[oi_from+1:] < 0.01).sum()



# EXCLUDE SINGLETONS FROM MAF ARRAY #
full_ref_panel_array_MAF_excl_singletons = np.copy(full_ref_panel_array_MAF)
full_ref_panel_array_MAF_excl_singletons[np.where( full_ref_panel_array_MAF < (1.1/full_ref_panel_array_MAF.shape[0]) )] = 1

# CALCULATE NUMBER OF RARE ALLELES (<5% freq) EXCLUDING SINGLETONS #
num_of_MAFs_below_5perc_excl_singletons = np.zeros(original_indicies_arr.shape[0] + 1, dtype=np.int64)
oi_from = -1
oi_to = -1
for i in trange(0,len(original_indicies_arr)):
    oi = original_indicies_arr[i]
    oi_to = oi
    num_of_MAFs_below_5perc_excl_singletons[i] = (full_ref_panel_array_MAF_excl_singletons[oi_from+1:oi_to] < 0.05).sum()
    oi_from = oi+1
num_of_MAFs_below_5perc_excl_singletons[-1] = (full_ref_panel_array_MAF_excl_singletons[oi_from+1:] < 0.05).sum()

# CALCULATE NUMBER OF RARE ALLELES (<1% freq) EXCLUDING SINGLETONS #
num_of_MAFs_below_1perc_excl_singletons = np.zeros(original_indicies_arr.shape[0] + 1, dtype=np.int64)
oi_from = -1
oi_to = -1
for i in trange(0,len(original_indicies_arr)):
    oi = original_indicies_arr[i]
    oi_to = oi
    num_of_MAFs_below_1perc_excl_singletons[i] = (full_ref_panel_array_MAF_excl_singletons[oi_from+1:oi_to] < 0.01).sum()
    oi_from = oi+1
num_of_MAFs_below_1perc_excl_singletons[-1] = (full_ref_panel_array_MAF_excl_singletons[oi_from+1:] < 0.01).sum()






### PLOT DISTRIBUTION OF N OF UNIQUE HAPLOTYPES ###

xlim_left, xlim_right = 0, full_ref_panel_array.shape[1]
xlim_left, xlim_right = 2, 1000

plt.hist(num_of_unique_haplotypes, bins=list(range(xlim_left, xlim_right+1)), histtype="step")  # type: ignore
# plt.title('beta field values distribution')
# plt.ylim(bottom=xlim_left, top=)
plt.ylabel(f"prevalence (from {len(num_of_unique_haplotypes)} intervals)")
plt.xlabel(f"n of unique haplotypes")
# legend = plt.legend(loc="upper right", edgecolor="black")
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((1, 1, 1, 0.3))

ax = plt.gca()
ax.tick_params(color=(0, 0, 0, 0.3))
for spine in ax.spines.values():
    spine.set_edgecolor((0, 0, 0, 0.3))

plt.xlim(left=xlim_left, right=xlim_right)

plt.savefig(
    f"distribution-of-n-of-unique-haplotypes-in-ref-panel-between-chip-sites_x={xlim_left},{xlim_right}.png",
    # facecolor=(1, 1, 1, 0.3),
    # transparent=True,
    dpi=300,
)
# plt.show()
plt.cla()
plt.clf()




### LOAD THE DUMPED num_of_unique_haplotypes ARRAY ###
# import pickle
# with open(
#     "SelfDecode_work/selphi-imputation/n-of-uniq-haplotypes-between-chip-sites.pkl",
#     "rb",
# ) as f:
#     num_of_unique_haplotypes = pickle.load(f)
import pickle
def pickle_dump(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
        return path
def pickle_load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# num_of_unique_haplotypes = pickle_load("n-of-uniq-haplotypes-between-chip-sites.pkl")


def group_average_array_resize(arr: np.ndarray, n: int):
    assert len(arr.shape) == 1, "1-d array should be passed"
    assert n > 0, "wtf u doing"

    if n == 1:
        return np.copy(arr)

    mod_n = len(arr) % n

    if mod_n == 0:
        return np.mean(arr.reshape(-1, n), axis=1)
    else:
        shortfall = n - mod_n

        extended_arr: np.ndarray = np.zeros(arr.shape[0] + shortfall)
        extended_arr[:arr.shape[0]] = arr[:]
        for i in range(arr.shape[0], extended_arr.shape[0]):
            extended_arr[i] = arr[-1]
        return np.mean(extended_arr.reshape(-1, n), axis=1)

def group_average_array(arr: np.ndarray, n: int):
    assert len(arr.shape) == 1, "1-d array should be passed"
    assert n > 0, "wtf u doing"

    if n == 1:
        return np.copy(arr)

    groups_n = np.int64(np.ceil(arr.shape[0] / n))
    newarr = np.copy(arr)

    for gi in range(groups_n-1):
        newarr[gi*n : gi*n + n] = np.average(arr[gi*n : gi*n + n])
    newarr[(groups_n-1)*n : ] = np.average(arr[(groups_n-1)*n : ])

    return newarr

def smooth_array(arr: np.ndarray, kernel_diameter: int):
    assert len(arr.shape) == 1, "1-d array should be passed"
    assert kernel_diameter % 2 == 1, "kernel_diameter has to be an odd integer"

    smooth_func = lambda subarr: subarr.sum()/subarr.shape[0]

    if kernel_diameter == 1:
        return np.copy(arr)

    kernel_radius = int((kernel_diameter + 1)/2)
    k = kernel_radius - 1

    newarr = np.zeros_like(arr)
    for i in range(0, k):
        newarr[i] = smooth_func(arr[:i+k+1])

    for i in range(k, arr.shape[0]-k):
        newarr[i] = smooth_func(arr[i-k : i+k+1])

    for i in range(arr.shape[0]-k, arr.shape[0]):
        newarr[i] = smooth_func(arr[i-k:])

    return newarr


def smooth_array_over_metric(arr: np.ndarray, custom_metric: np.ndarray, kernel_diameter: float):
    """
    smooth_array(arr, k) == smooth_array_over_metric(arr, np.arange(len(arr)), k)
    """
    assert len(arr.shape) == 1, "1-d array should be passed"
    assert arr.shape == custom_metric.shape

    if kernel_diameter == 0:
        return np.copy(arr)
    if kernel_diameter < 0:
        kernel_diameter = -kernel_diameter

    def interpolate_value_to_float_index_in_sorted_array(
        custom_metric: np.ndarray,
        metric_coordinate: float
    ) -> float:
        # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
        closest_gte_idx = np.searchsorted(custom_metric, metric_coordinate) # index of the closest value greater than or equal to `metric_coordinate`

        custom_metric_safe_index = lambda index: custom_metric[min(index, custom_metric.shape[0]-1)]

        if closest_gte_idx == 0:
            interval_length = custom_metric[1] - custom_metric[0]
        elif closest_gte_idx == custom_metric.shape[0]:
            closest_gte_idx -= 1
            interval_length = custom_metric[-1] - custom_metric[-2]
        else:
            interval_length = custom_metric[closest_gte_idx] - custom_metric[closest_gte_idx-1]

        """interpolate linearly"""
        interpolated_idx = closest_gte_idx + (
            metric_coordinate - custom_metric_safe_index(closest_gte_idx)
                                        ) / interval_length

        return interpolated_idx

    def sum_intervals_values_within_float_range(arr: np.ndarray, i_from: float, i_to: float):
        """
        arr - array of interval values.
            Meaning value held at arr[i] is a value assigned to the interval (i-0.5; i+0.5)

        if `a` is an integer, then
        sum_intervals_values_within_float_range(arr, a-0.5, a+0.5) == arr[a]

        if `a` and `b` are integers, then
        sum_intervals_values_within_float_range(arr, a-0.5, b-0.5) == sum(arr[a:b])

        Assumes always:
            - len(arr.shape) == 1
            - i_from > 0
            - i_to < arr.shape[0]
            - i_from <= i_to
        """

        i_from += 0.5
        i_to += 0.5

        ceil_i_from = np.ceil(i_from).astype(np.int64)
        floor_i_to = np.floor(i_to).astype(np.int64)

        if ceil_i_from <= floor_i_to:
            whole_part = arr[ceil_i_from:floor_i_to].sum()
            left_fractional_part = (ceil_i_from-i_from)*arr[max(ceil_i_from-1, 0)]
            right_fractional_part = (i_to-floor_i_to)*arr[min(floor_i_to, arr.shape[0]-1)]

            return left_fractional_part + whole_part + right_fractional_part
        else:
            # for cases when both i_to and i_from are between two adjancent integers,
            # i.e. n <= i_from <= i_to <= n+1,
            # there's no "whole", "left fractional", and "right fractional" parts,
            # there's only one middle fractional part
            return (i_to-i_from)*arr[min(floor_i_to, arr.shape[0]-1)]


    k = kernel_radius = (kernel_diameter/2)
    newarr = np.zeros_like(arr).astype(np.float64)

    for i in range(0, arr.shape[0]):
        i_from = interpolate_value_to_float_index_in_sorted_array(custom_metric, custom_metric[i] - kernel_radius)
        i_to   = interpolate_value_to_float_index_in_sorted_array(custom_metric, custom_metric[i] + kernel_radius)
        i_from, i_to = max(-0.5, i_from), min(i_to, custom_metric.shape[0]-1-0.5)

        if i_to != i_from:            
            newarr[i] = sum_intervals_values_within_float_range(arr, i_from, i_to) / (i_to - i_from)

    return newarr



### PLOT N OF UNIQUE HAPLOTYPES ACROSS THE CHR ###

# ylim_left, ylim_right = 0, full_ref_panel_array.shape[1]
ylim_bottom, ylim_top = 0, 6404

# plt.hist(
#     num_of_unique_haplotypes,
#     bins=list(range(xlim_left, xlim_right)),
#     histtype="step"
# )  # type: ignore


def plot_with_smoothings(
    X: np.ndarray,
    Y: np.ndarray,
    xlabel: str,
    ylabel: str,
    colors: Tuple[str, str, str] = ('#cfcf00', '#ff7f007f', '#ff00003f'),
    title: str = '',
):

    ax = plt.gca()
    ax.tick_params(color=(0, 0, 0, 0.3))
    for spine in ax.spines.values():
        spine.set_edgecolor((0, 0, 0, 0.3))

    xlim_left = X[0]
    xlim_right = X[-1]
    xrange = X[-1] - X[0]

    plt.plot(
        X,
        Y,
        # alpha=0.3,
        linewidth=1,
        color=colors[0]
    )

    plt.plot(
        X,
        smooth_array_over_metric(
            Y,
            custom_metric=X,
            kernel_diameter=xrange/300
        ),
        # alpha=0.3,
        linewidth=5,
        color=colors[1]
    )

    plt.plot(
        X,
        smooth_array_over_metric(
            Y,
            custom_metric=X,
            kernel_diameter=xrange/30
        ),
        # alpha=0.3,
        linewidth=10,
        # linewidth=50,# should be around 50 to match the scale of smoothing
        color=colors[2]
    )

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.xlim(left=xlim_left, right=xlim_right)
    plt.ylim(bottom=ylim_bottom, top=ylim_top)

    plt.title(title)

    # plt.savefig(
    #     f"distribution-of-n-of-unique-haplotypes-in-ref-panel-between-chip-sites_x={xlim_left},{xlim_right}.png",
    #     # facecolor=(1, 1, 1, 0.3),
    #     # transparent=True,
    #     dpi=300,
    # )
    plt.show()
    plt.cla()
    plt.clf()


plot_with_smoothings(
    X = np.arange(int(len(num_of_unique_haplotypes))),
    Y = num_of_unique_haplotypes,
    xlabel = f"interval #",
    ylabel = f"n of unique haplotypes",
    colors = ('#cfcf00', '#ff7f007f', '#ff00003f')
)

full_BP_positions, full_cM_coordinates = load_and_interpolate_genetic_map( 
    genetic_map_path = f"/home/nikita/s3-sd-platform-dev/efs/service/genome-files-processing/resource_files/genetic_map_plink/plink.{chrom}.GRCh38.map", 
    chip_id_list = full_id_list, 
)

chip_BP_positions_arr = np.array(chip_BP_positions)
full_BP_positions_arr = np.array(full_BP_positions)
chip_cM_coordinates_arr = np.array(chip_cM_coordinates)
full_cM_coordinates_arr = np.array(full_cM_coordinates)

# indices of chip variants in the list of full sequence variants
original_indicies_arr = np.array(original_indicies)


def points_list_to_intervals_middle_coordinates_list(
    list_of_points: np.ndarray,
    beginning_coordinate,
    end_coordinate
):
    list_of_points_w_ends = np.zeros(list_of_points.shape[0] + 2)
    list_of_points_w_ends[0]     = beginning_coordinate
    list_of_points_w_ends[1:-1]  = list_of_points[:]
    list_of_points_w_ends[-1]    = end_coordinate

    intervals_middle_coordinates: np.ndarray = np.zeros(list_of_points_w_ends.shape[0] - 1)
    intervals_middle_coordinates = (list_of_points_w_ends[1:] + list_of_points_w_ends[:-1])/2
    assert len(intervals_middle_coordinates) == len(list_of_points) + 1

    return intervals_middle_coordinates

chip_BP_positions_intervals_coordinates = points_list_to_intervals_middle_coordinates_list(
    list_of_points       = chip_BP_positions_arr,
    beginning_coordinate = full_BP_positions_arr[0],
    end_coordinate       = full_BP_positions_arr[-1],
).astype(np.int64) # type: ignore # pylance curses at .astype ??

chip_cM_coordinates_intervals_coordinates = points_list_to_intervals_middle_coordinates_list(
    list_of_points       = chip_cM_coordinates_arr,
    beginning_coordinate = full_cM_coordinates_arr[0],
    end_coordinate       = full_cM_coordinates_arr[-1],
)

chip_indices_intervals_coordinates = points_list_to_intervals_middle_coordinates_list(
    list_of_points=original_indicies_arr,
    beginning_coordinate = 0,
    end_coordinate       = len(full_BP_positions_arr) - 1,
).astype(np.int64) # type: ignore # pylance curses at .astype ??

plot_with_smoothings(
    X = chip_indices_intervals_coordinates,
    Y = num_of_unique_haplotypes,
    xlabel = f"full sequence variant #",
    ylabel = f"n of unique haplotypes",
    colors = ('#cfcf00', '#ff7f007f', '#ff00003f')
)

plot_with_smoothings(
    X = chip_BP_positions_intervals_coordinates/1_000_000,
    Y = num_of_unique_haplotypes,
    xlabel = f"physical coordinate (MBP)",
    ylabel = f"n of unique haplotypes",
    colors = ('#cfcf00', '#ff7f007f', '#ff00003f')
)

plot_with_smoothings(
    X = chip_cM_coordinates_intervals_coordinates,
    Y = num_of_unique_haplotypes,
    xlabel = f"genetic coordinate (cM)",
    ylabel = f"n of unique haplotypes",
    colors=('#cfcf00', '#ff7f007f', '#ff00003f')
)







ylim_bottom, ylim_top = 0, 3000

plot_with_smoothings(
    X = np.arange(int(len(num_of_MAFs_below_5perc))),
    Y = num_of_MAFs_below_5perc,
    xlabel = f"interval #",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f')
)
plot_with_smoothings(
    X = chip_indices_intervals_coordinates,
    Y = num_of_MAFs_below_5perc,
    xlabel = f"full sequence variant #",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f')
)
plot_with_smoothings(
    X = chip_BP_positions_intervals_coordinates/1_000_000,
    Y = num_of_MAFs_below_5perc,
    xlabel = f"physical coordinate (MBP)",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f')
)
plot_with_smoothings(
    X = chip_cM_coordinates_intervals_coordinates,
    Y = num_of_MAFs_below_5perc,
    xlabel=f"genetic coordinate (cM)",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f')
)


# plot_with_smoothings(
#     X = np.arange(int(len(num_of_MAFs_below_1perc))),
#     Y = num_of_MAFs_below_1perc,
#     xlabel = f"interval #",
#     ylabel = f"n rare variants w\\ MAF<1%",
#     colors = ('#2f6fef', '#006f8f7f', '#004f4f3f')
# )

plot_with_smoothings(
    X = np.arange(int(len(num_of_MAFs_below_1perc))),
    Y = num_of_MAFs_below_1perc,
    xlabel = f"interval #",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f')
)
plot_with_smoothings(
    X = chip_indices_intervals_coordinates,
    Y = num_of_MAFs_below_1perc,
    xlabel = f"full sequence variant #",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f')
)
plot_with_smoothings(
    X = chip_BP_positions_intervals_coordinates/1_000_000,
    Y = num_of_MAFs_below_1perc,
    xlabel = f"physical coordinate (MBP)",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f')
)
plot_with_smoothings(
    X = chip_cM_coordinates_intervals_coordinates,
    Y = num_of_MAFs_below_1perc,
    xlabel=f"genetic coordinate (cM)",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f')
)








plot_with_smoothings(
    X = np.arange(int(len(num_of_MAFs_below_5perc_excl_singletons))),
    Y = num_of_MAFs_below_5perc_excl_singletons,
    xlabel = f"interval #",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f'),
    title = f'excluding singletons',
)
plot_with_smoothings(
    X = chip_indices_intervals_coordinates,
    Y = num_of_MAFs_below_5perc_excl_singletons,
    xlabel = f"full sequence variant #",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f'),
    title = f'excluding singletons',
)
plot_with_smoothings(
    X = chip_BP_positions_intervals_coordinates/1_000_000,
    Y = num_of_MAFs_below_5perc_excl_singletons,
    xlabel = f"physical coordinate (MBP)",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f'),
    title = f'excluding singletons',
)
plot_with_smoothings(
    X = chip_cM_coordinates_intervals_coordinates,
    Y = num_of_MAFs_below_5perc_excl_singletons,
    xlabel=f"genetic coordinate (cM)",
    ylabel = f"n rare variants w\\ MAF<5%",
    colors = ('#00cfcf', '#009f5f7f', '#004f003f'),
    title = f'excluding singletons',
)


plot_with_smoothings(
    X = np.arange(int(len(num_of_MAFs_below_1perc_excl_singletons))),
    Y = num_of_MAFs_below_1perc_excl_singletons,
    xlabel = f"interval #",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f'),
    title = f'excluding singletons',
)
plot_with_smoothings(
    X = chip_indices_intervals_coordinates,
    Y = num_of_MAFs_below_1perc_excl_singletons,
    xlabel = f"full sequence variant #",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f'),
    title = f'excluding singletons',
)
plot_with_smoothings(
    X = chip_BP_positions_intervals_coordinates/1_000_000,
    Y = num_of_MAFs_below_1perc_excl_singletons,
    xlabel = f"physical coordinate (MBP)",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f'),
    title = f'excluding singletons',
)
plot_with_smoothings(
    X = chip_cM_coordinates_intervals_coordinates,
    Y = num_of_MAFs_below_1perc_excl_singletons,
    xlabel=f"genetic coordinate (cM)",
    ylabel = f"n rare variants w\\ MAF<1%",
    colors = ('#00afef', '#007f9f7f', '#002f2f3f'),
    title = f'excluding singletons',
)

