import math
import itertools
from tqdm import trange

import numpy as np

def get_transition_matrix(
    obs,
    states_1,
    states_2,
    chip_positions_dedup,
    num_hid=6400,
    ne=1000000,
    reverse=False
    ):
    """
    Transition Matrix for the forward-backward algorithm
    """
    # Transition Matrix
    transition = np.zeros((num_hid,num_hid))
    intersection_set = set.intersection(set(states_1), set(states_2))
    intersection_list = list(intersection_set)

    N = 20000
    dm = (
        int(chip_positions_dedup[obs+1]) - int(chip_positions_dedup[obs])
    ) * (1/1000000)
    tao_m = (1 - math.exp((-0.04 * ne*dm)/N))

    pos = itertools.product(states_1, states_2)

    rows, cols = zip(*pos)
    if reverse:
        transition[cols, rows] = (tao_m/N)
    else:
        transition[rows, cols] = (tao_m/N)

    transition[intersection_list,intersection_list] = 1-tao_m + (tao_m/N)
        
    return transition

def create_composite_chip_panel(all_haps, num_chip_vars=14778):
    """
    Create the chip panel that will be used to create the transition matrix
    """
    composite_chip_panel = np.zeros((len(all_haps), num_chip_vars)).astype(np.int16)
    composite_chip_panel_matches = np.zeros((len(all_haps), num_chip_vars)).astype(np.int16)
    for c, haploid_segments in enumerate(all_haps):
        for hap, current_index, matches in haploid_segments:
            composite_chip_panel[c, current_index:current_index+matches] = hap
            composite_chip_panel_matches[c, current_index:current_index+matches] = matches
    return composite_chip_panel, composite_chip_panel_matches

def forward(num_obs, ordered_matches, chip_positions_dedup):
    """
    V is observations array
    """
    start = 0
    end = 1000

    # initial_distribution = np.full((6400,), 1/6400)
    alpha = np.zeros((num_obs, 6400))
    alpha[start, ordered_matches[0]] = 1/len(ordered_matches[0])
    # alpha[start, :] = initial_distribution
 
    for t in trange(start+1, end):
        states_1 = ordered_matches[t-1]
        states_2 = ordered_matches[t]
        a = get_transition_matrix(
            t-1,
            states_1,
            states_2,
            chip_positions_dedup,
            reverse=True,
        )
        # a = a.T
        for j in range(a.shape[0]):
            # print('new')
            alpha[t, j] = alpha[t - 1,:].dot(a[j, :])
        if sum(alpha[t,:]) == 0:
            print("sumation 0")
            alpha[t, :] = 1/a.shape[0]
        else:
            alpha[t, :] = alpha[t, :]/sum(alpha[t,:])

    return alpha

def backward(num_obs, ordered_matches, chip_positions_dedup):
    start = 0
    end = 1000

    # T = 1000
    beta = np.zeros((num_obs, 6400))
    

    # setting beta(T) = 1
    # beta[end - 1, :] = np.zeros((6400))
    beta[end - 1, ordered_matches[end -1]] = 1/len(ordered_matches[end -1])
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in trange(end - 2, start -1, -1):
        states_1 = ordered_matches[t]
        states_2 = ordered_matches[t +1]
        a = get_transition_matrix(
            t,
            states_1,
            states_2,
            chip_positions_dedup
        )
        # a = a.T``
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1,:]).dot(a[j, :])
        if sum(beta[t,:]) == 0:
            print("sumation 0")
            beta[t, :] = 1/a.shape[0]
        else:
            beta[t, :] = beta[t, :]/sum(beta[t,:])

    return beta
