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

#-----------------------------------------------------
#Replicating Beagle
#-----------------------------------------------------

def pRecomb(
    obs,
    chip_positions_dedup,
    ordered_matches,
    BJ,
    distances_cm,
    num_hid=6400,
    ne=1000000,
    ):
    """
    pRecomb betweem obs and and obs-1
    """
    # param = max([len(np.unique(BJ[ordered_matches[x],x])) for x in range(obs,obs+1)])*2.5
    N = num_hid
    # dm = (
    #     int(chip_positions_dedup[obs]) - int(chip_positions_dedup[obs -1])
    # ) * (1/1000000)
    dm = distances_cm[obs] - distances_cm[obs-1]
    if dm == 0:
        dm = 0.0000001
    tao_m = (1 - math.exp((-0.04 *dm*ne)/N))
    
        
    return tao_m

def setFwdValues(num_obs, ordered_matches, chip_positions_dedup,BI, BJ,pnoerr_matrix,distances_cm):
    """
    set forward values
    """
    nRefHaps = 6400
    lastSum  = 1
    start = 0
    end = num_obs
    alpha = np.zeros((num_obs, nRefHaps))
    alpha[start, ordered_matches[0]] = 1/len(ordered_matches[0])
    nHaps = max([len(np.unique(ordered_matches[x])) for x in range(0,num_obs)])
    pErr = 0.0001
    pNoErr = 1 - pErr
    matches = np.roll(BI,1,axis=1) + BJ
    for m in trange(1, end):
        #startmymod
        # match_sum = []
        # _ = [match_sum.append(int(BJ[ordered_matches[m][x], m])) for x in range(0,len(ordered_matches[m]))]
        #endmymod
        pRecomb_var = pRecomb(m, chip_positions_dedup, ordered_matches,BJ,distances_cm)
        shift = pRecomb_var/nHaps
        scale = (1 - pRecomb_var)/lastSum
        # sum = 0
        # for j in range(0, nRefHaps):
        #     if j in ordered_matches[m]:
        #         em = pNoErr
        #         # x = ordered_matches[m].index(j)
        #         # length_factor = BJ[ordered_matches[m][x],m]/np.sum(BJ[ordered_matches[m], m])
        #         # if x == 0:
        #         #     order_factor = 1
        #         # else:
        #         #     order_factor = 1
                
        #     else:
        #         em = pErr
        #         # length_factor = 1
        #         # order_factor = 1
        #     # alpha[m, j] = order_factor*length_factor*em*(scale*alpha[m - 1,j] + shift)
        #     alpha[m, j] = em*(scale*alpha[m - 1,j] + shift)
        #     sum += alpha[m, j]
        # Vectorization optimization
        em = np.zeros((nRefHaps,))
        length_factor_arr = np.zeros((nRefHaps,))
        em = em + pErr
        em[ordered_matches[m]] = pNoErr
        
        # length_factor_arr[ordered_matches[m]] = matches[ordered_matches[m],m]/np.sum(matches[ordered_matches[m], m])
        # em[ordered_matches[m]] = pnoerr_matrix[ordered_matches[m],m]

        alpha[m, :] = em * (scale*alpha[m - 1,:] + shift)
        lastSum = sum(alpha[m,:])
    return alpha

def setBwdValues(alpha, num_obs, ordered_matches, chip_positions_dedup, BJ,pnoerr_matrix,distances_cm):
    """
    set backward values
    """
    nRefHaps = 6400
    lastSum  = 1
    end = num_obs
    beta = np.zeros((nRefHaps, 1))
    nHaps = max([len(np.unique(ordered_matches[x])) for x in range(0,num_obs)])
    beta[:, 0] = 1/nHaps
    pErr = 0.0001
    pNoErr = 1 - pErr
    for m in trange(end - 2, 0, -1):
        pRecomb_var = pRecomb(m + 1, chip_positions_dedup, ordered_matches,BJ,distances_cm)
        shift = pRecomb_var/nHaps
        scale = (1 - pRecomb_var)/lastSum
        bwdValSum = 0
        stateSum = 0
        # for j in range(0, nRefHaps):
        #     beta[j, 0] = scale*beta[j, 0] + shift
        #     alpha[m, j] = alpha[m, j] * beta[j,0]
        #     stateSum = stateSum + alpha[m, j]
        #     if j in ordered_matches[m]:
        #         em = pNoErr
        #     else:
        #         em = pErr
        #     beta[j, 0] = beta[j, 0] * em
        #     bwdValSum = bwdValSum + beta[j, 0]

        # Vectorization optimization
        beta[:,0] = scale*beta[:,0] + shift
        alpha[m,:] = alpha[m, :] * beta[:,0]
        stateSum = sum(alpha[m,:])
        em = np.zeros((nRefHaps,))
        em = em + pErr
        em[ordered_matches[m]] = pNoErr
        # em[ordered_matches[m]] = pnoerr_matrix[ordered_matches[m],m]
        beta[:, 0] = beta[:, 0] * em
        bwdValSum = sum(beta[:, 0])

        # for j in range(0, nRefHaps):
        alpha[m, :] = alpha[m, :]/stateSum
    return alpha

def get_pnoerr_matrix(BI, BJ):
    matches = np.roll(BI,1,axis=1) + BJ
    matches_2 = matches/2
    # matches = matches.astype(int)
    pnoerr_matrix = np.zeros(BJ.shape)
    for hap in trange(0,6400):
        for marker in range(0,BJ.shape[1]):
            if matches_2[hap, marker] < BJ[hap, marker]:
                num = BJ[hap, marker] - matches_2[hap, marker]
                pnoerr_matrix[hap, marker] = (num/matches_2[hap, marker]) * (0.997 - 0.999) + 0.999
            else:
                num = matches_2[hap, marker] - BJ[hap, marker]
                pnoerr_matrix[hap, marker] = (num/matches_2[hap, marker]) * (0.997 - 0.999) + 0.999
    return pnoerr_matrix