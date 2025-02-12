import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy.ndimage import gaussian_filter1d
from scipy.special import factorial
from scipy.stats import spearmanr
import random
import pandas as pd
import seaborn as sns
plt.rcParams.update({'figure.facecolor': 'w',
                     'figure.dpi': 300})
def GetData(data, fs=20, bound=(0, 4847)):
    """
    To
    - re-organize the data.
    - Separate data to left and right
    
    """
    # Specific time slice of the data. Because we want to separate left and right.
    ind1, ind2 = bound
    
    # Position
    y = data['lin_pos']
    y = y[bound[0]:bound[1]]
    
    # dt, time bin size
    dt = 1/fs
    
    # Burst idx (within the data slice)
    burst_tidxs = data['spike_idx'][0]
    mask = (burst_tidxs >= bound[0]) & (burst_tidxs < bound[1])
    burst_tidxs = burst_tidxs[mask] - bound[0]
    
    # Cluster ids (within the data slice)
    burst_cluids = data['ids_clust'][mask]
    
    # Spike times for each neuron (within the data slice)
    num_neurons = len(data['spike_idx_cells'])
    neuron_tspidx = []
    t_trials=[]
    for ni in range(num_neurons):
        tspidx = data['spike_idx_cells'][ni][0]
        tspidx = tspidx[(tspidx >= bound[0]) & (tspidx < bound[1])]
        tspidx = tspidx - bound[0]
        neuron_tspidx.append(tspidx)

    # neuron_tspidx=[] 
    # for ni in range(num_neurons):
    #     if len (data['spike_idx_cells'][ni])>1:

    #         tspidx = data['spike_idx_cells'][ni][0]
    #         tspidx = tspidx[(tspidx >= bound[0]) & (tspidx < bound[1])]
    #         tspidx = tspidx - bound[0]
    #         neuron_tspidx.append(tspidx)
    #     else :
    #         tspidx=[]
    #         neuron_tspidx.append(tspidx)

    neuron_tspidx = np.array(neuron_tspidx, dtype=object)  # Spike times per neuron
    



    return y, neuron_tspidx, burst_tidxs, burst_cluids, num_neurons, dt


def GetOccupancy(y, yedges, dt, sigma_yidx):
    """
    Compute the occupancy (time spent by the animal in each position bin).
    
    y : ndarray
        1-D array. Position of the animal.
        
    yedges : ndarray
        1-D array. Edges of the position bins.
        
    dt : float
        Bin size of time for each array index.
        
    sigma_yidx: float
        Sigma of the gaussian filter in the unit of array index. 
        The size of the gaussian should be similar to the size of the animal.
    
    """
    # Occupancy is computed as (counts of data point) * dt
    counts_y, _ = np.histogram(y, bins=yedges)
    occ = counts_y.astype(float) * dt
    
    # Gaussian filtering the resultant histogram for smoothing purpose. 
    # It is more efficient than filtering before the data points.
    occ_gau = gaussian_filter1d(occ.astype(float), sigma=sigma_yidx, mode="nearest")
    return occ, occ_gau

def GetRate(ysp, occ_gau, yedges, sigma_yidx):
    """
    Compute the spike counts and firing rates (= occupancy/SpikeCounts) as a function of position.
    
    ysp : ndarray
        1-D array. Position of the animal at the spike times.
    
    occ_gau : ndarray
        1-D array. Gaussian-smoothed occupancy histogram. 
    
    yedges : ndarray
        1-D array. Edges of the position bins.
        
    sigma_yidx: float
        Sigma of the gaussian filter in the unit of array index.
    
    """
    
    counts_ysp, _ = np.histogram(ysp, bins=yedges)
    counts_ysp_gau = gaussian_filter1d(counts_ysp.astype(float), sigma=sigma_yidx, mode="nearest")
    rate = counts_ysp_gau / occ_gau
    P_x_f = occ_gau/occ_gau.sum()
    return rate, counts_ysp_gau,P_x_f

def MidAx(edges):
    """
    Compute the middle points of the edges. 
    [-0.5, 0.5, 1.5, 2.5] --> [0.0, 1.0, 2.0]
    """
    return (edges[:-1] + edges[1:]) / 2




def exclude_idx(input_idxs, mask_idxs):
    """
    For the purpose of excluding data points at the burst events when constructing place fields.
    The function removes those values in "input_idxs" that are contained in the array of "mask_idxs"
    
    input_idxs : iterable
        Indexes to be masked.
    mask_idxs : iterable
        Mask to be appplied to exclude thosse in input_idxs.
    """
    output_idxs = np.array([input_idxs[i] for i in range(len(input_idxs)) if input_idxs[i] not in mask_idxs]).astype(np.int64)
    return output_idxs


def expand_neighbins(all_burst_tidxs, expand_neighs=0):
    """
    For the purpose of excluding not just the data points at the burst events, but also the neighbouring points around the burst events.
    
    The function expands indexes in an array (burst time indexes) to their neighbouring values. 
    
    E.g. [2, 8, 50] -> [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 48, 49, 50, 51, 52], with expand_neighs=2.
    
    all_burst_tidxs : iterable
        Values to be expanded to their neighbours.
        
    expand_neighs : int
        Number of neighbours to be expanded.
    
    """
    
    expanded_all_burst_tidxs = []
    for burst_tidx in all_burst_tidxs:
        expanded_tidxs = np.arange(burst_tidx-expand_neighs, burst_tidx+expand_neighs+1)
        expanded_all_burst_tidxs.extend(expanded_tidxs)
    expanded_all_burst_tidxs = np.array(expanded_all_burst_tidxs).astype(np.int64)
    return expanded_all_burst_tidxs




foo1 = [2, 10, 50]
foo2 = expand_neighbins(foo1, expand_neighs=2)
foo3 = [3, 7, 12, 30, 51, 100]
foo4 = exclude_idx(foo3, foo2)

print('Illustration of the use of these functions\n', 
      'Mask indexes (burst array indexes): \n', foo1, 
      '\nMask indexes expanded to its 2 neighbours: \n', foo2, 
      '\nInput indexes (array indexes of the positions y or spike positions ysp)\n', foo3, 
      '\nInput indexes excluding those in the set of mask indexes\n', foo4)


def spike_shuffle(data, tspidx, oddeven=None):
    """
    Circularly shuffles spike times within each trial.

    This function takes a set of spike time indices and "circularly shuffles" them 
    on a per-trial basis. For each trial (or a subset of trials specified by the `oddeven`
    parameter), the function identifies the spike times that fall within the trial, 
    then performs a circular (or cyclic) shift of these spike times by a random amount.
    
    Parameters
    ----------
    data : dict
        A dictionary containing the trial data. It must include the key 'trial_idx_mask',
        which is an array-like object indicating the trial index for each time point.
    tspidx : array-like
        An array or list of spike time indices (e.g., time points where spikes occurred).
    oddeven : {None, 'odd', 'even'}, optional
        Determines which trials are processed:
          - None (default): All trials are processed.
          - 'odd': Only trials at odd-numbered positions (i.e., the 2nd, 4th, 6th, ...)
            in the sorted unique trial indices are processed.
          - 'even': Only trials at even-numbered positions (i.e., the 1st, 3rd, 5th, ...)
            in the sorted unique trial indices are processed.
    
    Returns
    -------
    spk_sh : list
        A list containing the circularly shuffled spike time indices from the processed trials.
    spk : numpy.ndarray
        A NumPy array of the original spike time indices that were identified within the processed trials.
    
    Notes
    -----
    For each trial:
      1. The function identifies the time points corresponding to that trial using the trial
         index mask.
      2. It then extracts the spike times (from `tspidx`) that occur during the trial.
      3. A boolean mask is created to mark the positions of these spikes within the trial's time points.
      4. This boolean mask is circularly shifted (rolled) by a random number of positions, effectively
         shuffling the spike times while preserving the overall distribution.
      5. The shuffled spike times are then recorded.
    
    Example
    -------
    >>> import numpy as np
    >>> # Create dummy data with three trials (each trial having 3 time points)
    >>> data = {'trial_idx_mask': np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])}
    >>> tspidx = [0, 2, 4, 7]  # Some example spike indices
    >>> # Shuffle spikes across all trials
    >>> spk_shuffled, spk_original = spike_shuffle(data, tspidx)
    >>> print("Shuffled spike times:", spk_shuffled)
    >>> print("Original spike times:", spk_original)
    """
    import random

    if oddeven == None:
        unqtrial = np.unique(data['trial_idx_mask'])
    elif oddeven == 'odd':
        unqtrial = np.unique(data['trial_idx_mask'])[1::2]
    elif oddeven == 'even':
        unqtrial = np.unique(data['trial_idx_mask'])[::2]
    
    t_all = np.arange(len(data['trial_idx_mask']))
    spk_sh = []
    spk = []
    for trl in unqtrial:
        trl_msk = np.asarray(data['trial_idx_mask'] == trl)
        trl_time = t_all[trl_msk]
        spike_times = np.asarray(tspidx)[np.isin(tspidx, trl_time)]
        if len(spike_times) > 0:
            spk.extend(spike_times)
        bst = np.isin(trl_time, np.asarray(spike_times))
        # Circularly shift the boolean mask by a random number of positions
        spike_times_sh = np.asarray(trl_time)[np.roll(bst, random.randint(1, len(bst) - 1))]
        if len(spike_times_sh) > 0:
            spk_sh.extend(spike_times_sh)
    return spk_sh, np.asarray(spk)


# def spike_shuffle(data,tspidx,oddeven=None):
#     if oddeven==None:
#         unqtrial=np.unique(data['trial_idx_mask'])
#     elif oddeven=='odd':
#         unqtrial=np.unique(data['trial_idx_mask'])[1::2]
#     elif oddeven=='even':
#         unqtrial=np.unique(data['trial_idx_mask'])[::2]
    
#     t_all=np.arange(len(data['trial_idx_mask']))
#     import random
#     spk_sh=[]
#     spk=[]
#     for trl in unqtrial:
#         trl_msk=(np.asarray(data['trial_idx_mask']==trl) )
        


#         trl_time=t_all[trl_msk]
#         spike_times=np.asarray(tspidx)[(np.isin(tspidx,trl_time))]
#         if len(spike_times)>0:
#             spk.extend(spike_times)
#         #bst=np.asarray(np.isin(spike_times,trl_time))
#         bst=np.isin(trl_time,np.asarray(spike_times))
#         spike_times_sh=np.asarray(trl_time)[np.roll(bst,  random.randint(1, len(bst) - 1))]
        
#         if len(spike_times_sh)>0:
#             spk_sh.extend(spike_times_sh)
#             #print(spike_times_sh,spike_times,trl)
#     return(spk_sh,np.asarray(spk))


def computeSpatialInfo(r_x, P_x, r_0, epsilon = 1e-10):
    """
    Compute spatial information per second and per spike, for each neuron. 
    (You could vectorize the computation to calculate I's of all neurons at once, if you like.)
    
    Parameters
    ----------
    r_x : ndarray
        1-d array. r(x). Firing rates for all spatial bins (x) of the linear track.
    P_x : ndarray
        1-d array. P(x). Probability of the rat being in the spatial bins. 
        You could obtain the probability by dividing the occpancy T(x) by the sum of it.
    r_0 : float
        Overall mean firing rate of one neuron, averaged over the P(x).
    epsilon : float
        Small value to avoid computing log(0). Default 1e-10. 

    Returns
    -------
    I : float
        Spatial information per second.
    """
    # ================================= Exercise starts here ========================================
    #r_x=r_x-np.min(r_x)
    r_x[r_x<0]=0
    if r_0<0:
        r_0=0
    I = np.sum(P_x * r_x * np.log2((r_x + epsilon)/(r_0 + epsilon)))
    # ================================= Exercise ends here ========================================
    return I


def decode_neighbins(burst_tidx, y_ax, r_all, neuron_tspidx, dt=0.05, num_neigh=0):
    """
    Decode the posterior probabilities of all positions for EACH burst event, using also the neighbouring data points of the burst.
    
    Since the spike times are not so reliable, it might be beneficial to include the neighbouring bins.
    
    For references to the Bayesian decoder, see Chapter 3.3 Population decoding (equation 3.30) in
    Dayan, P., & Abbott, L. F. (2001). Theoretical neuroscience: Computational and mathematical modeling of neural systems.

    
    Parameters
    ----------
    burst_tidx: ndarray
        1-D array containing all the time indexes of the burst events.
        
    y_ax: ndarray
        Shape = (NumPositionBins, ). Position bins. (also the mid-points of the position edges of the histogram)
        
    r_all: ndarray
        Shape = (NumNeurons, NemPositionBins). Mean firing rates in Hz for each neuron and each position.
    
    neuron_tsp: ndarray
        Shape = (NumNeurons,). Each element contains an 1-d ndarray of shape = (NumSpikeTimes, )
    
    dt: float
        Time bin size for decoding. (not the bin size for time index.) 
        
    num_neigh : int
        The number of neighbouring bins to be included when decoding. You can turn it off by setting it to 0.

        
    """

    # Time bins, but for decoding
    dt_total = dt * (num_neigh * 2 + 1)  # the larger the time bins, the larger the "delta t" for computing the rate.
    tidx_min, tidx_max = burst_tidx - num_neigh, burst_tidx + num_neigh
    
    # Lengths/shapes
    num_neurons = r_all.shape[0]
    num_ybins = y_ax.shape[0]
    
    # Firing rate area
    rdt_all = r_all * dt_total  # rate = rate_per_time * time
    
    # spike counts:
    counts_all = np.zeros(num_neurons)
    for nid in range(num_neurons):
        counts_all[nid] = np.sum((neuron_tspidx[nid] >= tidx_min) & (neuron_tspidx[nid] <= tidx_max)) # both sides inclusive.



    # Log likelihood. 
    k_tmp = counts_all.reshape(-1, 1)
    logL = np.sum(k_tmp * np.log(rdt_all + 1e-10) - rdt_all - np.log(factorial(k_tmp)), axis=0)
    
    # Use Likelihood instead of log-likelihood to do the normalization
    L = np.exp(logL)
    
    # Normalized to each time bin. The posterior probabilities for all positions are summed to 1 in each time bin.
    # The purpose of the normalization is mainly to help in visualization. Probabilities between time bins are more comparable. 
    # Also statistically we do not consider position bins that are beyound our data (outside of the range 0-1)
    L_norm = L / np.nansum(L)
    return logL, L_norm



# def mean_rate2(all_rates_L,P_x_f,num_neurons):
#     r0_f = np.zeros(num_neurons)

#     for nid in range(num_neurons):
#         rates_f = all_rates_L[nid, :]
#         # Compute the mean rate r0 here.
#         r0_f[nid] =  np.sum(P_x_f * rates_f)
#         pass

#     return r0_f,

def mean_rate2(all_rates_L, P_x_f, num_neurons):
    """
    Compute the weighted mean firing rate for each neuron.

    This function calculates the mean firing rate for each neuron by taking a weighted sum 
    of its firing rate map with respect to the normalized occupancy distribution. The occupancy 
    distribution (`P_x_f`) represents the probability of the animal being in each spatial bin, 
    and is used to weight the firing rates accordingly.

    Parameters
    ----------
    all_rates_L : ndarray
        A 2D NumPy array of shape (num_neurons, n_bins) containing the firing rate maps for each 
        neuron, where each row corresponds to a neuron and each column corresponds to a spatial bin.
    P_x_f : ndarray
        A 1D NumPy array representing the normalized occupancy (i.e., the probability distribution) 
        across the spatial bins. Typically, this is computed as:
            P_x_f = occ_gau / occ_gau.sum()
        in the `GetRate` function.
    num_neurons : int
        The total number of neurons to process.

    Returns
    -------
    r0_f : tuple of ndarray
        A tuple containing a 1D NumPy array of length `num_neurons`. Each element in this array is 
        the weighted mean firing rate for the corresponding neuron, calculated as:
            r0_f[nid] = sum(P_x_f * all_rates_L[nid, :])
    
    Example
    -------
    >>> # Assuming all_rates_L is a 2D array of firing rates and P_x_f is the normalized occupancy:
    >>> r0, = mean_rate2(all_rates_L, P_x_f, num_neurons)
    >>> print("Weighted mean firing rates:", r0)
    """
    import numpy as np

    r0_f = np.zeros(num_neurons)

    for nid in range(num_neurons):
        rates_f = all_rates_L[nid, :]
        # Compute the weighted mean firing rate for the neuron.
        r0_f[nid] = np.sum(P_x_f * rates_f)

    return r0_f,











def calculate_common_ratio(data):
    """
    Calculate the ratio of common significant place cells between learning and learned.

    Parameters:
    - data: Dictionary containing the data.
    - base: 'learning' or 'learned' to specify which condition to base the ratio on.

    Returns:
    - Dictionary with ratios for each animal.
    """
    ratios = {}
    ratios_sig={'learning':{},'learned':{},'all':{}}
    ratios_sig = {}
    PC_to_nonPC={}
    for animal in data['learning']:
        common_L = sum(data['learning'][animal]['L'] & data['learned'][animal]['L'])
        common_R = sum(data['learning'][animal]['R'] & data['learned'][animal]['R'])
        
        # total_L = sum(data['learning'][animal]['L'])
        # total_R = sum(data['learning'][animal]['R'])
        

        total_L=(np.sum((data['learning'][animal]['L'])|(data['learned'][animal]['L'])))# ratio of common significant PCs
        total_R=(np.sum((data['learning'][animal]['R'])|(data['learned'][animal]['R'])))# ratio of common significant PCs


        # ratio_L = common_L / total_L if total_L > 0 else 0
        # ratio_R = common_R / total_R if total_R > 0 else 0 # ratio of common cells to sig cells 
        

        ratio_L = common_L / len(data['learning'][animal]['L']) if len(data['learning'][animal]['L']) > 0 else 0# ratio of common cells to all cells
        ratio_R = common_R / len(data['learning'][animal]['R']) if len(data['learning'][animal]['R']) > 0 else 0


        ratios[animal] = {'L': ratio_L, 'R': ratio_R}




        ratio_L_sig_learning= np.sum(data['learning'][animal]['L'])/len(data['learning'][animal]['L'])
        ratio_R_sig_learning= np.sum(data['learning'][animal]['R'])/len(data['learning'][animal]['R'])


        ratio_L_sig_learned= np.sum(data['learned'][animal]['L'])/len(data['learned'][animal]['L'])
        ratio_R_sig_learned= np.sum(data['learned'][animal]['R'])/len(data['learned'][animal]['R'])

        ratios_sig[animal] = {'learning':{'L': ratio_L_sig_learning, 'R': ratio_R_sig_learning},'learned':{'L': ratio_L_sig_learned, 'R': ratio_R_sig_learned}}
        #ratios_sig[animal] = {}


        # Count how many PC turned to non PC 
        vector_1= np.asarray(data['learning'][animal]['L'])
        vector_2= np.asarray(data['learned'][animal]['L'])
        false_to_true_L = np.sum((vector_1 == False) & (vector_2 == True))/len(vector_1)
        true_to_false_L = np.sum((vector_1 == True) & (vector_2 == False))/len(vector_1)


                # Count how many PC turned to non PC 
        vector_1_r= np.asarray(data['learning'][animal]['R'])
        vector_2_r= np.asarray(data['learned'][animal]['R'])
        false_to_true_R = np.sum((vector_1_r == False) & (vector_2_r == True))/len(vector_1_r)
        true_to_false_R = np.sum((vector_1_r == True) & (vector_2_r == False))/len(vector_1_r)
        PC_to_nonPC[animal]={'L':{'NonPC_to_PC':false_to_true_L,'PC_NonPC':true_to_false_L},'R':{'NonPC_to_PC':false_to_true_R,'PC_NonPC':true_to_false_R}}

    return ratios,ratios_sig,PC_to_nonPC




# def compute_corrleation(sorted_l,sorted_r):
# #    find the level of similarity of PF between left and right runs
#     #inpute is the place fields of left and right runs
#     # output is the correlation between left and right PFs
#     # imprtatnt!!!: left argument (firts) is the one cells are certed by
#     eps=0#.000000001# to avaoid nans 

#     sorted_l_cut=sorted_l[np.argmax(sorted_l,axis=1)>500]+eps
#     sorted_r_cut=sorted_r[np.argmax(sorted_l,axis=1)>500]+eps

#     similarity_between_l_and_r_arms=[]# arms
#     for l_cell,r_cell in (zip(sorted_l_cut,sorted_r_cut)):
#         similarity_between_l_and_r_arms.append(np.corrcoef(l_cell,r_cell)[0,1])


#     sorted_l_cut_stem=sorted_l[np.argmax(sorted_l,axis=1)<=500]+eps
#     sorted_r_cut_stem=sorted_r[np.argmax(sorted_l,axis=1)<=500]+eps

#     similarity_between_l_and_r_stem=[]# stem
#     for l_cell,r_cell in (zip(sorted_l_cut_stem,sorted_r_cut_stem)):
#         similarity_between_l_and_r_stem.append(np.corrcoef(l_cell,r_cell)[0,1])

#     mask_cut=np.argmax(sorted_l,axis=1)>500


#     similarity_between_l_and_r_all=[]# all
#     for l_cell,r_cell in (zip(sorted_l,sorted_r)):
#         similarity_between_l_and_r_all.append(np.corrcoef(l_cell,r_cell)[0,1])



#     return similarity_between_l_and_r_arms,similarity_between_l_and_r_stem,similarity_between_l_and_r_all,mask_cut


def compute_corrleation(sorted_l, sorted_r):
    """
    Compute the Pearson correlation coefficients between left and right place fields.
    # imprtatnt!!!: left argument (firts) is the one cells are certed by.
    This function calculates the similarity between the place field maps (e.g., firing rate maps)
    of left and right runs for a group of neurons. It uses the left run place fields to determine a
    cutoff for classifying cells into two groups:
    
      - **Arms**: Cells for which the index of the maximum response in the left place field is greater
        than 500.
      - **Stem**: Cells for which the index of the maximum response in the left place field is less than
        or equal to 500.
    
    For each group, as well as for all cells combined, the function computes the Pearson correlation
    coefficient between the corresponding left and right place fields.

    Parameters
    ----------
    sorted_l : ndarray
        A 2D NumPy array of shape (n_cells, n_bins) representing the left place field maps for each
        neuron. The array should be pre-sorted (e.g., by the position of the peak response).
    sorted_r : ndarray
        A 2D NumPy array of shape (n_cells, n_bins) representing the right place field maps for each
        neuron. The order should correspond to that of `sorted_l`.

    Returns
    -------
    similarity_between_l_and_r_arms : list of float
        A list of Pearson correlation coefficients for cells classified as "arms" 
        (i.e., where the index of the maximum response in `sorted_l` is greater than 500).
    similarity_between_l_and_r_stem : list of float
        A list of Pearson correlation coefficients for cells classified as "stem" 
        (i.e., where the index of the maximum response in `sorted_l` is less than or equal to 500).
    similarity_between_l_and_r_all : list of float
        A list of Pearson correlation coefficients computed for all cells.
    mask_cut : ndarray of bool
        A Boolean array of shape (n_cells,) where each element is True if the corresponding cell's
        maximum in `sorted_l` is greater than 500 (i.e., the cell belongs to the "arms" group), and
        False otherwise.

    Notes
    -----
    - A small constant `eps` is added to the place field data to avoid potential NaN issues during
      the correlation calculation. Currently, `eps` is set to 0 (the commented value suggests that a
      small nonzero value might be used if needed).
    - The cutoff value of 500 is used to differentiate between "arms" and "stem" cells. Depending on your
      data, you might need to adjust this threshold.

    Example
    -------
    >>> import numpy as np
    >>>
    >>> # Suppose left_fields and right_fields are 2D arrays representing place fields (e.g., firing rate maps)
    >>> # for a set of neurons, and they have been pre-sorted based on the peak location in the left fields.
    >>> left_fields = np.random.rand(100, 1000)  # 100 cells, 1000 spatial bins
    >>> right_fields = np.random.rand(100, 1000)
    >>>
    >>> # Compute the similarity between left and right place fields
    >>> sim_arms, sim_stem, sim_all, mask_cut = compute_corrleation(left_fields, right_fields)
    >>> print("Correlation for arms:", sim_arms)
    >>> print("Correlation for stem:", sim_stem)
    >>> print("Overall correlation:", sim_all)
    """
    import numpy as np

    eps = 0  # Small constant to avoid NaNs (e.g., set to 1e-9 if needed)

    # Select cells where the index of the maximum in the left place field is > 500 ("arms")
    sorted_l_cut = sorted_l[np.argmax(sorted_l, axis=1) > 500] + eps
    sorted_r_cut = sorted_r[np.argmax(sorted_l, axis=1) > 500] + eps

    similarity_between_l_and_r_arms = []  # correlations for "arms"
    for l_cell, r_cell in zip(sorted_l_cut, sorted_r_cut):
        similarity_between_l_and_r_arms.append(np.corrcoef(l_cell, r_cell)[0, 1])

    # Select cells where the index of the maximum in the left place field is <= 500 ("stem")
    sorted_l_cut_stem = sorted_l[np.argmax(sorted_l, axis=1) <= 500] + eps
    sorted_r_cut_stem = sorted_r[np.argmax(sorted_l, axis=1) <= 500] + eps

    similarity_between_l_and_r_stem = []  # correlations for "stem"
    for l_cell, r_cell in zip(sorted_l_cut_stem, sorted_r_cut_stem):
        similarity_between_l_and_r_stem.append(np.corrcoef(l_cell, r_cell)[0, 1])

    # Boolean mask indicating cells in the "arms" category (max index > 500)
    mask_cut = np.argmax(sorted_l, axis=1) > 500

    # Compute correlations for all cells regardless of category
    similarity_between_l_and_r_all = []
    for l_cell, r_cell in zip(sorted_l, sorted_r):
        similarity_between_l_and_r_all.append(np.corrcoef(l_cell, r_cell)[0, 1])

    return similarity_between_l_and_r_arms, similarity_between_l_and_r_stem, similarity_between_l_and_r_all, mask_cut







def shuffling_cluster_rates(sorted_l,sorted_r,cluster_pc_fractions_R,cluster_pc_fractions_L):    



    max_clstr=np.max(np.concatenate((cluster_pc_fractions_L['cluster_numbers'],cluster_pc_fractions_R['cluster_numbers'])))

    # sig_sort_idx_L=np.argsort(np.argmax(rates_left_sig,axis=1))# left run are refrence and righ runs are shuffled
    # sorted_l=np.asarray(rates_left_sig)[sig_sort_idx_L]
    # sorted_r=np.asarray(rates_right_all)[sig_sort_idx_L]

    correlation_l_r,correlation_l_r_stem,similarity_between_l_and_r_all,mask_cut= compute_corrleation(sorted_l,sorted_r)
    rates_left_sig_org= np.zeros((int(max_clstr)+1,np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
    rates_left_sig_org[cluster_pc_fractions_L['singnificnts_cluster_ids']]=cluster_pc_fractions_L['rate_significant']

    similarity_shuffled_cells=[]# compute the similarity for shuffled data
    for itr in range(len(cluster_pc_fractions_R['rate_shuffled_clusters'])):
        rates_right_all_sh= np.zeros((int(max_clstr)+1,np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
        rates_right_all_sh[cluster_pc_fractions_R['cluster_numbers']]=cluster_pc_fractions_R['rate_shuffled_clusters'][itr]


        zero_mask=np.sum(rates_left_sig_org,axis=1)>0
        #rates_left_sig_org=rates_left_sig_org[zero_mask]
        #rates_right_all_sh=rates_right_all_sh[zero_mask]

        sig_sort_idx_L_=np.argsort(np.argmax(rates_left_sig_org,axis=1))
        sorted_l=np.asarray(rates_left_sig_org)[sig_sort_idx_L_]
        sorted_r_sh=np.asarray(rates_right_all_sh)[sig_sort_idx_L_]


        correlation_l_r,correlation_l_r_stem,similarity_between_l_and_r_all_sh,mask_cut2= compute_corrleation(sorted_l,sorted_r_sh)
        similarity_shuffled_cells.append(similarity_between_l_and_r_all_sh)

    binary_mat=(np.asarray(similarity_shuffled_cells)<similarity_between_l_and_r_all)# compare similarity of original data with shuffled data

    reshaped_data = []
    num_shuffles=len(binary_mat)
    p_val_cells=np.zeros(len(binary_mat[0]))
    # Loop over each cell
    for cell in range(len(binary_mat[0])):
        # Extract the data for the current cell across all trials
        cell_data = [binary_mat[trial][cell] for trial in range(num_shuffles)]
        p_val_cells[cell]= 1-np.sum(cell_data)/num_shuffles# p-values of cells with significant similarity between left and right runs
        reshaped_data.append(cell_data)
    #p_val_cells[np.isnan(similarity_between_l_and_r_all)]=False# if the correlation is not defined (nan for zero vectors)

    return p_val_cells


def largest_index_with_condition(values, conditions):
    # Mask the values where the condition is False or the value is nan
    masked_values = np.array(values)
    masked_values = np.where(conditions, masked_values, -np.inf)
    
    # Find the index of the maximum value in the masked array
    max_index = np.nanargmax(masked_values)
    
    return max_index


def sequence_template_memebership(seqinclusters_bool,seqinclusters_sim):
    
    # for each event get the template index with largest similarity 

    seq_template_membership=[]# id_clust
    for ievent in range(len(seqinclusters_bool[0])):
        significance=[seqinclusters_bool[x][ievent] for x in seqinclusters_bool]
        similarity=np.asarray([seqinclusters_sim[x][ievent] for x in seqinclusters_bool])#[mask_i_event]
        # Get the index of the largest value where the condition is True
        index = largest_index_with_condition(similarity,significance, )
        print(f'The index of the largest value where the condition is True is: {index}',similarity[index],significance[index],ievent)
        seq_template_membership.append(index)
    return seq_template_membership



def distance_cell_to_cluster(cluster_rate_all_animals,Cluster_types_all,cell_idx,rate_L_R,animal_name,plotrates):
    #'original':[],'shuffled':[]
    c2c_Distances={}
    c2c_Distances_sh={}
    for cell_tpe in Cluster_types_all[animal_name].keys():# type
        
        rate_cluster=[]
        for ic, tmp_nbr in enumerate(Cluster_types_all[animal_name][cell_tpe]):# cluster
            rate_cell=[]
            distance_cell_to_cluster_org=[]
            distance_cell_to_cluster_shuffled=[]
            for cell_nbr in cell_idx[tmp_nbr]:# cells
                #dist=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                dist=(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))

                distance_cell_to_cluster_org.append(dist)
                
                distance_cell_to_cluster_shuffled_cell=[]# shuffled dictance for each cell

                for sh in range(1000):
                    rmap=rate_L_R[animal_name+'_L']['rate_all'][cell_nbr]
                    rate_sh= np.roll(rmap,  random.randint(1, len(rmap) - 1))# shuffle rate map
                    #dist_sh=np.abs(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                    dist_sh=(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))

                    distance_cell_to_cluster_shuffled_cell.append(dist_sh)




                # mean_distance_sh=[]
                # for ish in range(len(rate_L_R[animal_name+'_L']['rate_all_shuffled'])):
                #     dist_sh=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all_shuffled'][ish][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                #     distance_cell_to_cluster_shuffled_cell.append(dist_sh)
                distance_cell_to_cluster_shuffled.append(distance_cell_to_cluster_shuffled_cell)# distances of all shuffled data to cluster
                
                
                p_val=np.sum(dist>distance_cell_to_cluster_shuffled_cell)/len(distance_cell_to_cluster_shuffled_cell)
                #print(cell_tpe+str(tmp_nbr)+'_'+str(cell_nbr)+':',p_val)
                rate_cell.append(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])

                # plt.figure()
                # plt.hist(distance_cell_to_cluster_shuffled_cell,alpha=.2)
                # plt.vlines(dist,0,100,color='r')
                # plt.title(cell_tpe+str(tmp_nbr)+'_'+str(cell_nbr)+':'+str(p_val))


                #mean_distance_sh.append(np.mean(distance_cell_to_cluster_shuffled_cell))# avg of shuffled for each cell

            #p_val_all_cells=np.sum(mean_distance_sh>distance_cell_to_cluster)/len(distance_cell_to_cluster_shuffled_cell)# compare distance to clusters for all cells with the average of the shuffled


            # c2c_Distances[cell_tpe+str(ic)]=distance_cell_to_cluster_org
            # c2c_Distances_sh[cell_tpe+str(ic)]=distance_cell_to_cluster_shuffled

            c2c_Distances[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_org
            c2c_Distances_sh[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_shuffled


            rate_cluster.append(cluster_rate_all_animals[animal_name]['L'][tmp_nbr])

            if plotrates:
                fig, (ax,ax2) = plt.subplots(1, 2,figsize=(8, 6))
                im = ax.pcolormesh(y_ax, np.arange(np.shape(rate_cell)[0]), rate_cell)

                
                plt.colorbar(im, ax=ax)
                plt.subplot(122)
                ax.set_title('Cell rate maps'+ animal_name+cell_tpe+str(tmp_nbr))
                #fig, ax = plt.subplots(figsize=(16, 8))
                #im = ax2.pcolormesh(y_ax, np.arange(np.shape(cluster_pc_fractions_L['rate_significant'])[0]), cluster_pc_fractions_L['rate_significant'])

                # if len (rate_cluster)>1:
                #     im = ax2.pcolormesh(y_ax, np.arange(np.shape(rate_cluster)[0]), rate_cluster)
                # else:

                ax2.plot(cluster_rate_all_animals[animal_name]['L'][tmp_nbr])


                # plt.colorbar(im, ax=ax2)
                ax2.set_title('cluster'+ animal_name+cell_tpe+str(tmp_nbr))
                # plt.suptitle('Target cells \n'+'Median of correlation beween R and L runs Arms: '+"%.2f"%(np.nanmedian(np.asarray(similarity_between_l_and_r_all)[mask_cut & mask_corr])))

                # plt.figure()
                # plt.hist(distance_cell_to_cluster)
                # plt.title('distance to cluster'+ animal_name+cell_tpe+str(tmp_nbr))

    return c2c_Distances,c2c_Distances_sh









# Function to compare clusters between learning and learned conditions and return the result
def compare_clusters(learning, learned):
    results = {}
    
    for animal, clusters_learning in learning.items():
        clusters_learned = learned[animal]  # Get the corresponding clusters from learned condition

        results[animal] = {}
        
        for phase in ['L', 'R']:
            learning_clusters = clusters_learning[phase]
            learned_clusters = clusters_learned[phase]

            # Find clusters in learned not in learning
            not_in_learning = np.setdiff1d(learned_clusters, learning_clusters)

            # Find clusters in learning not in learned
            not_in_learned = np.setdiff1d(learning_clusters, learned_clusters)

            # Save the results in the dictionary
            results[animal][phase] = {
                "not_in_learning": not_in_learning.tolist(),
                "not_in_learned": not_in_learned.tolist()
            }

    return results




# Function to compare clusters between learning and learned conditions and return the result
def compare_phases(learning, learned):
    results = {}
    
    for animal, phases_learning in learning.items():
        phases_learned = learned[animal]  # Get the corresponding phases from learned condition

        results[animal] = {}
        
        for phase in ['TC_arm', 'PC_arm', 'TC_stem', 'PC_stem']:
            learning_clusters = np.array(phases_learning[phase], dtype=np.int64)
            learned_clusters = np.array(phases_learned[phase], dtype=np.int64)

            # Find clusters in learned not in learning
            not_in_learning = np.setdiff1d(learned_clusters, learning_clusters)

            # Find clusters in learning not in learned
            not_in_learned = np.setdiff1d(learning_clusters, learned_clusters)

            # Save the results in the dictionary
            results[animal][phase] = {
                "not_in_learning": not_in_learning.tolist(),
                "not_in_learned": not_in_learned.tolist()
            }

    return results


#######################Replay functions###################################################
# def get_rate_all(data,neuron_tspidx,occ_gau,yedges,sigma_yidx,expanded_all_burst_tidxs,num_neurons,y_ax,y,shuffle_spikes=False):
#     # This function uses the spike time to estimate the rate maps of all cells 
#     # Compute the place fields
#     counts_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float) # for storing spike counts. Not used.
#     p_all=  np.zeros(y_ax.shape[0]).astype(float)
#     r_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float)  # for storing firing rates


#     for nid in range(num_neurons):
#         tspidx = neuron_tspidx[nid]

        
#         # Exclude spike times occuring in the neighours(=1) of the burst events.
#         masked_tspidx = exclude_idx(tspidx, expanded_all_burst_tidxs)
        
#         if shuffle_spikes:# shuffle spike times for getting random place fields

#             masked_tspidx,spk=spike_shuffle(data,masked_tspidx,None)


#         # Get "spike positions"
#         neuron_ysp = y[masked_tspidx]
#         # Compute smoothed firing rate map
#         rate, counts_ysp_gau,P_x = GetRate(neuron_ysp, occ_gau, yedges, sigma_yidx)

#         #print(len(masked_tspidx),len(masked_tspidx_sh))
        
#         # Store all of them
#         r_all[nid, :] = rate
#         counts_all[nid, :] = counts_ysp_gau  
#         p_all=P_x
#     return r_all,counts_all,p_all
def get_rate_all(data, neuron_tspidx, occ_gau, yedges, sigma_yidx, expanded_all_burst_tidxs, 
                 num_neurons, y_ax, y, shuffle_spikes=False):
    """
    Compute the firing rate maps for all neurons.

    This function estimates the rate maps (or place fields) for multiple neurons by processing
    their spike time indices. For each neuron, the function:
      1. Excludes spikes occurring near burst events using `exclude_idx`.
      2. Optionally shuffles the spike times (circularly) using `spike_shuffle` for control analyses.
      3. Maps the (possibly shuffled) spike times to the animal's y positions.
      4. Computes a smoothed firing rate map using Gaussian smoothing via the `GetRate` function.

    The firing rate is calculated as the ratio of the Gaussian-smoothed spike counts (in each position
    bin) to the Gaussian-smoothed occupancy (`occ_gau`). Additionally, `p_all` is computed as the
    normalized occupancy distribution over the y-axis bins (i.e., each bin's occupancy divided by the
    total occupancy).

    Parameters
    ----------
    data : dict
        Dictionary containing trial or behavioral information. This must include the key 
        'trial_idx_mask', which is used by the `spike_shuffle` function if spike shuffling is enabled.
    neuron_tspidx : list or array-like
        A list (or array) where each element contains the spike time indices for a given neuron.
    occ_gau : array-like
        A 1-D array representing the Gaussian-smoothed occupancy histogram over positions.
    yedges : array-like
        A 1-D array defining the edges of the bins along the y-axis.
    sigma_yidx : float or int
        Standard deviation (sigma) of the Gaussian filter applied to the spike count histogram,
        expressed in the unit of the array index.
    expanded_all_burst_tidxs : array-like
        Array of time indices corresponding to burst events. Spike times occurring near these
        indices are excluded from the rate map calculation.
    num_neurons : int
        The total number of neurons to process.
    y_ax : array-like
        The y-axis bins or centers used for storing the computed rate maps. Its length determines 
        the number of position bins.
    y : array-like
        The animal's y positions for each time point. This array is used to map spike times to positions.
    shuffle_spikes : bool, optional
        If True, spike times for each neuron are circularly shuffled (using `spike_shuffle`) to generate 
        randomized rate maps for control analyses. Default is False.

    Returns
    -------
    r_all : ndarray
        A 2D NumPy array of shape (num_neurons, len(y_ax)) containing the smoothed firing rate maps
        for each neuron.
    counts_all : ndarray
        A 2D NumPy array of shape (num_neurons, len(y_ax)) containing the Gaussian-smoothed spike
        counts for each neuron in each y-axis bin.
    p_all : ndarray
        A 1D NumPy array representing the normalized occupancy (or probability distribution) across 
        the y-axis bins. This is computed in the `GetRate` function as:
            p_all = occ_gau / occ_gau.sum()

    Notes
    -----
    This function depends on several external functions:
      - `exclude_idx`: Removes spike times that occur near burst events.
      - `spike_shuffle`: Circularly shuffles spike times on a per-trial basis if `shuffle_spikes` is True.
      - `GetRate`: Computes the smoothed firing rate map. It takes as input the positions of the spikes,
        the occupancy (`occ_gau`), bin edges (`yedges`), and the Gaussian smoothing parameter (`sigma_yidx`),
        and returns the rate, the Gaussian-smoothed spike counts, and the normalized occupancy distribution (`p_all`).

    Example
    -------
    >>> # Example usage assuming all necessary variables and external functions are defined:
    >>> r_all, counts_all, p_all = get_rate_all(data, neuron_tspidx, occ_gau, yedges, sigma_yidx,
    ...                                          expanded_all_burst_tidxs, num_neurons, y_ax, y,
    ...                                          shuffle_spikes=True)
    >>> print("Rate maps shape:", r_all.shape)
    >>> print("Spike counts shape:", counts_all.shape)
    >>> print("Normalized occupancy (p_all):", p_all)
    """
    import numpy as np

    # Initialize arrays to store the spike counts and firing rates.
    counts_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float)
    p_all = np.zeros(y_ax.shape[0]).astype(float)
    r_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float)

    for nid in range(num_neurons):
        tspidx = neuron_tspidx[nid]

        # Exclude spike times that occur near burst events.
        masked_tspidx = exclude_idx(tspidx, expanded_all_burst_tidxs)

        # Optionally shuffle spike times to create randomized rate maps.
        if shuffle_spikes:
            masked_tspidx, spk = spike_shuffle(data, masked_tspidx, None)

        # Map spike times to y positions.
        neuron_ysp = y[masked_tspidx]

        # Compute the smoothed firing rate map and occupancy using Gaussian smoothing.
        rate, counts_ysp_gau, P_x = GetRate(neuron_ysp, occ_gau, yedges, sigma_yidx)

        # Store the computed values.
        r_all[nid, :] = rate
        counts_all[nid, :] = counts_ysp_gau
        p_all = P_x  # p_all is identical for all neurons as it depends solely on occupancy.

    return r_all, counts_all, p_all





def get_rate_all_clusters(data,target_cluids_long,y,occ_gau,yedges,sigma_yidx,expanded_all_burst_tidxs,num_neurons,y_ax,burst_tidxs,burst_cluids,shuffle_spikes):
    # This function estimates rate maps of clusters using sequence event data
    # Compute the place fields
    counts_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float) # for storing spike counts. Not used.
    p_all=  np.zeros(y_ax.shape[0]).astype(float)
    r_all = np.zeros((len(target_cluids_long), y_ax.shape[0])).astype(float)  # for storing firing rates


    for cid,target_cluid in enumerate(target_cluids_long):
        #target_cluid=10
        # Select the burst events that belong to this cluster id
        clu_idxs = np.where(burst_cluids == target_cluid)[0]
        num_clu_bursts = len(clu_idxs)

        clu_idx = clu_idxs
        burst_tidx = burst_tidxs[clu_idx]



    #for nid in range(num_neurons):
        tspidx = burst_tidx

        
        # Exclude spike times occuring in the neighours(=1) of the burst events.
        masked_tspidx = exclude_idx(tspidx, expanded_all_burst_tidxs)
        if shuffle_spikes:# shuffle spike times for getting random place fields

            masked_tspidx,spk=spike_shuffle(data,masked_tspidx,None)

        #spikes.append(masked_tspidx)
        # Get "spike positions"
        neuron_ysp = y[masked_tspidx]
               



        neuron_ysp_sh = y[masked_tspidx]
        rate, counts_ysp_gau,P_x_sh = GetRate(neuron_ysp, occ_gau, yedges, sigma_yidx)

        #    # Store all of them
        r_all[cid, :] = rate
        #counts_all_sh[cid, :] = counts_ysp_gau 
        p_all=P_x_sh
    return r_all,counts_all,p_all






# Function to calculate percentages for all animals, contexts, and clusters
def calculate_all_percentages_merged(data, condition, cluster, side):
    results = {}
    if condition in data:
        merged_data = {}
        for animal_id in data[condition].keys():  # Loop through all animal IDs
            cluster_data = data[condition][animal_id].get(cluster, {}).get(side, {})
            for context, subkey_data in cluster_data.items():
                if 'inward' in context:
                    continue
                if context not in merged_data:
                    merged_data[context] = {'pvals_neg': [], 'pvals_pos': []}
                if 'pvals_neg' in subkey_data:
                    merged_data[context]['pvals_neg'].extend(subkey_data['pvals_neg'])
                if 'pvals_pos' in subkey_data:
                    merged_data[context]['pvals_pos'].extend(subkey_data['pvals_pos'])

        for context, values in merged_data.items():
            neg_percentage = (sum(1 for v in values['pvals_neg'] if v < 0.025) / len(values['pvals_neg'])) * 100 if values['pvals_neg'] else 0.0
            pos_percentage = (sum(1 for v in values['pvals_pos'] if v < 0.025) / len(values['pvals_pos'])) * 100 if values['pvals_pos'] else 0.0
            pos_ratio_one_sided= (sum(1 for v in values['pvals_pos'] if v < 0.05) / len(values['pvals_pos'])) * 1 if values['pvals_pos'] else 0.0
            results[context] = {
                'pvals_neg': neg_percentage,
                'pvals_pos': pos_percentage,
                'pos_ratio_one_sided':pos_ratio_one_sided
            }
    return results


# Updated function to extract Z values for a specific context, cluster, and side (left/right)
def extract_z_values_by_context_cluster_side(replay_stats,condition, context, cluster_name, side):
    z_values = []
    if condition in replay_stats:
        for animal_id in replay_stats[condition].keys():
            cluster_data = replay_stats[condition][animal_id].get(cluster_name, {})
            if side in cluster_data:
                for key, condition_data in cluster_data[side].items():
                    if context in key.lower():
                        if 'Z' in condition_data:
                            z_values.extend(condition_data['Z'])
    return z_values







###################Cluster visualizations#################################################

def equal_cluster_length(cluster_pc_fractions_L,cluster_pc_fractions_R):

    cluster_rate_all_animals={}
    max_clstr=np.max(np.concatenate((cluster_pc_fractions_L['cluster_numbers'],cluster_pc_fractions_R['cluster_numbers'])))

    rates_left_sig= np.zeros((int(max_clstr)+1,np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
    rates_right_all= np.zeros_like(rates_left_sig)

    #rates_right_sig= np.zeros_like(rates_left_sig)
    rates_left_all= np.zeros_like(rates_left_sig)


    # l_length[fol1]=len(cluster_pc_fractions_L['cluster_numbers'])
    # r_length[fol1]=len(cluster_pc_fractions_R['cluster_numbers'])
    
    cluster_rates_all={'L':{},'R':{}}
    #rates_left_sig[cluster_pc_fractions_L['singnificnts_cluster_ids']]=cluster_pc_fractions_L['rate_significant']
    rates_left_sig[cluster_pc_fractions_L['cluster_numbers']]=cluster_pc_fractions_L['rate_all_clusters']

    if len(cluster_pc_fractions_R['cluster_numbers'])>0:

        rates_right_all[cluster_pc_fractions_R['cluster_numbers']]=cluster_pc_fractions_R['rate_all_clusters']
    cluster_rates_all['L']=rates_left_sig
    cluster_rates_all['R']=rates_right_all

    cluster_rate_all_animals=cluster_rates_all

    return rates_left_sig,rates_right_all,cluster_rate_all_animals









# def shuffling_cluster_rates_new(sorted_l,sorted_r,cluster_pc_fractions_L,cluster_pc_fractions_R,sig_sort_idx_l):    


#     max_clstr=np.max(np.concatenate((cluster_pc_fractions_L['cluster_numbers'],cluster_pc_fractions_R['cluster_numbers'])))

#     correlation_l_r,correlation_l_r_stem,similarity_between_l_and_r_all,mask_cut= compute_corrleation(sorted_l,sorted_r)

#     rates_left_sig_org= np.zeros((int(max_clstr)+1,np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
#     #rates_left_sig_org[cluster_pc_fractions_L['singnificnts_cluster_ids']]=cluster_pc_fractions_L['rate_significant']
#     rates_left_sig_org[cluster_pc_fractions_L['cluster_numbers']]=cluster_pc_fractions_L['rate_all_clusters']# include all clusters instead of only si



#     similarity_shuffled_cells=[]# compute the similarity for shuffled data
#     for itr in range(len(cluster_pc_fractions_R['rate_shuffled_clusters'])):
#         rates_right_all_sh= np.zeros((int(max_clstr)+1,np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
#         rates_right_all_sh[cluster_pc_fractions_R['cluster_numbers']]=cluster_pc_fractions_R['rate_shuffled_clusters'][itr]


#         zero_mask=np.sum(rates_left_sig_org,axis=1)>0
#         #rates_left_sig_org=rates_left_sig_org[zero_mask]
#         #rates_right_all_sh=rates_right_all_sh[zero_mask]

#         #sig_sort_idx_L_=np.argsort(np.argmax(rates_left_sig_org,axis=1))
#         #sorted_l=np.asarray(rates_left_sig_org)[sig_sort_idx_l]
#         sorted_r_sh=np.asarray(rates_right_all_sh)[sig_sort_idx_l]


#         correlation_l_r,correlation_l_r_stem,similarity_between_l_and_r_all_sh,mask_cut2= compute_corrleation(sorted_l,sorted_r_sh)
#         similarity_shuffled_cells.append(similarity_between_l_and_r_all_sh)

#     binary_mat=(np.asarray(similarity_shuffled_cells)<similarity_between_l_and_r_all)# compare similarity of original data with shuffled data

#     reshaped_data = []
#     num_shuffles=len(binary_mat)
#     p_val_cells=np.zeros(len(binary_mat[0]))
#     # Loop over each cell
#     for cell in range(len(binary_mat[0])):
#         # Extract the data for the current cell across all trials
#         cell_data = [binary_mat[trial][cell] for trial in range(num_shuffles)]
#         p_val_cells[cell]= 1-np.sum(cell_data)/num_shuffles# p-values of cells with significant similarity between left and right runs
#         reshaped_data.append(cell_data)
#     #p_val_cells[np.isnan(similarity_between_l_and_r_all)]=False# if the correlation is not defined (nan for zero vectors)

#     return p_val_cells,mask_cut

def shuffling_cluster_rates_new(sorted_l, sorted_r, cluster_pc_fractions_L, cluster_pc_fractions_R, sig_sort_idx_l):
    """
    Compute p-values for cluster rate correlations by comparing actual and shuffled place field data.

    This function evaluates the significance of the similarity between left and right place field 
    maps at the cluster level. It first computes the original correlation (similarity) between the 
    sorted left and right place field maps using `compute_corrleation`. Then, for each shuffled version 
    of the right place field data (provided in the cluster data), it computes the corresponding correlation 
    after reordering with the same sorting indices as the left data. The function then compares the 
    shuffled correlations to the original correlation for each cluster, and computes a p-value per cell 
    (or cluster) representing the proportion of shuffles in which the similarity was lower than in the 
    original data.

    Parameters
    ----------
    sorted_l : ndarray
        A 2D NumPy array of shape (n_cells, n_bins) containing the sorted left place field maps. 
        Sorting is typically done based on the peak response location.
    sorted_r : ndarray
        A 2D NumPy array of shape (n_cells, n_bins) containing the sorted right place field maps. 
        The order should correspond to that of `sorted_l`.
    cluster_pc_fractions_L : dict
        A dictionary containing place cell data for the left runs. Expected keys include:
          - 'cluster_numbers': An array of cluster identifiers for each cell.
          - 'rate_all_clusters': A 2D array of the actual rate maps for each cluster.
          - 'rate_significant': A 2D array of rate maps (or other significance-related data) that 
            may be used to define the shape of the arrays.
    cluster_pc_fractions_R : dict
        A dictionary containing place cell data for the right runs. Expected keys include:
          - 'cluster_numbers': An array of cluster identifiers for each cell.
          - 'rate_shuffled_clusters': A list (or array) of 2D arrays, where each element represents 
            a shuffled version of the right rate maps for all clusters.
    sig_sort_idx_l : ndarray or list
        An array of indices representing the sorting order for the left place field maps. This 
        ordering is applied to both left and shuffled right data before correlation is computed.

    Returns
    -------
    p_val_cells : ndarray
        A 1D NumPy array containing the p-values for each cell (or cluster). Each p-value is computed 
        as 1 minus the fraction of shuffles in which the correlation (similarity) between the left and 
        right (shuffled) rate maps is less than the original correlation.
    mask_cut : ndarray
        A Boolean array indicating the cells (or clusters) used to compute the correlations based on a 
        threshold (derived in `compute_corrleation`). This mask is obtained from the original correlation 
        computation and can be used to distinguish between different spatial groups (e.g., "arms" vs. "stem").

    Notes
    -----
    - The function relies on `compute_corrleation` to compute the Pearson correlation coefficients 
      between the left and right (or shuffled right) rate maps.
    - The maximum cluster number is determined from the concatenated cluster numbers in both left 
      and right datasets. The rate maps for the left side are organized into an array (`rates_left_sig_org`)
      where the rows correspond to cluster numbers.
    - For each shuffle of the right rate maps, the function reorders the data using `sig_sort_idx_l` 
      and computes the correlation with the sorted left data.
    - A binary matrix is constructed by comparing the shuffled correlations to the original correlation. 
      P-values are computed on a per-cell basis.
      
    Example
    -------
    >>> # Assuming sorted_l and sorted_r are 2D arrays with sorted place field maps,
    >>> # and cluster_pc_fractions_L/R are dictionaries containing the required cluster data:
    >>> p_val_cells, mask_cut = shuffling_cluster_rates_new(sorted_l, sorted_r, 
    ...                                                      cluster_pc_fractions_L, 
    ...                                                      cluster_pc_fractions_R, 
    ...                                                      sig_sort_idx_l)
    >>> print("P-values for clusters:", p_val_cells)
    >>> print("Mask of clusters used in analysis:", mask_cut)
    """
    import numpy as np

    # Determine the maximum cluster number across left and right datasets.
    max_clstr = np.max(np.concatenate((cluster_pc_fractions_L['cluster_numbers'], 
                                         cluster_pc_fractions_R['cluster_numbers'])))

    # Compute the original correlation between the sorted left and right place field maps.
    correlation_l_r, correlation_l_r_stem, similarity_between_l_and_r_all, mask_cut = compute_corrleation(sorted_l, sorted_r)

    # Build the left rate map organized by cluster.
    rates_left_sig_org = np.zeros((int(max_clstr) + 1, np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
    rates_left_sig_org[cluster_pc_fractions_L['cluster_numbers']] = cluster_pc_fractions_L['rate_all_clusters']
    
    # Initialize list to store similarity values for each shuffle.
    similarity_shuffled_cells = []
    
    # Loop over each shuffled version of the right rate maps.
    for itr in range(len(cluster_pc_fractions_R['rate_shuffled_clusters'])):
        # Build the shuffled right rate map organized by cluster.
        rates_right_all_sh = np.zeros((int(max_clstr) + 1, np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
        rates_right_all_sh[cluster_pc_fractions_R['cluster_numbers']] = cluster_pc_fractions_R['rate_shuffled_clusters'][itr]

        # (Optional) Create a mask for clusters with non-zero rates (currently not applied)
        zero_mask = np.sum(rates_left_sig_org, axis=1) > 0

        # Reorder the shuffled right rate map using the left-side sorting indices.
        sorted_r_sh = np.asarray(rates_right_all_sh)[sig_sort_idx_l]

        # Compute the correlation for the shuffled data.
        correlation_l_r, correlation_l_r_stem, similarity_between_l_and_r_all_sh, mask_cut2 = compute_corrleation(sorted_l, sorted_r_sh)
        similarity_shuffled_cells.append(similarity_between_l_and_r_all_sh)

    # Compare the similarity of shuffled data to the original similarity.
    binary_mat = (np.asarray(similarity_shuffled_cells) < similarity_between_l_and_r_all)

    # Compute p-values for each cell (or cluster) based on the shuffled data.
    num_shuffles = len(binary_mat)
    p_val_cells = np.zeros(len(binary_mat[0]))
    
    for cell in range(len(binary_mat[0])):
        # Extract similarity comparisons for the current cell across all shuffles.
        cell_data = [binary_mat[trial][cell] for trial in range(num_shuffles)]
        p_val_cells[cell] = 1 - np.sum(cell_data) / num_shuffles  # p-value: fraction of shuffles not lower than original

    return p_val_cells, mask_cut



# def significant_pc_to_tc_2(Rates,Rates_sh,TC_learned,PC_learning,PC_learned,animal,y_ax):
#     # get indices of the cells that are pc in learneg and became tc in learned
#     epsilon=0
#     pc_to_tc_learned = TC_learned[PC_learning]# place cells that are TC after learning
#     pc_to_pc_learned = PC_learned[PC_learning]# place cells that are TC after learning

#     non_si_learned=~(TC_learned | PC_learned)

#     pc_to_nonsi= non_si_learned[PC_learning]# PCs in learning that are niether pc nor tc after learning

#     rate_pc_learning_L=(Rates['learning'][animal]['L'][PC_learning])
#     rate_pc_learning_R=(Rates['learning'][animal]['R'][PC_learning])

#     rate_pc_learned_L=(Rates['learned'][animal]['L'][PC_learning])
#     rate_pc_learned_R=(Rates['learned'][animal]['R'][PC_learning])

#     correlation_pc_L=[np.corrcoef(x,y)[0][1] for x,y in zip(rate_pc_learning_L,rate_pc_learned_R)]
#     correlation_pc_R=[np.corrcoef(x,y)[0][1] for x,y in zip(rate_pc_learning_R,rate_pc_learned_L)]


#     shuffle_numbers=len(Rates_sh['learning'][animal]['L'])
#     correlation_pc_sh_L=[]#np.zeros(shuffle_numbers)
#     correlation_pc_sh_R=[]
#     for sh in np.arange(shuffle_numbers):
#         #rate_sh_pc_learning=(Rates_sh['learning'][animal]['L'][sh][PC_to_Non_PC])
#         rate_sh_pc_learned_R=(Rates_sh['learned'][animal]['R'][sh][PC_learning])
#         correlation_pc_sh_L.append([np.corrcoef(x,y+epsilon)[0][1] for x,y in zip(rate_pc_learning_L,rate_sh_pc_learned_R)])

#         rate_sh_pc_learned_L=(Rates_sh['learned'][animal]['L'][sh][PC_learning])
#         correlation_pc_sh_R.append([np.corrcoef(x,y+epsilon)[0][1] for x,y in zip(rate_pc_learning_R,rate_sh_pc_learned_L)])

    
#     overlap=500# total shift length
#     max_corrs, stable_fields_L = compute_max_correlation(rate_pc_learning_L, rate_pc_learned_R, overlap)
#     max_corrs, stable_fields_R = compute_max_correlation(rate_pc_learning_R, rate_pc_learned_L, overlap)
#     significants_L=stable_fields_L
#     significants_R=stable_fields_R




#     # pvals_L = np.sum([np.asarray(correlation_pc_L)>np.asarray(x) for x in correlation_pc_sh_L],axis=0)/shuffle_numbers
#     # significants_L=pvals_L>.95
#     # #significants_L = pc_to_tc_learned & significants2# PC cells that are signficantly stable after learning and are TC in learned

#     # pvals_R = np.sum([np.asarray(correlation_pc_R)>np.asarray(x) for x in correlation_pc_sh_R],axis=0)/shuffle_numbers
#     # significants_R=pvals_R>.95
#     # #significants_R = pc_to_tc_learned & significants2# PC cells that are signficantly stable  after learning and are TC in learned

#     significant_TC_stbl=(significants_L | significants_R) & pc_to_tc_learned# PC cells that are signficantly stable after learning and are TC in learned
#     significant_PC_stbl=(significants_L | significants_R) & pc_to_pc_learned# PC cells that are signficantly stable after learning and are PC in learned
#     significant_nonsi_stbl= pc_to_nonsi# PC cells that are signficantly stable after learning but are not TC or PC in learned

#     TC_unstbl=~(significants_L | significants_R) & pc_to_tc_learned# PC cells that are signficantly stable after learning and are TC in learned
#     PC_unstbl=~(significants_L | significants_R) & pc_to_pc_learned# PC cells that are signficantly stable after learning and are TC in learned



#     if 1:
#         significant_TC_stbl_L=(significants_L ) & pc_to_tc_learned# PC cells that are signficantly stable after learning and are TC in learned

#             # Create figure and axes
#         fig, (ax, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={'width_ratios': [9, 9, 9,9]})
#         # Get max for normalization

#         sig_sort_idx_e = np.argsort(np.argmax(rate_pc_learning_L[significant_TC_stbl_L], axis=1))

#         max_o_e = np.max([np.max(rate_pc_learning_L[significant_TC_stbl_L][sig_sort_idx_e]), np.max(rate_pc_learned_R[significant_TC_stbl_L][sig_sort_idx_e])])

#         # Plot the rate maps for PC in Learning
#         im1 = ax.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learning_L[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
#         plt.colorbar(im1, ax=ax)
#         ax.set_title('PC in Learning (Left)', fontsize=18)
#         ax.set_xlabel('Position [norm]', fontsize=16)
#         ax.set_ylabel('Cell #', fontsize=16)

#         # Plot the rate maps for PC in Learning
#         im1 = ax2.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learning_R[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
#         plt.colorbar(im1, ax=ax2)
#         ax2.set_title('PC in Learning (Right)', fontsize=18)
#         ax2.set_xlabel('Position [norm]', fontsize=16)
#         ax2.set_ylabel('Cell #', fontsize=16)

#         # Plot the rate maps for PC in Learning
#         im1 = ax3.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learned_L[ significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
#         plt.colorbar(im1, ax=ax3)
#         ax3.set_title('TC in Learned (Left)', fontsize=10)
#         ax3.set_xlabel('Position [norm]', fontsize=16)
#         ax3.set_ylabel('Cell #', fontsize=16)

#         # Plot the rate maps for PC in Learning
#         im4 = ax4.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learned_R[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
#         plt.colorbar(im4, ax=ax4)
#         ax4.set_title('TC in Learned(Right)', fontsize=10)
#         ax4.set_xlabel('Position [norm]', fontsize=16)
#         ax4.set_ylabel('Cell #', fontsize=16)

#     return significant_TC_stbl,significant_PC_stbl,TC_unstbl,PC_unstbl,significant_nonsi_stbl

def significant_pc_to_tc_2(Rates, Rates_sh, TC_learned, PC_learning, PC_learned, animal, y_ax):
    """
    Categorize cells based on the stability of their rate maps between the learning and learned phases.

    This function compares the rate maps (for left and right runs) from the learning phase to those
    from the learned phase in order to determine the stability of the place fields for a set of cells.
    Stability is assessed by computing the maximum correlation between the learning and learned rate maps
    using a sliding window approach (with a maximum allowed shift defined by an overlap value). Based on the
    stability measure and the cell type classifications provided as boolean arrays, cells are divided into:
    
      - **significant_TC_stbl**: Cells that were classified as place cells during learning, became TC 
        (task cells) in the learned phase, and meet the stability criterion.
      - **significant_PC_stbl**: Cells that were place cells in learning, remained PC (place cells) in the 
        learned phase, and meet the stability criterion.
      - **TC_unstbl**: Cells that are TC in the learned phase but do not meet the stability criterion.
      - **PC_unstbl**: Cells that are PC in the learned phase but do not meet the stability criterion.
      - **significant_nonsi_stbl**: Cells that were place cells during learning but are classified as neither 
        PC nor TC in the learned phase.

    The function uses the following steps:
      1. Extracts the rate maps for the learning phase (both left and right) and for the learned phase 
         (both left and right) for the cells that were classified as place cells during learning (PC_learning).
      2. Computes the Pearson correlation between the learning-phase rate maps and the learned-phase rate maps 
         for each cell.
      3. Processes shuffled rate maps (from Rates_sh) in a loop to obtain control correlations (the p-value 
         computation using the shuffled data is present but commented out in this code).
      4. Calls the helper function `compute_max_correlation` (with an overlap of 500) to determine, for each cell,
         the maximum correlation across shifts and a corresponding stability flag (stable if the optimal shift is 
         within ±100 (0.1 interval of the whole field).
      5. Combines the stability flags from both left-to-right and right-to-left comparisons.
      6. Uses the input cell type classifications (TC_learned, PC_learning, and PC_learned) to assign cells 
         into the five categories listed above.

    Note:
      - A plotting block is included at the end of the function for visualization of the rate maps. This block 
        does not affect the computed stability classifications and can be ignored if visualization is not required.
    
    Parameters
    ----------
    Rates : dict
        A dictionary containing rate maps for both 'learning' and 'learned' phases. For each phase, it must have 
        keys 'L' and 'R' for left and right rate maps. The data is indexed by animal and then by cell, e.g.,
        Rates['learning'][animal]['L'].
    Rates_sh : dict
        A dictionary containing shuffled rate maps for the 'learning' and 'learned' phases. Its structure is similar
        to that of Rates.
    TC_learned : array-like (boolean)
        Boolean array indicating, for each cell, whether it is classified as a TC (task cell) in the learned phase.
    PC_learning : array-like (boolean)
        Boolean array indicating, for each cell, whether it is classified as a place cell (PC) during the learning phase.
    PC_learned : array-like (boolean)
        Boolean array indicating, for each cell, whether it is classified as a PC in the learned phase.
    animal : str
        Identifier (key) for the animal whose data is being analyzed. Used to index into the Rates and Rates_sh dictionaries.
    y_ax : array-like
        Array representing the spatial bins or positions. This is used for plotting (for normalization and visualization)
        and does not affect the computed stability measures.
    
    Returns
    -------
    significant_TC_stbl : ndarray (boolean)
        Boolean array indicating cells that were place cells during learning, became TC in the learned phase, and are stable.
    significant_PC_stbl : ndarray (boolean)
        Boolean array indicating cells that were place cells during learning, remained PC in the learned phase, and are stable.
    TC_unstbl : ndarray (boolean)
        Boolean array indicating cells that are TC in the learned phase but do not meet the stability criterion.
    PC_unstbl : ndarray (boolean)
        Boolean array indicating cells that are PC in the learned phase but do not meet the stability criterion.
    significant_nonsi_stbl : ndarray (boolean)
        Boolean array indicating cells that were place cells in learning but are classified as neither PC nor TC in the learned phase.
    
    Example
    -------
    >>> significant_TC_stbl, significant_PC_stbl, TC_unstbl, PC_unstbl, significant_nonsi_stbl = \
    ...     significant_pc_to_tc_2(Rates, Rates_sh, TC_learned, PC_learning, PC_learned, animal, y_ax)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # No offset added (epsilon can be adjusted if necessary)
    epsilon = 0

    # Identify cells that became TC or remained PC in the learned phase, considering only cells that were PC in learning.
    pc_to_tc_learned = TC_learned[PC_learning]  # Cells that are TC in learned phase (from the subset of PCs in learning)
    pc_to_pc_learned = PC_learned[PC_learning]  # Cells that are PC in learned phase (from the subset of PCs in learning)

    # Identify cells that are not classified as either TC or PC in the learned phase.
    non_si_learned = ~(TC_learned | PC_learned)
    pc_to_nonsi = non_si_learned[PC_learning]  # PCs in learning that become neither PC nor TC in learned phase

    # Extract rate maps for the learning phase (L and R) and learned phase (L and R) for cells that were PC in learning.
    rate_pc_learning_L = Rates['learning'][animal]['L'][PC_learning]
    rate_pc_learning_R = Rates['learning'][animal]['R'][PC_learning]
    rate_pc_learned_L = Rates['learned'][animal]['L'][PC_learning]
    rate_pc_learned_R = Rates['learned'][animal]['R'][PC_learning]

    # Compute correlations between learning and learned phases.
    correlation_pc_L = [np.corrcoef(x, y)[0][1] for x, y in zip(rate_pc_learning_L, rate_pc_learned_R)]
    correlation_pc_R = [np.corrcoef(x, y)[0][1] for x, y in zip(rate_pc_learning_R, rate_pc_learned_L)]

    # Process shuffled data to compute correlations (control), although the p-value calculation is not finalized.
    shuffle_numbers = len(Rates_sh['learning'][animal]['L'])
    correlation_pc_sh_L = []
    correlation_pc_sh_R = []
    for sh in np.arange(shuffle_numbers):
        rate_sh_pc_learned_R = Rates_sh['learned'][animal]['R'][sh][PC_learning]
        correlation_pc_sh_L.append([np.corrcoef(x, y + epsilon)[0][1] for x, y in zip(rate_pc_learning_L, rate_sh_pc_learned_R)])
        rate_sh_pc_learned_L = Rates_sh['learned'][animal]['L'][sh][PC_learning]
        correlation_pc_sh_R.append([np.corrcoef(x, y + epsilon)[0][1] for x, y in zip(rate_pc_learning_R, rate_sh_pc_learned_L)])

    # Determine stability using a sliding window approach.
    overlap = 500  # Maximum shift length for computing correlations
    _, stable_fields_L = compute_max_correlation(rate_pc_learning_L, rate_pc_learned_R, overlap)
    _, stable_fields_R = compute_max_correlation(rate_pc_learning_R, rate_pc_learned_L, overlap)
    significants_L = stable_fields_L
    significants_R = stable_fields_R

    # Combine the stability indicators from both comparisons.
    significant_TC_stbl = (significants_L | significants_R) & pc_to_tc_learned
    significant_PC_stbl = (significants_L | significants_R) & pc_to_pc_learned
    significant_nonsi_stbl = pc_to_nonsi
    TC_unstbl = ~(significants_L | significants_R) & pc_to_tc_learned
    PC_unstbl = ~(significants_L | significants_R) & pc_to_pc_learned

    # --- Plotting block (for visualization only; does not affect the returned values) ---
    if 1:
        significant_TC_stbl_L = (significants_L) & pc_to_tc_learned  # Subset for visualization
        fig, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={'width_ratios': [9, 9, 9, 9]})
        sig_sort_idx_e = np.argsort(np.argmax(rate_pc_learning_L[significant_TC_stbl_L], axis=1))
        max_o_e = np.max([np.max(rate_pc_learning_L[significant_TC_stbl_L][sig_sort_idx_e]),
                          np.max(rate_pc_learned_R[significant_TC_stbl_L][sig_sort_idx_e])])
        im1 = ax.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), 
                             rate_pc_learning_L[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im1, ax=ax)
        ax.set_title('PC in Learning (Left)', fontsize=18)
        ax.set_xlabel('Position [norm]', fontsize=16)
        ax.set_ylabel('Cell #', fontsize=16)
        im1 = ax2.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), 
                              rate_pc_learning_R[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im1, ax=ax2)
        ax2.set_title('PC in Learning (Right)', fontsize=18)
        ax2.set_xlabel('Position [norm]', fontsize=16)
        ax2.set_ylabel('Cell #', fontsize=16)
        im1 = ax3.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), 
                              rate_pc_learned_L[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im1, ax=ax3)
        ax3.set_title('TC in Learned (Left)', fontsize=10)
        ax3.set_xlabel('Position [norm]', fontsize=16)
        ax3.set_ylabel('Cell #', fontsize=16)
        im4 = ax4.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), 
                              rate_pc_learned_R[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im4, ax=ax4)
        ax4.set_title('TC in Learned (Right)', fontsize=10)
        ax4.set_xlabel('Position [norm]', fontsize=16)
        ax4.set_ylabel('Cell #', fontsize=16)
        plt.close(fig)  # Optionally close the figure if not needed interactively
    # --- End of plotting block ---

    return significant_TC_stbl, significant_PC_stbl, TC_unstbl, PC_unstbl, significant_nonsi_stbl


def compute_significant_transitions_new(data):
    transitions = {}
    for animal in data['learning'].keys():
        cond1 = 'learning'
        cond2 = 'learned'

        # Define SI neurons for learning and learned phases
        si_learning = (
            data[cond1][animal]['TC_arm'] |
            data[cond1][animal]['TC_stem'] |
            data[cond1][animal]['PC_arm'] |
            data[cond1][animal]['PC_stem']
        )
        si_learned = (
            data[cond2][animal]['TC_arm'] |
            data[cond2][animal]['TC_stem'] |
            data[cond2][animal]['PC_arm'] |
            data[cond2][animal]['PC_stem']
        )

        # Sum of SI neurons in learning and learned phases
        total_cells_learning = np.sum(si_learning)
        total_cells_learned = np.sum(si_learned)

        transitions[animal] = {}
        for region in data['learning'][animal].keys():
            learning_cells = data['learning'][animal][region]
            learned_cells = data['learned'][animal][region]

            # Cells that were significant during learning and became insignificant in learned
            sig_to_nonsig = np.sum(learning_cells & ~learned_cells)
            
            # Cells that were insignificant during learning and became significant in learned
            nonsig_to_sig = np.sum(~learning_cells & learned_cells)
            
            # Cells that remained significant in both learning and learned phases
            remained_sig_learning = np.sum(learning_cells & learned_cells)
            remained_sig_learned = remained_sig_learning  # Same as remained_sig_learning, but normalized differently

            # Normalize transitions
            transitions[animal][region] = {
                'Sig to Non-Sig': sig_to_nonsig / total_cells_learning if total_cells_learning > 0 else 0,
                'Non-Sig to Sig': nonsig_to_sig / total_cells_learned if total_cells_learned > 0 else 0,
                'Remained Sig (Learning)': remained_sig_learning / total_cells_learning if total_cells_learning > 0 else 0,
                'Remained Sig (Learned)': remained_sig_learned / total_cells_learned if total_cells_learned > 0 else 0
            }
    return transitions








# def compute_average_transitions_with_remained_new(data):
#     avg_transitions = {
#         'Sig to Non-Sig': {},
#         'Non-Sig to Sig': {},
#         'Remained Sig (Learning)': {},
#         'Remained Sig (Learned)': {}
#     }

#     for region in data['478'].keys():  # Loop through regions
#         # Collect values for each transition type across animals
#         sig_to_nonsig_vals = [data[animal][region]['Sig to Non-Sig'] for animal in data]
#         nonsig_to_sig_vals = [data[animal][region]['Non-Sig to Sig'] for animal in data]
#         remained_sig_learning_vals = [data[animal][region]['Remained Sig (Learning)'] for animal in data]
#         remained_sig_learned_vals = [data[animal][region]['Remained Sig (Learned)'] for animal in data]

#         # Compute averages for each transition type
#         avg_transitions['Sig to Non-Sig'][region] = np.mean(sig_to_nonsig_vals)
#         avg_transitions['Non-Sig to Sig'][region] = np.mean(nonsig_to_sig_vals)
#         avg_transitions['Remained Sig (Learning)'][region] = np.mean(remained_sig_learning_vals)
#         avg_transitions['Remained Sig (Learned)'][region] = np.mean(remained_sig_learned_vals)

#     return avg_transitions


def compute_average_transitions_with_remained_new(data):
    """
    Compute the average transition ratios across animals for each region.

    This function calculates the mean values for various transition types across multiple animals,
    for each region present in the dataset. The transition types computed are:
    
      - 'Sig to Non-Sig': Transition ratio from significant to non-significant.
      - 'Non-Sig to Sig': Transition ratio from non-significant to significant.
      - 'Remained Sig (Learning)': Transition ratio for cells that remained significant during
        the learning phase.
      - 'Remained Sig (Learned)': Transition ratio for cells that remained significant during
        the learned phase.
    
    The input data is assumed to be a nested dictionary where:
      - The top-level keys are animal identifiers.
      - Each animal's value is another dictionary, keyed by region identifiers.
      - For each region, there is a dictionary containing the transition types as keys with numerical
        values.
    
    The function uses the regions from a reference animal (here, the animal with key '478') to loop
    through all regions, and for each region it aggregates the corresponding transition values across all animals.
    It then computes the average (mean) for each transition type for that region.

    Parameters
    ----------
    data : dict
        A nested dictionary containing transition ratio data for each animal. For example:
        
            {
                '478': {
                    'Region1': {
                        'Sig to Non-Sig': value1,
                        'Non-Sig to Sig': value2,
                        'Remained Sig (Learning)': value3,
                        'Remained Sig (Learned)': value4
                    },
                    'Region2': { ... }
                },
                'Animal2': {
                    'Region1': { ... },
                    'Region2': { ... }
                },
                ...
            }
        
        It is assumed that the animal with key '478' exists and that its regions are representative
        of all regions in the dataset.

    Returns
    -------
    avg_transitions : dict
        A dictionary with the same four transition types as keys. Each key maps to another dictionary
        where the keys are region identifiers and the values are the average transition ratio (computed
        as the mean across all animals) for that region. For example:
        
            {
                'Sig to Non-Sig': {'Region1': avg_value, 'Region2': avg_value, ...},
                'Non-Sig to Sig': {'Region1': avg_value, 'Region2': avg_value, ...},
                'Remained Sig (Learning)': {'Region1': avg_value, 'Region2': avg_value, ...},
                'Remained Sig (Learned)': {'Region1': avg_value, 'Region2': avg_value, ...}
            }

    Example
    -------
    >>> # Assuming 'transitions_ratios' is a nested dictionary with the proper structure:
    >>> avg_transitions = compute_average_transitions_with_remained_new(transitions_ratios)
    >>> # Print the average transition ratio for 'Sig to Non-Sig' in 'Region1'
    >>> print(avg_transitions['Sig to Non-Sig']['Region1'])
    """

    # Initialize dictionary to hold the average transitions per region for each transition type.
    avg_transitions = {
        'Sig to Non-Sig': {},
        'Non-Sig to Sig': {},
        'Remained Sig (Learning)': {},
        'Remained Sig (Learned)': {}
    }

    # Loop through regions using a reference animal (here, '478').
    for region in data['478'].keys():
        # Collect values for each transition type across all animals.
        sig_to_nonsig_vals = [data[animal][region]['Sig to Non-Sig'] for animal in data]
        nonsig_to_sig_vals = [data[animal][region]['Non-Sig to Sig'] for animal in data]
        remained_sig_learning_vals = [data[animal][region]['Remained Sig (Learning)'] for animal in data]
        remained_sig_learned_vals = [data[animal][region]['Remained Sig (Learned)'] for animal in data]

        # Compute averages for each transition type.
        avg_transitions['Sig to Non-Sig'][region] = np.mean(sig_to_nonsig_vals)
        avg_transitions['Non-Sig to Sig'][region] = np.mean(nonsig_to_sig_vals)
        avg_transitions['Remained Sig (Learning)'][region] = np.mean(remained_sig_learning_vals)
        avg_transitions['Remained Sig (Learned)'][region] = np.mean(remained_sig_learned_vals)

    return avg_transitions




# # Function to compute the ratio of significant cells (True) to all cells
# def compute_significant_ratios(data, condition):
#     ratios = {}
#     for animal, animal_data in data[condition].items():
#         cond1=condition
#         cond_mask=data[cond1][animal]['TC_arm'] | data[cond1][animal]['TC_stem'] | data[cond1][animal]['PC_arm'] | data[cond1][animal]['PC_stem']

#         # si_learning=data[cond1][animal]['TC_arm'] | cell_type_mask_phases[cond1][animal]['TC_stem'] | cell_type_mask_phases[cond1][animal]['PC_arm'] | cell_type_mask_phases[cond1][animal]['PC_stem']
#         # si_learned=data[cond2][animal]['TC_arm'] | cell_type_mask_phases[cond2][animal]['TC_stem'] | cell_type_mask_phases[cond2][animal]['PC_arm'] | cell_type_mask_phases[cond2][animal]['PC_stem']

#         # si_sum=np.sum(si_learning | si_learned)
#         ratios[animal] = {}
#         for region, array in animal_data.items():
#             total_cells = np.sum(cond_mask)#len(array)# Devide by Si cells of both learning and learned instead of all cells
#             significant_cells = np.sum(array)
#             ratio = significant_cells / total_cells if total_cells > 0 else 0
#             ratios[animal][region] = ratio
#     return ratios
def compute_significant_ratios(data, condition):
    """
    Compute the ratio of significant cells in each region for a given condition.

    This function calculates, for each animal and region, the proportion of significant cells
    relative to the total number of cells that are considered significant based on a union of
    cell-type masks. The significant cells are defined as those belonging to any of the following
    categories: 'TC_arm', 'TC_stem', 'PC_arm', or 'PC_stem'. For each animal, the function uses
    these four masks to determine the total pool of significant cells (the denominator) and then,
    for each region, it computes the ratio of cells that are marked as significant (True) in that
    region (the numerator) to the total pool.

    Parameters
    ----------
    data : dict
        A nested dictionary containing cell type data for each condition and animal. It is assumed
        that `data[condition]` is a dictionary where:
            - The keys are animal identifiers.
            - Each animal's value is another dictionary containing:
                - Four boolean arrays with keys 'TC_arm', 'TC_stem', 'PC_arm', and 'PC_stem'
                  that indicate the significant cell masks.
                - One or more additional keys (e.g., region names) whose associated boolean arrays
                  indicate which cells are significant in that region.
    condition : str
        The condition to process (e.g., 'learning' or 'learned'). This key is used to index into the
        top-level `data` dictionary.

    Returns
    -------
    ratios : dict
        A dictionary where each key is an animal identifier and the corresponding value is another
        dictionary. In this inner dictionary, the keys are region names and the values are the computed
        ratios (i.e., the fraction of significant cells in that region relative to the total number of
        significant cells for that animal). If no significant cells are present (i.e., the denominator is 0),
        the ratio is set to 0.

    Example
    -------
    >>> import numpy as np
    >>> # Example input structure for a single animal in 'learning' condition.
    >>> data = {
    ...     'learning': {
    ...         'animal1': {
    ...             'TC_arm': np.array([True, False, True, False]),
    ...             'TC_stem': np.array([False, True, False, False]),
    ...             'PC_arm': np.array([True, True, False, True]),
    ...             'PC_stem': np.array([False, False, True, False]),
    ...             'RegionA': np.array([True, False, True, False]),
    ...             'RegionB': np.array([False, True, False, True])
    ...         }
    ...     }
    ... }
    >>> condition = 'learning'
    >>> ratios = compute_significant_ratios(data, condition)
    >>> # For animal1 in RegionA, the total pool is computed as:
    >>> # total_cells = sum(TC_arm OR TC_stem OR PC_arm OR PC_stem)
    >>> # and the ratio is the number of True values in 'RegionA' divided by total_cells.
    >>> print(ratios['animal1']['RegionA'])
    """
    import numpy as np

    ratios = {}
    for animal, animal_data in data[condition].items():
        # Use the condition (e.g., 'learning') to obtain the significant cell masks.
        cond1 = condition
        cond_mask = (data[cond1][animal]['TC_arm'] |
                     data[cond1][animal]['TC_stem'] |
                     data[cond1][animal]['PC_arm'] |
                     data[cond1][animal]['PC_stem'])
        
        ratios[animal] = {}
        for region, array in animal_data.items():
            # Compute the total number of significant cells (denominator).
            total_cells = np.sum(cond_mask)  # Alternatively, one might use len(array) if appropriate.
            # Compute the number of significant cells in the given region (numerator).
            significant_cells = np.sum(array)
            # Compute the ratio, guarding against division by zero.
            ratio = significant_cells / total_cells if total_cells > 0 else 0
            ratios[animal][region] = ratio
    return ratios








# def distance_cell_to_cluster_type2(cluster_rate_all_animals,data_animal,Cluster_types_all,rate_L_R,cell_types_all,animal_name,plotrates=False):
#     '''#Cluster_types_all : Index of clusters and their identity
#     #rate_L_R : Ratemap of all cells
#     #  cell_types_all: index of cells and their identity
#     # This function find the distance between peaks of rate maps of different cell types and peaks of different clusters of the same type
#         '''

#     c2c_Distances={}
#     c2c_Distances_sh={}
#     for cell_tpe in Cluster_types_all[animal_name].keys():# type
        
#         rate_cluster=[]
#         for iclstr,tmp_nbr in enumerate(Cluster_types_all[animal_name][cell_tpe]):# cluster
            
#             template_cell_idx=data_animal['template'][tmp_nbr]# template cell indices for a type

#             cluster_cll_idx=cell_types_all[animal_name][cell_tpe]#  cell indices for a type

#             cluster_and_cell_type=np.intersect1d(cluster_cll_idx,template_cell_idx)# cell of a type that are in a cluster type



#             rate_cell=[]
#             distance_cell_to_cluster_org=[]
#             distance_cell_to_cluster_shuffled=[]
#             #for icl,cell_nbr in enumerate(cell_types_all[animal_name][cell_tpe]):# cells

#             for icl,cell_nbr in enumerate(cluster_and_cell_type):# cells

                
#                 #dist=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
#                 distlr=[]
#                 distlr.append(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
#                 distlr.append(np.argmax(rate_L_R[animal_name+'_R']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))

#                 distance_cell_to_cluster_org.extend(distlr)
                
#                 distance_cell_to_cluster_shuffled_cell=[]# shuffled dictance for each cell

#                 distlsh=[]
#                 for sh in range(1000):
#                     rmap=rate_L_R[animal_name+'_L']['rate_all'][cell_nbr]
#                     rate_sh= np.roll(rmap,  random.randint(1, len(rmap) - 1))# shuffle rate map
#                     #dist_sh=np.abs(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
#                     distlsh.append(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))

#                     #distance_cell_to_cluster_shuffled_cell.append(dist_sh)
#                 distrsh=[]
#                 for sh in range(1000):
#                     rmap=rate_L_R[animal_name+'_R']['rate_all'][cell_nbr]
#                     rate_sh= np.roll(rmap,  random.randint(1, len(rmap) - 1))# shuffle rate map
#                     #dist_sh=np.abs(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
#                     distrsh.append(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))

#                     #distance_cell_to_cluster_shuffled_cell.append(dist_sh)

                


#                 # mean_distance_sh=[]
#                 # for ish in range(len(rate_L_R[animal_name+'_L']['rate_all_shuffled'])):
#                 #     dist_sh=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all_shuffled'][ish][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
#                 #     distance_cell_to_cluster_shuffled_cell.append(dist_sh)
#                 distance_cell_to_cluster_shuffled.extend(np.vstack((distlsh,distrsh)))# distances of all shuffled data to cluster
                
                
#                 #p_val=np.sum(dist>distance_cell_to_cluster_shuffled_cell)/len(distance_cell_to_cluster_shuffled_cell)
#                 #print(cell_tpe+str(tmp_nbr)+'_'+str(cell_nbr)+':',p_val)
#                 rate_cell.append(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])

#                 # plt.figure()
#                 # plt.hist(distance_cell_to_cluster_shuffled_cell,alpha=.2)
#                 # plt.vlines(dist,0,100,color='r')
#                 # plt.title(cell_tpe+str(tmp_nbr)+'_'+str(cell_nbr)+':'+str(p_val))


#                 #mean_distance_sh.append(np.mean(distance_cell_to_cluster_shuffled_cell))# avg of shuffled for each cell

#             #p_val_all_cells=np.sum(mean_distance_sh>distance_cell_to_cluster)/len(distance_cell_to_cluster_shuffled_cell)# compare distance to clusters for all cells with the average of the shuffled


#             # c2c_Distances[cell_tpe+str(iclstr)]=distance_cell_to_cluster_org
#             # c2c_Distances_sh[cell_tpe+str(iclstr)]=distance_cell_to_cluster_shuffled

#             c2c_Distances[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_org
#             c2c_Distances_sh[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_shuffled

#             rate_cluster.append(cluster_rate_all_animals[animal_name]['L'][tmp_nbr])

#             if 0:
#                 fig, (ax,ax2) = plt.subplots(1, 2,figsize=(16, 8))
#                 im = ax.pcolormesh(y_ax, np.arange(np.shape(rate_cell)[0]), rate_cell)

            
                    
#                 plt.colorbar(im, ax=ax)
#                 plt.subplot(122)
#                 ax.set_title('Cell rate maps'+ animal_name+cell_tpe+str(tmp_nbr))
#                 #fig, ax = plt.subplots(figsize=(16, 8))
#                 #im = ax2.pcolormesh(y_ax, np.arange(np.shape(cluster_pc_fractions_L['rate_significant'])[0]), cluster_pc_fractions_L['rate_significant'])

#                 # if len (rate_cluster)>1:
#                 #     im = ax2.pcolormesh(y_ax, np.arange(np.shape(rate_cluster)[0]), rate_cluster)
#                 # else:

#                 ax2.plot(cluster_rate_all_animals[animal_name]['L'][tmp_nbr])


#                 # plt.colorbar(im, ax=ax2)
#                 ax2.set_title('cluster'+ animal_name+cell_tpe+str(tmp_nbr))
#                 # plt.suptitle('Target cells \n'+'Median of correlation beween R and L runs Arms: '+"%.2f"%(np.nanmedian(np.asarray(similarity_between_l_and_r_all)[mask_cut & mask_corr])))

#                 # plt.figure()
#                 # plt.hist(distance_cell_to_cluster)
#                 # plt.title('distance to cluster'+ animal_name+cell_tpe+str(tmp_nbr)

#     return  c2c_Distances,c2c_Distances_sh

def distance_cell_to_cluster_type2(cluster_rate_all_animals, data_animal, Cluster_types_all, rate_L_R, cell_types_all, animal_name, plotrates=False):
    """
    Compute the distances between the peak positions of individual cell rate maps and their corresponding cluster template peaks,
    along with distances obtained from shuffled (randomly shifted) rate maps.

    For each cell type (e.g., 'TC_arm', 'PC_arm', etc.) for a given animal, this function performs the following:
      1. For each cluster number associated with that cell type (from Cluster_types_all), the function retrieves:
         - The template cell indices for that cluster from data_animal['template'].
         - The cell indices for the cell type from cell_types_all.
      2. It determines the intersection of these indices—i.e., the cells that belong both to the cell type and to the given cluster.
      3. For each such cell, the function computes the difference between the index of the peak (maximum value)
         in the cell's rate map and the index of the peak in the corresponding cluster template's rate map.
         This computation is performed separately for the left ('L') and right ('R') directions.
      4. In addition, for each cell the function computes a control distribution of distances by shuffling:
         - For 1000 iterations, the cell's rate map is circularly shifted (using a random offset via np.roll),
           and the difference between the shifted rate map's peak and the cluster template's peak is computed.
         - These shuffled distances (for both left and right directions) are stored together.
      5. The results for each cluster are stored in two dictionaries:
         - One dictionary (c2c_Distances) contains the original (unshuffled) distance differences.
         - The other dictionary (c2c_Distances_sh) contains the corresponding shuffled distance differences
           (as an array obtained by stacking left and right shuffled distances).

    Parameters
    ----------
    cluster_rate_all_animals : dict
        Dictionary containing the cluster-level rate maps for each animal. Expected structure:
            cluster_rate_all_animals[animal_name]['L'][cluster_number]
            and
            cluster_rate_all_animals[animal_name]['R'][cluster_number]
    data_animal : dict
        Data for the given animal. Must include a key 'template' such that data_animal['template'][cluster_number]
        provides the indices of template cells for that cluster.
    Cluster_types_all : dict
        Dictionary mapping animal names to cell type clusters. For each animal, Cluster_types_all[animal_name] is a
        dictionary where keys are cell type names and values are lists or arrays of cluster numbers for that cell type.
    rate_L_R : dict
        Dictionary containing the rate maps for all cells, separated by left and right directions.
        Expected keys are of the form animal_name+'_L' and animal_name+'_R'. Each of these maps should contain a key
        'rate_all' which is a 2D array (cells × spatial bins).
    cell_types_all : dict
        Dictionary mapping animal names to cell type indices. For each animal, cell_types_all[animal_name] is a
        dictionary where keys are cell type names and values are arrays of cell indices corresponding to that type.
    animal_name : str
        Identifier for the animal whose data is being processed.
    plotrates : bool, optional
        If True, additional plotting code (currently disabled) would be executed to visualize rate maps and distances.
        Default is False.

    Returns
    -------
    c2c_Distances : dict
        Dictionary mapping a combined key (cell type concatenated with cluster number) to a list of original distance differences
        (computed from both left and right rate maps) between each cell's peak and the corresponding cluster template's peak.
    c2c_Distances_sh : dict
        Dictionary mapping a combined key (cell type concatenated with cluster number) to an array of shuffled distance differences.
        The shuffled distances are computed by performing 1000 random circular shifts on the cell's rate map (separately for left and
        right) and computing the differences in peak positions relative to the cluster template.

    Notes
    -----
    - The function uses np.argmax to determine the index of the maximum value (i.e., the peak) in a rate map.
    - Shuffling is performed using np.roll with a random offset in the range [1, len(rate_map)-1].
    - The plotting code is present (guarded by an if-statement) but is disabled by default.

    Example
    -------
    >>> c2c_Distances, c2c_Distances_sh = distance_cell_to_cluster_type2(
    ...     cluster_rate_all_animals, data_animal, Cluster_types_all, rate_L_R, cell_types_all, animal_name, plotrates=False)
    >>> # c2c_Distances contains the original distance differences,
    >>> # while c2c_Distances_sh contains the shuffled distance arrays.
    """
    import numpy as np
    import random

    c2c_Distances = {}
    c2c_Distances_sh = {}
    # Iterate over each cell type for the given animal.
    for cell_tpe in Cluster_types_all[animal_name].keys():
        # Process each cluster number for this cell type.
        for iclstr, tmp_nbr in enumerate(Cluster_types_all[animal_name][cell_tpe]):
            # Retrieve the template cell indices for the cluster.
            template_cell_idx = data_animal['template'][tmp_nbr]
            # Retrieve the indices for cells of this type.
            cluster_cll_idx = cell_types_all[animal_name][cell_tpe]
            # Find the cells that are in both the template and the cell type.
            cluster_and_cell_type = np.intersect1d(cluster_cll_idx, template_cell_idx)

            distance_cell_to_cluster_org = []
            distance_cell_to_cluster_shuffled = []

            # Compute distances for each cell in the intersection.
            for icl, cell_nbr in enumerate(cluster_and_cell_type):
                # Compute differences in peak positions for left and right rate maps.
                distlr = []
                distlr.append(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr]) -
                              np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                distlr.append(np.argmax(rate_L_R[animal_name+'_R']['rate_all'][cell_nbr]) -
                              np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))
                distance_cell_to_cluster_org.extend(distlr)

                # Compute shuffled distances for the left direction.
                distlsh = []
                for sh in range(1000):
                    rmap = rate_L_R[animal_name+'_L']['rate_all'][cell_nbr]
                    rate_sh = np.roll(rmap, random.randint(1, len(rmap) - 1))
                    distlsh.append(np.argmax(rate_sh) -
                                   np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                # Compute shuffled distances for the right direction.
                distrsh = []
                for sh in range(1000):
                    rmap = rate_L_R[animal_name+'_R']['rate_all'][cell_nbr]
                    rate_sh = np.roll(rmap, random.randint(1, len(rmap) - 1))
                    distrsh.append(np.argmax(rate_sh) -
                                   np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))
                # Stack shuffled distances from left and right and add them to the overall list.
                distance_cell_to_cluster_shuffled.extend(np.vstack((distlsh, distrsh)))

            # Store the distances using a key composed of the cell type and the cluster number.
            c2c_Distances[cell_tpe + str(tmp_nbr)] = distance_cell_to_cluster_org
            c2c_Distances_sh[cell_tpe + str(tmp_nbr)] = distance_cell_to_cluster_shuffled

            # Optionally, if plotting is desired, plotting code would be executed (currently disabled).
            if plotrates:
                # (Plotting code has been excluded.)
                pass

    return c2c_Distances, c2c_Distances_sh

def distance_cell_to_cluster2(cluster_rate_all_animals,Cluster_types_all,cell_idx,rate_L_R,animal_name,plotrates):
    #'original':[],'shuffled':[]

    c2c_Distances={}
    c2c_Distances_sh={}
    for cell_tpe in Cluster_types_all[animal_name].keys():# type
        
        rate_cluster=[]
        for ic, tmp_nbr in enumerate(Cluster_types_all[animal_name][cell_tpe]):# cluster
            rate_cell=[]
            distance_cell_to_cluster_org=[]
            distance_cell_to_cluster_shuffled=[]
            distance_cell_to_cluster_shuffled_L=[]
            distance_cell_to_cluster_shuffled_R=[]

            for cell_nbr in cell_idx[tmp_nbr]:# cells
                #dist=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                dist=(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                rl_dist=[]
                rl_dist.append(np.argmax(rate_L_R[animal_name+'_R']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))
                rl_dist.append(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))

                distance_cell_to_cluster_org.extend(rl_dist)

                
                distance_cell_to_cluster_shuffled_cell=[]# shuffled dictance for each cell
                dist_sh_l=[]
                for sh in range(1000):
                    
                    rmap=rate_L_R[animal_name+'_L']['rate_all'][cell_nbr]
                    rate_sh= np.roll(rmap,  random.randint(1, len(rmap) - 1))# shuffle rate map
                    dist_sh_l.append(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                dist_sh_r=[]
                for sh in range(1000):
                    rmap=rate_L_R[animal_name+'_R']['rate_all'][cell_nbr]
                    rate_sh= np.roll(rmap,  random.randint(1, len(rmap) - 1))# shuffle rate map
                    dist_sh_r.append(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))


                    #distance_cell_to_cluster_shuffled_cell.append(dist_sh_lr)

                distance_cell_to_cluster_shuffled.extend(np.vstack((dist_sh_l,dist_sh_r)))
                #distance_cell_to_cluster_shuffled_R.extend(dist_sh_r)


                # mean_distance_sh=[]
                # for ish in range(len(rate_L_R[animal_name+'_L']['rate_all_shuffled'])):
                #     dist_sh=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all_shuffled'][ish][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                #     distance_cell_to_cluster_shuffled_cell.append(dist_sh)
                #distance_cell_to_cluster_shuffled_cell = [item for sublist in distance_cell_to_cluster_shuffled_cell for item in sublist]# flatten

                #distance_cell_to_cluster_shuffled.extend(distance_cell_to_cluster_shuffled_cell)# distances of all shuffled data to cluster
                
                
                p_val=np.sum(dist>distance_cell_to_cluster_shuffled_cell)/len(distance_cell_to_cluster_shuffled_cell)
                #print(cell_tpe+str(tmp_nbr)+'_'+str(cell_nbr)+':',p_val)
                rate_cell.append(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])


            #distance_cell_to_cluster_org = [item for sublist in distance_cell_to_cluster_org for item in sublist]# flatten
            #distance_cell_to_cluster_shuffled=np.hstack((distance_cell_to_cluster_shuffled_L,distance_cell_to_cluster_shuffled_R))
            c2c_Distances[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_org
            c2c_Distances_sh[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_shuffled


            rate_cluster.append(cluster_rate_all_animals[animal_name]['L'][tmp_nbr])

            if 0:
                fig, (ax,ax2) = plt.subplots(1, 2,figsize=(8, 6))
                im = ax.pcolormesh(y_ax, np.arange(np.shape(rate_cell)[0]), rate_cell)

                
                plt.colorbar(im, ax=ax)
                plt.subplot(122)
                ax.set_title('Cell rate maps'+ animal_name+cell_tpe+str(tmp_nbr))
                #fig, ax = plt.subplots(figsize=(16, 8))
                #im = ax2.pcolormesh(y_ax, np.arange(np.shape(cluster_pc_fractions_L['rate_significant'])[0]), cluster_pc_fractions_L['rate_significant'])

                # if len (rate_cluster)>1:
                #     im = ax2.pcolormesh(y_ax, np.arange(np.shape(rate_cluster)[0]), rate_cluster)
                # else:

                ax2.plot(cluster_rate_all_animals[animal_name]['L'][tmp_nbr])


                # plt.colorbar(im, ax=ax2)
                ax2.set_title('cluster'+ animal_name+cell_tpe+str(tmp_nbr))
                # plt.suptitle('Target cells \n'+'Median of correlation beween R and L runs Arms: '+"%.2f"%(np.nanmedian(np.asarray(similarity_between_l_and_r_all)[mask_cut & mask_corr])))

                # plt.figure()
                # plt.hist(distance_cell_to_cluster)
                # plt.title('distance to cluster'+ animal_name+cell_tpe+str(tmp_nbr))
    return c2c_Distances,c2c_Distances_sh









def plot_kl_distributions_ss2(js_divergence_ss, p_value_corr_js_, name, type='Correct'):
    plt.figure(figsize=(5, 3))

    # Define the bins for both histograms
    bins = np.linspace(0, max(max(js_divergence_ss[type]), max(js_divergence_ss[type + '_sh'])), 30)
    
    # Determine the maximum y limit for both plots
    max_y = max(
        max(np.histogram(js_divergence_ss[type], bins=bins)[0]),
        max(np.histogram(js_divergence_ss[type + '_sh'], bins=bins)[0])
    ) + 2

    # First subplot for the original distribution
    plt.subplot(1, 2, 1)
    sns.histplot(js_divergence_ss[type], bins=bins, kde=False, color='#1f77b4', edgecolor='black', 
                 line_kws={'lw': 2}, label='Original')
    plt.title(name[:-2], fontsize=14)
    plt.xlabel(name[-2:], fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10, frameon=False)
    plt.ylim(0, max_y)
    
    # Add p-value text inside the first plot
    plt.text(0.7, 0.8, f'p = {p_value_corr_js_:.3f}', ha='center', va='center',
             transform=plt.gca().transAxes, fontsize=12, color='black')

    # Second subplot for the shuffled distribution
    sns.histplot(js_divergence_ss[type + '_sh'], bins=bins, kde=False, color='gray', edgecolor='black',
                 line_kws={'lw': 2}, label='Shuffled')
    plt.xlabel(name[-2:], fontsize=12)
    plt.ylabel('Occurrence', fontsize=12)
    plt.legend(fontsize=10, loc='upper right', frameon=False)
    plt.ylim(0, max_y)

    # Overall adjustments for a cleaner look
    plt.tight_layout()
    #plt.show()
    ax=plt.gca()
    return ax











########################Cell_vis functions ##########################################





def shuffling_rates2(cluster_pc_fractions_L,cluster_pc_fractions_R,sig_sort_idx_l):
    # Sort rate maps of left runs and shuffle right runs


    # left_runs=rate_L_R[animal_name+'_L']['rate_all'][rate_L_R[animal_name+'_L']['significant']]# Left as reference
    # right_runs=rate_L_R[animal_name+'_R']['rate_all'][rate_L_R[animal_name+'_L']['significant']]# Left as reference

    left_runs=cluster_pc_fractions_L['rate_all']#[cluster_pc_fractions_L['significant']]# dir as reference
    right_runs=cluster_pc_fractions_R['rate_all']#[cluster_pc_fractions_L['significant']]# dir as reference
    #sig_sort_idx_l=np.argsort(np.argmax(left_runs,axis=1))
    #sig_sort_idx_r=np.argsort(np.argmax(right_runs,axis=1))


    correlation_l_r,correlation_l_r_stem,similarity_between_l_and_r_all,mask_cut= compute_corrleation(left_runs[sig_sort_idx_l],right_runs[sig_sort_idx_l])






    similarity_shuffled_cells=[]# compute the similarity for shuffled data
    for itr in range(len(cluster_pc_fractions_L['rate_all_shuffled'])):
        
        sorted_r_sh=np.asarray(cluster_pc_fractions_R['rate_all_shuffled'][itr][sig_sort_idx_l])
        correlation_l_r,correlation_l_r_stem,similarity_between_l_and_r_all_sh,mask_cut= compute_corrleation(left_runs[sig_sort_idx_l],sorted_r_sh)




        similarity_shuffled_cells.append(similarity_between_l_and_r_all_sh)

    binary_mat=(np.asarray(similarity_shuffled_cells)<similarity_between_l_and_r_all)# compare similarity of original data with shuffled data

    reshaped_data = []
    num_shuffles=len(binary_mat)
    p_val_cells=np.zeros(len(binary_mat[0]))
    # Loop over each cell
    for cell in range(len(binary_mat[0])):
        # Extract the data for the current cell across all trials
        cell_data = [binary_mat[trial][cell] for trial in range(num_shuffles)]
        p_val_cells[cell]= 1-np.sum(cell_data)/num_shuffles# p-values of cells with significant similarity between left and right runs
        reshaped_data.append(cell_data)

    return p_val_cells,mask_cut



# Function to compute the ratio of significant cells (True) to all cells
def compute_significant_ratios(data, condition):
    ratios = {}
    for animal, animal_data in data[condition].items():
        cond1=condition
        cond_mask=data[cond1][animal]['TC_arm'] | data[cond1][animal]['TC_stem'] | data[cond1][animal]['PC_arm'] | data[cond1][animal]['PC_stem']

        # si_learning=data[cond1][animal]['TC_arm'] | cell_type_mask_phases[cond1][animal]['TC_stem'] | cell_type_mask_phases[cond1][animal]['PC_arm'] | cell_type_mask_phases[cond1][animal]['PC_stem']
        # si_learned=data[cond2][animal]['TC_arm'] | cell_type_mask_phases[cond2][animal]['TC_stem'] | cell_type_mask_phases[cond2][animal]['PC_arm'] | cell_type_mask_phases[cond2][animal]['PC_stem']

        # si_sum=np.sum(si_learning | si_learned)
        ratios[animal] = {}
        for region, array in animal_data.items():
            total_cells = np.sum(cond_mask)#len(array)# Devide by Si cells of both learning and learned instead of all cells
            significant_cells = np.sum(array)
            ratio = significant_cells / total_cells if total_cells > 0 else 0
            ratios[animal][region] = ratio
    return ratios


def compute_average_ratios(ratios):
    avg_ratios = {}
    std_ratios={}
    for region in ratios[next(iter(ratios))]:  # Loop over regions using the first animal's keys
        region_values = [ratios[animal][region] for animal in ratios]
        avg_ratios[region] = np.mean(region_values)
        std_ratios[region] = np.std(region_values)

    return avg_ratios,std_ratios



def compute_average_transitions_with_remained_new(data):
    avg_transitions = {
        'Sig to Non-Sig': {},
        'Non-Sig to Sig': {},
        'Remained Sig (Learning)': {},
        'Remained Sig (Learned)': {}
    }

    for region in data['478'].keys():  # Loop through regions
        # Collect values for each transition type across animals
        sig_to_nonsig_vals = [data[animal][region]['Sig to Non-Sig'] for animal in data]
        nonsig_to_sig_vals = [data[animal][region]['Non-Sig to Sig'] for animal in data]
        remained_sig_learning_vals = [data[animal][region]['Remained Sig (Learning)'] for animal in data]
        remained_sig_learned_vals = [data[animal][region]['Remained Sig (Learned)'] for animal in data]

        # Compute averages for each transition type
        avg_transitions['Sig to Non-Sig'][region] = np.mean(sig_to_nonsig_vals)
        avg_transitions['Non-Sig to Sig'][region] = np.mean(nonsig_to_sig_vals)
        avg_transitions['Remained Sig (Learning)'][region] = np.mean(remained_sig_learning_vals)
        avg_transitions['Remained Sig (Learned)'][region] = np.mean(remained_sig_learned_vals)

    return avg_transitions






def compute_significant_transitions_new(data):
    transitions = {}
    for animal in data['learning'].keys():
        cond1 = 'learning'
        cond2 = 'learned'

        # Define SI neurons for learning and learned phases
        si_learning = (
            data[cond1][animal]['TC_arm'] |
            data[cond1][animal]['TC_stem'] |
            data[cond1][animal]['PC_arm'] |
            data[cond1][animal]['PC_stem']
        )
        si_learned = (
            data[cond2][animal]['TC_arm'] |
            data[cond2][animal]['TC_stem'] |
            data[cond2][animal]['PC_arm'] |
            data[cond2][animal]['PC_stem']
        )

        # Sum of SI neurons in learning and learned phases
        total_cells_learning = np.sum(si_learning)
        total_cells_learned = np.sum(si_learned)

        transitions[animal] = {}
        for region in data['learning'][animal].keys():
            learning_cells = data['learning'][animal][region]
            learned_cells = data['learned'][animal][region]

            # Cells that were significant during learning and became insignificant in learned
            sig_to_nonsig = np.sum(learning_cells & ~learned_cells)
            
            # Cells that were insignificant during learning and became significant in learned
            nonsig_to_sig = np.sum(~learning_cells & learned_cells)
            
            # Cells that remained significant in both learning and learned phases
            remained_sig_learning = np.sum(learning_cells & learned_cells)
            remained_sig_learned = remained_sig_learning  # Same as remained_sig_learning, but normalized differently

            # Normalize transitions
            transitions[animal][region] = {
                'Sig to Non-Sig': sig_to_nonsig / total_cells_learning if total_cells_learning > 0 else 0,
                'Non-Sig to Sig': nonsig_to_sig / total_cells_learned if total_cells_learned > 0 else 0,
                'Remained Sig (Learning)': remained_sig_learning / total_cells_learning if total_cells_learning > 0 else 0,
                'Remained Sig (Learned)': remained_sig_learned / total_cells_learned if total_cells_learned > 0 else 0
            }
    return transitions






def plot_pc_turnedto_tc(rate_pc_learning_L,rate_pc_learning_R, rate_pc_to_nonpc_learned_R,rate_pc_to_nonpc_learned_L,significant_cells,animal,savefolder,y_ax,label='Left'):
    # this code shows the data of PC in learning phase that became TC in learned phase
    if label=='Left':
        label2='Right'
    elif label=='Right':
        label2='Left'
    sig_sort_idx_e = np.argsort(np.argmax(rate_pc_learning_L[significant_cells], axis=1))
    #, gridspec_kw={'width_ratios': [9, 9, 9,9]}
    # Create figure and axes
    #fig, (ax, ax2, ax3,ax4) = plt.subplots(2, 2, figsize=(6, 6))
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    ax, ax2, ax4, ax3 = axes.flatten()
    # Get max for normalization
    #max_o_e = np.max([np.max(rate_pc_learning_L[significant_cells][sig_sort_idx_e]), np.max(rate_pc_to_nonpc_learned_R[significant_cells][sig_sort_idx_e])])

    max_L = np.max(rate_pc_learning_L[significant_cells][sig_sort_idx_e], initial=-np.inf)
    max_R = np.max(rate_pc_to_nonpc_learned_R[significant_cells][sig_sort_idx_e], initial=-np.inf)

# Now compute the overall maximum
    max_o_e = np.max([max_L, max_R])
    if max_o_e == -np.inf:
        return  
    # Plot the rate maps for PC in Learning
    im1 = ax.pcolormesh(y_ax, np.arange(np.sum(significant_cells)), rate_pc_learning_L[significant_cells][sig_sort_idx_e] / max_o_e, vmin=0,vmax=1,rasterized=True)
    plt.colorbar(im1, ax=ax)
    ax.set_title('PC in Learning '+ label, fontsize=18)
    ax.set_xlabel('Position [norm]', fontsize=16)
    ax.set_ylabel('Cell #', fontsize=16)


    im2 = ax2.pcolormesh(y_ax, np.arange(np.sum(significant_cells)), rate_pc_learning_R[significant_cells][sig_sort_idx_e] / max_o_e, vmin=0,vmax=1, rasterized=True)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('PC in Learning '+ label2, fontsize=18)
    ax2.set_xlabel('Position [norm]', fontsize=16)
    ax2.set_ylabel('Cell #', fontsize=16)


    # Plot the rate maps for Non-PC in Learned
    im3 = ax3.pcolormesh(y_ax, np.arange(np.sum(significant_cells)), rate_pc_to_nonpc_learned_R[significant_cells][sig_sort_idx_e] / max_o_e, vmin=0,vmax=1,rasterized=True)
    plt.colorbar(im3, ax=ax3)

    ax3.set_title('TC in Learned '+label2, fontsize=18)
    ax3.set_xlabel('Position [norm]', fontsize=16)


    # Plot the rate maps for Non-PC in Learned
    im4 = ax4.pcolormesh(y_ax, np.arange(np.sum(significant_cells)), rate_pc_to_nonpc_learned_L[significant_cells][sig_sort_idx_e] / max_o_e, vmin=0,vmax=1,rasterized=True)
    plt.colorbar(im4, ax=ax4)
    ax4.set_title('TC in Learned '+ label, fontsize=18)
    ax4.set_xlabel('Position [norm]', fontsize=16)


    plt.suptitle('Only signficnats animal #:' +animal)
    plt.tight_layout()
    plt.savefig(savefolder+animal+label+'Heatmap PC to TC same position.svg', format='svg',dpi=300)
    plt.show()






def compute_max_correlation(rate_pc_learning_L, rate_pc_learned_R, overlap):
    """
    Compute the maximum correlation and stability status for each cell.
    
    Parameters:
    - rate_pc_learning_L: np.ndarray, rate maps for left runs (n x m)
    - rate_pc_learned_R: np.ndarray, rate maps for right runs (n x m)
    - overlap: int, the maximum shift allowed
    
    Returns:
    - max_corrs: np.ndarray, maximum correlation for each cell
    - stable_fields: np.ndarray, boolean array indicating cells with stable field
    """
    n, m = rate_pc_learning_L.shape
    max_corrs = np.zeros(n)  # Maximum correlations for each cell
    stable_fields = np.zeros(n, dtype=bool)  # Stability status for each cell

    for cell_idx in range(n):
        left_rate = rate_pc_learning_L[cell_idx]
        right_rate = rate_pc_learned_R[cell_idx]
        correlations = []

        # Calculate correlation for shifts within +/- overlap
        for shift in range(-overlap, overlap + 1):
            shifted_right_rate = np.roll(right_rate, shift)
            correlation = np.corrcoef(left_rate, shifted_right_rate)[0, 1]
            correlations.append((correlation, shift))
        
        # Find maximum correlation and corresponding shift
        max_corr, best_shift = max(correlations, key=lambda x: x[0])
        max_corrs[cell_idx] = max_corr

        # Determine if the field is stable based on the shift
        if abs(best_shift) <= 100:
            stable_fields[cell_idx] = True

    return max_corrs, stable_fields




def significant_pc_to_tc(Rates,Rates_sh,TC_learned,PC_to_Non_PC,animal):
    # get indices of the cells that are pc in learneg and became tc in learned
    epsilon=0
    pc_to_tc_learned = TC_learned[PC_to_Non_PC]# place cells that are TC after learning

    rate_pc_learning_L=(Rates['learning'][animal]['L'][PC_to_Non_PC])
    rate_pc_learning_R=(Rates['learning'][animal]['R'][PC_to_Non_PC])

    rate_pc_to_nonpc_learned_L=(Rates['learned'][animal]['L'][PC_to_Non_PC])
    rate_pc_to_nonpc_learned_R=(Rates['learned'][animal]['R'][PC_to_Non_PC])

    correlation_pc_non_pc_L=[np.corrcoef(x,y)[0][1] for x,y in zip(rate_pc_learning_L,rate_pc_to_nonpc_learned_R)]
    correlation_pc_non_pc_R=[np.corrcoef(x,y)[0][1] for x,y in zip(rate_pc_learning_R,rate_pc_to_nonpc_learned_L)]


    shuffle_numbers=len(Rates_sh['learning'][animal]['L'])
    correlation_pc_non_pc_sh_L=[]#np.zeros(shuffle_numbers)
    correlation_pc_non_pc_sh_R=[]
    for sh in np.arange(shuffle_numbers):
        #rate_sh_pc_learning=(Rates_sh['learning'][animal]['L'][sh][PC_to_Non_PC])
        rate_sh_pc_to_nonpc_learned_R=(Rates_sh['learned'][animal]['R'][sh][PC_to_Non_PC])
        correlation_pc_non_pc_sh_L.append([np.corrcoef(x,y+epsilon)[0][1] for x,y in zip(rate_pc_learning_L,rate_sh_pc_to_nonpc_learned_R)])

        rate_sh_pc_to_nonpc_learned_L=(Rates_sh['learned'][animal]['L'][sh][PC_to_Non_PC])
        correlation_pc_non_pc_sh_R.append([np.corrcoef(x,y+epsilon)[0][1] for x,y in zip(rate_pc_learning_R,rate_sh_pc_to_nonpc_learned_L)])



    pvals_L = np.sum([np.asarray(correlation_pc_non_pc_L)>np.asarray(x) for x in correlation_pc_non_pc_sh_L],axis=0)/shuffle_numbers
    significants2=pvals_L>.95
    significants_L = pc_to_tc_learned & significants2# cells that are signficant (learning PC with learned non-pc) and TC in learned

    pvals_R = np.sum([np.asarray(correlation_pc_non_pc_R)>np.asarray(x) for x in correlation_pc_non_pc_sh_R],axis=0)/shuffle_numbers
    significants2=pvals_R>.95
    significants_R = pc_to_tc_learned & significants2# cells that are signficant (learning PC with learned non-pc) and TC in learned

    return significants_L,significants_R,rate_pc_to_nonpc_learned_L,rate_pc_to_nonpc_learned_R,rate_pc_learning_L,rate_pc_learning_R




########################### Place_code_analysis #######################################


def get_rate_all(data,neuron_tspidx,occ_gau,yedges,sigma_yidx,expanded_all_burst_tidxs,num_neurons,y,y_ax,shuffle_spikes=False):
    # This function uses the spike time to estimate the rate maps of all cells 
    # Compute the place fields
    counts_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float) # for storing spike counts. Not used.
    p_all=  np.zeros(y_ax.shape[0]).astype(float)
    r_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float)  # for storing firing rates


    for nid in range(num_neurons):
        tspidx = neuron_tspidx[nid]

        
        # Exclude spike times occuring in the neighours(=1) of the burst events.
        masked_tspidx = exclude_idx(tspidx, expanded_all_burst_tidxs)
        
        if shuffle_spikes:# shuffle spike times for getting random place fields

            masked_tspidx,spk=spike_shuffle(data,masked_tspidx,None)


        # Get "spike positions"
        neuron_ysp = y[masked_tspidx]
        # Compute smoothed firing rate map
        rate, counts_ysp_gau,P_x = GetRate(neuron_ysp, occ_gau, yedges, sigma_yidx)

        #print(len(masked_tspidx),len(masked_tspidx_sh))
        
        # Store all of them
        r_all[nid, :] = rate
        counts_all[nid, :] = counts_ysp_gau  
        p_all=P_x
    return r_all,counts_all,p_all







def get_rate_all_clusters(data,target_cluids_long,y,occ_gau,yedges,sigma_yidx,expanded_all_burst_tidxs,num_neurons,y_ax,burst_tidxs,burst_cluids,shuffle_spikes):
    
    # Compute the place fields
    counts_all = np.zeros((num_neurons, y_ax.shape[0])).astype(float) # for storing spike counts. Not used.
    p_all=  np.zeros(y_ax.shape[0]).astype(float)
    r_all = np.zeros((len(target_cluids_long), y_ax.shape[0])).astype(float)  # for storing firing rates


    for cid,target_cluid in enumerate(target_cluids_long):
        #target_cluid=10
        # Select the burst events that belong to this cluster id
        clu_idxs = np.where(burst_cluids == target_cluid)[0]
        num_clu_bursts = len(clu_idxs)

        clu_idx = clu_idxs
        burst_tidx = burst_tidxs[clu_idx]



    #for nid in range(num_neurons):
        tspidx = burst_tidx

        
        # Exclude spike times occuring in the neighours(=1) of the burst events.
        masked_tspidx = exclude_idx(tspidx, expanded_all_burst_tidxs)
        if shuffle_spikes:# shuffle spike times for getting random place fields

            masked_tspidx,spk=spike_shuffle(data,masked_tspidx,None)

        #spikes.append(masked_tspidx)
        # Get "spike positions"
        neuron_ysp = y[masked_tspidx]
               



        neuron_ysp_sh = y[masked_tspidx]
        rate, counts_ysp_gau,P_x_sh = GetRate(neuron_ysp, occ_gau, yedges, sigma_yidx)

        #    # Store all of them
        r_all[cid, :] = rate
        #counts_all_sh[cid, :] = counts_ysp_gau 
        p_all=P_x_sh
    return r_all,counts_all,p_all