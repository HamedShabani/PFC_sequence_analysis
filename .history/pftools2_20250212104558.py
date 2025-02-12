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




def spike_shuffle(data,tspidx,oddeven=None):
    if oddeven==None:
        unqtrial=np.unique(data['trial_idx_mask'])
    elif oddeven=='odd':
        unqtrial=np.unique(data['trial_idx_mask'])[1::2]
    elif oddeven=='even':
        unqtrial=np.unique(data['trial_idx_mask'])[::2]
    
    t_all=np.arange(len(data['trial_idx_mask']))
    import random
    spk_sh=[]
    spk=[]
    for trl in unqtrial:
        trl_msk=(np.asarray(data['trial_idx_mask']==trl) )
        


        trl_time=t_all[trl_msk]
        spike_times=np.asarray(tspidx)[(np.isin(tspidx,trl_time))]
        if len(spike_times)>0:
            spk.extend(spike_times)
        #bst=np.asarray(np.isin(spike_times,trl_time))
        bst=np.isin(trl_time,np.asarray(spike_times))
        spike_times_sh=np.asarray(trl_time)[np.roll(bst,  random.randint(1, len(bst) - 1))]
        
        if len(spike_times_sh)>0:
            spk_sh.extend(spike_times_sh)
            #print(spike_times_sh,spike_times,trl)
    return(spk_sh,np.asarray(spk))


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



def mean_rate2(all_rates_L,P_x_f,num_neurons):
    r0_f = np.zeros(num_neurons)

    for nid in range(num_neurons):
        rates_f = all_rates_L[nid, :]
        # Compute the mean rate r0 here.
        r0_f[nid] =  np.sum(P_x_f * rates_f)
        # ================================= Exercise ends here ========================================
        pass

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




def compute_corrleation(sorted_l,sorted_r):
#    find the level of similarity of PF between left and right runs
    #inpute is the place fields of left and right runs
    # output is the correlation between left and right PFs
    # imprtatnt!!!: left argument (firts) is the one cells are certed by
    eps=0#.000000001# to avaoid nans 

    sorted_l_cut=sorted_l[np.argmax(sorted_l,axis=1)>500]+eps
    sorted_r_cut=sorted_r[np.argmax(sorted_l,axis=1)>500]+eps

    similarity_between_l_and_r_arms=[]# arms
    for l_cell,r_cell in (zip(sorted_l_cut,sorted_r_cut)):
        similarity_between_l_and_r_arms.append(np.corrcoef(l_cell,r_cell)[0,1])


    sorted_l_cut_stem=sorted_l[np.argmax(sorted_l,axis=1)<=500]+eps
    sorted_r_cut_stem=sorted_r[np.argmax(sorted_l,axis=1)<=500]+eps

    similarity_between_l_and_r_stem=[]# stem
    for l_cell,r_cell in (zip(sorted_l_cut_stem,sorted_r_cut_stem)):
        similarity_between_l_and_r_stem.append(np.corrcoef(l_cell,r_cell)[0,1])

    mask_cut=np.argmax(sorted_l,axis=1)>500


    similarity_between_l_and_r_all=[]# all
    for l_cell,r_cell in (zip(sorted_l,sorted_r)):
        similarity_between_l_and_r_all.append(np.corrcoef(l_cell,r_cell)[0,1])



    return similarity_between_l_and_r_arms,similarity_between_l_and_r_stem,similarity_between_l_and_r_all,mask_cut









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
def get_rate_all(data,neuron_tspidx,occ_gau,yedges,sigma_yidx,expanded_all_burst_tidxs,num_neurons,y_ax,y,shuffle_spikes=False):
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









def shuffling_cluster_rates_new(sorted_l,sorted_r,cluster_pc_fractions_L,cluster_pc_fractions_R,sig_sort_idx_l):    


    max_clstr=np.max(np.concatenate((cluster_pc_fractions_L['cluster_numbers'],cluster_pc_fractions_R['cluster_numbers'])))

    correlation_l_r,correlation_l_r_stem,similarity_between_l_and_r_all,mask_cut= compute_corrleation(sorted_l,sorted_r)

    rates_left_sig_org= np.zeros((int(max_clstr)+1,np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
    #rates_left_sig_org[cluster_pc_fractions_L['singnificnts_cluster_ids']]=cluster_pc_fractions_L['rate_significant']
    rates_left_sig_org[cluster_pc_fractions_L['cluster_numbers']]=cluster_pc_fractions_L['rate_all_clusters']# include all clusters instead of only si



    similarity_shuffled_cells=[]# compute the similarity for shuffled data
    for itr in range(len(cluster_pc_fractions_R['rate_shuffled_clusters'])):
        rates_right_all_sh= np.zeros((int(max_clstr)+1,np.shape(cluster_pc_fractions_L['rate_significant'])[1]))
        rates_right_all_sh[cluster_pc_fractions_R['cluster_numbers']]=cluster_pc_fractions_R['rate_shuffled_clusters'][itr]


        zero_mask=np.sum(rates_left_sig_org,axis=1)>0
        #rates_left_sig_org=rates_left_sig_org[zero_mask]
        #rates_right_all_sh=rates_right_all_sh[zero_mask]

        #sig_sort_idx_L_=np.argsort(np.argmax(rates_left_sig_org,axis=1))
        #sorted_l=np.asarray(rates_left_sig_org)[sig_sort_idx_l]
        sorted_r_sh=np.asarray(rates_right_all_sh)[sig_sort_idx_l]


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

    return p_val_cells,mask_cut




def significant_pc_to_tc_2(Rates,Rates_sh,TC_learned,PC_learning,PC_learned,animal,y_ax):
    # get indices of the cells that are pc in learneg and became tc in learned
    epsilon=0
    pc_to_tc_learned = TC_learned[PC_learning]# place cells that are TC after learning
    pc_to_pc_learned = PC_learned[PC_learning]# place cells that are TC after learning

    non_si_learned=~(TC_learned | PC_learned)

    pc_to_nonsi= non_si_learned[PC_learning]# PCs in learning that are niether pc nor tc after learning

    rate_pc_learning_L=(Rates['learning'][animal]['L'][PC_learning])
    rate_pc_learning_R=(Rates['learning'][animal]['R'][PC_learning])

    rate_pc_learned_L=(Rates['learned'][animal]['L'][PC_learning])
    rate_pc_learned_R=(Rates['learned'][animal]['R'][PC_learning])

    correlation_pc_L=[np.corrcoef(x,y)[0][1] for x,y in zip(rate_pc_learning_L,rate_pc_learned_R)]
    correlation_pc_R=[np.corrcoef(x,y)[0][1] for x,y in zip(rate_pc_learning_R,rate_pc_learned_L)]


    shuffle_numbers=len(Rates_sh['learning'][animal]['L'])
    correlation_pc_sh_L=[]#np.zeros(shuffle_numbers)
    correlation_pc_sh_R=[]
    for sh in np.arange(shuffle_numbers):
        #rate_sh_pc_learning=(Rates_sh['learning'][animal]['L'][sh][PC_to_Non_PC])
        rate_sh_pc_learned_R=(Rates_sh['learned'][animal]['R'][sh][PC_learning])
        correlation_pc_sh_L.append([np.corrcoef(x,y+epsilon)[0][1] for x,y in zip(rate_pc_learning_L,rate_sh_pc_learned_R)])

        rate_sh_pc_learned_L=(Rates_sh['learned'][animal]['L'][sh][PC_learning])
        correlation_pc_sh_R.append([np.corrcoef(x,y+epsilon)[0][1] for x,y in zip(rate_pc_learning_R,rate_sh_pc_learned_L)])

    
    overlap=500# total shift length
    max_corrs, stable_fields_L = compute_max_correlation(rate_pc_learning_L, rate_pc_learned_R, overlap)
    max_corrs, stable_fields_R = compute_max_correlation(rate_pc_learning_R, rate_pc_learned_L, overlap)
    significants_L=stable_fields_L
    significants_R=stable_fields_R




    # pvals_L = np.sum([np.asarray(correlation_pc_L)>np.asarray(x) for x in correlation_pc_sh_L],axis=0)/shuffle_numbers
    # significants_L=pvals_L>.95
    # #significants_L = pc_to_tc_learned & significants2# PC cells that are signficantly stable after learning and are TC in learned

    # pvals_R = np.sum([np.asarray(correlation_pc_R)>np.asarray(x) for x in correlation_pc_sh_R],axis=0)/shuffle_numbers
    # significants_R=pvals_R>.95
    # #significants_R = pc_to_tc_learned & significants2# PC cells that are signficantly stable  after learning and are TC in learned

    significant_TC_stbl=(significants_L | significants_R) & pc_to_tc_learned# PC cells that are signficantly stable after learning and are TC in learned
    significant_PC_stbl=(significants_L | significants_R) & pc_to_pc_learned# PC cells that are signficantly stable after learning and are PC in learned
    significant_nonsi_stbl= pc_to_nonsi# PC cells that are signficantly stable after learning but are not TC or PC in learned

    TC_unstbl=~(significants_L | significants_R) & pc_to_tc_learned# PC cells that are signficantly stable after learning and are TC in learned
    PC_unstbl=~(significants_L | significants_R) & pc_to_pc_learned# PC cells that are signficantly stable after learning and are TC in learned



    if 1:
        significant_TC_stbl_L=(significants_L ) & pc_to_tc_learned# PC cells that are signficantly stable after learning and are TC in learned

            # Create figure and axes
        fig, (ax, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={'width_ratios': [9, 9, 9,9]})
        # Get max for normalization

        sig_sort_idx_e = np.argsort(np.argmax(rate_pc_learning_L[significant_TC_stbl_L], axis=1))

        max_o_e = np.max([np.max(rate_pc_learning_L[significant_TC_stbl_L][sig_sort_idx_e]), np.max(rate_pc_learned_R[significant_TC_stbl_L][sig_sort_idx_e])])

        # Plot the rate maps for PC in Learning
        im1 = ax.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learning_L[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im1, ax=ax)
        ax.set_title('PC in Learning (Left)', fontsize=18)
        ax.set_xlabel('Position [norm]', fontsize=16)
        ax.set_ylabel('Cell #', fontsize=16)

        # Plot the rate maps for PC in Learning
        im1 = ax2.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learning_R[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im1, ax=ax2)
        ax2.set_title('PC in Learning (Right)', fontsize=18)
        ax2.set_xlabel('Position [norm]', fontsize=16)
        ax2.set_ylabel('Cell #', fontsize=16)

        # Plot the rate maps for PC in Learning
        im1 = ax3.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learned_L[ significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im1, ax=ax3)
        ax3.set_title('TC in Learned (Left)', fontsize=10)
        ax3.set_xlabel('Position [norm]', fontsize=16)
        ax3.set_ylabel('Cell #', fontsize=16)

        # Plot the rate maps for PC in Learning
        im4 = ax4.pcolormesh(y_ax, np.arange(np.sum(significant_TC_stbl_L)), rate_pc_learned_R[significant_TC_stbl_L][sig_sort_idx_e] / max_o_e, rasterized=True)
        plt.colorbar(im4, ax=ax4)
        ax4.set_title('TC in Learned(Right)', fontsize=10)
        ax4.set_xlabel('Position [norm]', fontsize=16)
        ax4.set_ylabel('Cell #', fontsize=16)

    return significant_TC_stbl,significant_PC_stbl,TC_unstbl,PC_unstbl,significant_nonsi_stbl



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








def distance_cell_to_cluster_type2(cluster_rate_all_animals,data_animal,Cluster_types_all,rate_L_R,cell_types_all,animal_name,plotrates=False):
    '''#Cluster_types_all : Index of clusters and their identity
    #rate_L_R : Ratemap of all cells
    #  cell_types_all: index of cells and their identity
    # This function find the distance between peaks of rate maps of different cell types and peaks of different clusters of the same type
        '''

    c2c_Distances={}
    c2c_Distances_sh={}
    for cell_tpe in Cluster_types_all[animal_name].keys():# type
        
        rate_cluster=[]
        for iclstr,tmp_nbr in enumerate(Cluster_types_all[animal_name][cell_tpe]):# cluster
            
            template_cell_idx=data_animal['template'][tmp_nbr]# template cell indices for a type

            cluster_cll_idx=cell_types_all[animal_name][cell_tpe]#  cell indices for a type

            cluster_and_cell_type=np.intersect1d(cluster_cll_idx,template_cell_idx)# cell of a type that are in a cluster type



            rate_cell=[]
            distance_cell_to_cluster_org=[]
            distance_cell_to_cluster_shuffled=[]
            #for icl,cell_nbr in enumerate(cell_types_all[animal_name][cell_tpe]):# cells

            for icl,cell_nbr in enumerate(cluster_and_cell_type):# cells

                
                #dist=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                distlr=[]
                distlr.append(np.argmax(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                distlr.append(np.argmax(rate_L_R[animal_name+'_R']['rate_all'][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))

                distance_cell_to_cluster_org.extend(distlr)
                
                distance_cell_to_cluster_shuffled_cell=[]# shuffled dictance for each cell

                distlsh=[]
                for sh in range(1000):
                    rmap=rate_L_R[animal_name+'_L']['rate_all'][cell_nbr]
                    rate_sh= np.roll(rmap,  random.randint(1, len(rmap) - 1))# shuffle rate map
                    #dist_sh=np.abs(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                    distlsh.append(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))

                    #distance_cell_to_cluster_shuffled_cell.append(dist_sh)
                distrsh=[]
                for sh in range(1000):
                    rmap=rate_L_R[animal_name+'_R']['rate_all'][cell_nbr]
                    rate_sh= np.roll(rmap,  random.randint(1, len(rmap) - 1))# shuffle rate map
                    #dist_sh=np.abs(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                    distrsh.append(np.argmax(rate_sh)-np.argmax(cluster_rate_all_animals[animal_name]['R'][tmp_nbr]))

                    #distance_cell_to_cluster_shuffled_cell.append(dist_sh)

                


                # mean_distance_sh=[]
                # for ish in range(len(rate_L_R[animal_name+'_L']['rate_all_shuffled'])):
                #     dist_sh=np.abs(np.argmax(rate_L_R[animal_name+'_L']['rate_all_shuffled'][ish][cell_nbr])-np.argmax(cluster_rate_all_animals[animal_name]['L'][tmp_nbr]))
                #     distance_cell_to_cluster_shuffled_cell.append(dist_sh)
                distance_cell_to_cluster_shuffled.extend(np.vstack((distlsh,distrsh)))# distances of all shuffled data to cluster
                
                
                #p_val=np.sum(dist>distance_cell_to_cluster_shuffled_cell)/len(distance_cell_to_cluster_shuffled_cell)
                #print(cell_tpe+str(tmp_nbr)+'_'+str(cell_nbr)+':',p_val)
                rate_cell.append(rate_L_R[animal_name+'_L']['rate_all'][cell_nbr])

                # plt.figure()
                # plt.hist(distance_cell_to_cluster_shuffled_cell,alpha=.2)
                # plt.vlines(dist,0,100,color='r')
                # plt.title(cell_tpe+str(tmp_nbr)+'_'+str(cell_nbr)+':'+str(p_val))


                #mean_distance_sh.append(np.mean(distance_cell_to_cluster_shuffled_cell))# avg of shuffled for each cell

            #p_val_all_cells=np.sum(mean_distance_sh>distance_cell_to_cluster)/len(distance_cell_to_cluster_shuffled_cell)# compare distance to clusters for all cells with the average of the shuffled


            # c2c_Distances[cell_tpe+str(iclstr)]=distance_cell_to_cluster_org
            # c2c_Distances_sh[cell_tpe+str(iclstr)]=distance_cell_to_cluster_shuffled

            c2c_Distances[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_org
            c2c_Distances_sh[cell_tpe+str(tmp_nbr)]=distance_cell_to_cluster_shuffled

            rate_cluster.append(cluster_rate_all_animals[animal_name]['L'][tmp_nbr])

            if 0:
                fig, (ax,ax2) = plt.subplots(1, 2,figsize=(16, 8))
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
                # plt.title('distance to cluster'+ animal_name+cell_tpe+str(tmp_nbr)

    return  c2c_Distances,c2c_Distances_sh


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