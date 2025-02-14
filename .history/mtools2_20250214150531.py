# code from laptop

import enum
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import networkx as nx
import scipy.cluster.hierarchy as scich
import copy
from sklearn.manifold import TSNE
import scipy.io as spio
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn import cluster as cluster2
from sklearn import  datasets, mixture
from scipy.stats import ranksums
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib as mpl
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.signal import find_peaks
from helper_functions import separate_in,separate_in_2d_array,linearize_2d_track_single_run
import os
Tspare=.5




def extract_seq(times):
    stimes=np.sort(times)
    ids=np.argsort(times)
    return ids[np.where(~np.isnan(stimes))[0]], ids



    
# def rankseq(s1,s2):

#     #compute rank order correlation between sequences

#     # set things straight
#     s1=np.array(s1).flatten()
#     s2=np.array(s2).flatten()
#     l1=len(s1)
#     l2=len(s2)
    
#     #difference matrix
#     d=np.ones((l1,1))*s2 - (np.ones((l2,1))*s1).transpose()
  
#     # binary identity matrix
#     d=(d==0)
  

#     # make s0 the shorter sequence
#     s=s1
#     s0=s2
#     l0=l2
#     ln=l1
#     if l1<l2:
#         s=s2;
#         s0=s1;
#         l0=l1
#         ln=l2
#         d=d.transpose()


 
        
#     #compute cell overlap (neurons contained in both)
#     minseq=s[np.where(np.sum(d,axis=1)>0)[0]];
#     lm=len(minseq)
  
#     # delete neurons from the shorter sequence that are not in the minimal
#     # sequence
#     #
    
#     d0=np.ones((l0,1))*minseq - (np.ones((lm,1))*s0).transpose()
#     d0=(d0==0)
#     s0=s0[np.sum(d0,axis=1)>0]
#     l0=len(s0)
  
  
#     #find ordinal rank in the shorter sequence
#     dd=np.ones((lm,1))*s0 - (np.ones((l0,1))*minseq).transpose()
  
#     #compute spearmans r
#     if len(dd)>1:
#         ids=np.argmin(np.abs(dd),axis=0)
        
#         rc = np.corrcoef(np.arange(len(ids)),ids)[0,1]
#         ln=len(ids)
#     else:
#         rc=np.nan;
#         ln=np.nan
  
   
    
#     return rc, ln



# def shuffle(narr):

#     nrep=10000
    
#     ret=[]
#     for n in narr:

#         s2=np.arange(n)

#         rval=np.zeros(nrep)
#         for m in range(nrep):

#             s1=np.random.permutation(n)

#             rval[m],dummy=rankseq(s1,s2)

#         c=np.mean(rval)
#         sd=np.std(rval)
#         prctl=np.quantile(rval,.95)
        
#         ret.append([n, c, sd, prctl])

#     return ret

    
# def allmot(seqs,nrm):

#     nseqs=len(seqs)

#     narr=np.array(nrm)[:,0]

#     corrmat=np.zeros((nseqs,nseqs))
#     zmat=np.zeros((nseqs,nseqs))
#     bmat=np.zeros((nseqs,nseqs))
#     pval=np.zeros(nseqs)
#     nsig=np.zeros(nseqs)
    
#     for ns in range(nseqs):

#         s1=seqs[ns]

#         zmat[ns,ns]=np.nan
#         bmat[ns,ns]=np.nan
#         for ms in range(ns+1,nseqs):

            

#             s2=seqs[ms]

#             rc,ln=rankseq(s1,s2)
            

#             if ln>=50:
#                 mns=nrm[-1]
#             else:
#                 whichone=np.array(np.where(ln==narr)).flatten()
#                 if len(whichone)==0:
#                     mns=np.empty(4)
#                     mns[:]=np.nan
#                 else:
#                     mns=nrm[whichone[0]]
                    
                    
#             ztmp=(rc-mns[1])/mns[2]
#             #print(ns,mns,ztmp)
#             corrmat[ns,ms]=rc
#             corrmat[ms,ns]=rc

#             zmat[ns,ms]=ztmp
#             zmat[ms,ns]=ztmp
#             bmat[ns,ms]=1.*(ztmp>mns[3])
#             bmat[ms,ns]=1.*(ztmp>mns[3])

#         nsig[ns] = np.nansum(bmat[ns,:])
#         pval[ns] = 1-binom.cdf(nsig[ns],nseqs-1,.05)


#     rep_index = nsig/np.std(nsig)


#     return rep_index, nsig, pval, bmat, zmat, corrmat
import numpy as np

def rankseq(s1, s2):
    """
    Compute the Spearman rank order correlation coefficient between two sequences.

    This function calculates the rank order correlation between two sequences, `s1` and `s2`,
    by first determining the overlap between their elements (interpreted as neurons) and then
    computing the Spearman correlation between the ordinal positions of the overlapping elements.
    If one sequence is shorter than the other, it is used as the reference for ranking.

    Parameters
    ----------
    s1 : array_like
        First input sequence (e.g., a list or 1D numpy array).
    s2 : array_like
        Second input sequence (e.g., a list or 1D numpy array).

    Returns
    -------
    rc : float
        Spearman rank correlation coefficient between the two sequences. Returns NaN if the
        correlation cannot be computed (e.g., due to insufficient overlapping elements).
    ln : int or float
        The number of overlapping elements (neurons) used in the correlation computation.
        Returns NaN if the correlation is not computed.
    """
    # Ensure the sequences are flattened numpy arrays.
    s1 = np.array(s1).flatten()
    s2 = np.array(s2).flatten()
    l1 = len(s1)
    l2 = len(s2)
    
    # Create a difference matrix where each element compares an element from s1 to an element from s2.
    d = np.ones((l1, 1)) * s2 - (np.ones((l2, 1)) * s1).transpose()
    # Convert the difference matrix to a binary identity matrix (True where elements are equal).
    d = (d == 0)
    
    # Choose the longer sequence as 's' and the shorter as 's0'. Adjust the difference matrix accordingly.
    s = s1
    s0 = s2
    ln = l1
    if l1 < l2:
        s = s2
        s0 = s1
        ln = l2
        d = d.transpose()
        
    # Identify overlapping elements (neurons present in both sequences).
    minseq = s[np.where(np.sum(d, axis=1) > 0)[0]]
    
    # Remove elements from the shorter sequence that are not in the overlapping set.
    d0 = np.ones((len(s0), 1)) * minseq - (np.ones((len(minseq), 1)) * s0).transpose()
    d0 = (d0 == 0)
    s0 = s0[np.sum(d0, axis=1) > 0]
    
    # Prepare a matrix to determine ordinal ranking differences.
    dd = np.ones((len(minseq), 1)) * s0 - (np.ones((len(s0), 1)) * minseq).transpose()
    
    # Compute Spearman's rank correlation coefficient if there is more than one overlapping element.
    if len(dd) > 1:
        ids = np.argmin(np.abs(dd), axis=0)
        rc = np.corrcoef(np.arange(len(ids)), ids)[0, 1]
        ln = len(ids)
    else:
        rc = np.nan
        ln = np.nan
    
    return rc, ln


def shuffle(narr):
    """
    Compute statistics of Spearman rank correlations for randomly shuffled sequences.

    For each integer `n` in the input array `narr`, this function generates 10,000 random
    permutations of the sequence [0, 1, ..., n-1] and computes the Spearman rank correlation
    with the sorted sequence using the `rankseq` function. It then calculates the mean, standard
    deviation, and 95th percentile of these correlation coefficients.

    Parameters
    ----------
    narr : array_like
        An array or list of integers. Each integer represents the length of a sequence to be
        shuffled and analyzed.

    Returns
    -------
    ret : list of lists
        A list where each sublist corresponds to an input `n` and has the format:
        [n, mean_correlation, std_correlation, correlation_95th_percentile].
    """
    nrep = 10000  # Number of repetitions for the shuffling procedure.
    ret = []
    for n in narr:
        s2 = np.arange(n)  # Create a sorted sequence of length n.
        rval = np.zeros(nrep)
        for m in range(nrep):
            s1 = np.random.permutation(n)  # Generate a random permutation.
            rval[m], _ = rankseq(s1, s2)     # Compute rank correlation with the sorted sequence.
        c = np.mean(rval)
        sd = np.std(rval)
        prctl = np.quantile(rval, 0.95)
        ret.append([n, c, sd, prctl])
    return ret



# def check_template(seqs,tmpl,nrm):

#     nseqs=len(seqs)
#     s1=np.array(tmpl).flatten()

#     narr=np.array(nrm)[:,0]    
#     sig=np.zeros(nseqs)
#     zval=np.zeros(nseqs)
        
#     for ns in range(nseqs):
    
#         s2=seqs[ns]
#         rc,ln=rankseq(s1,s2)
#         #print(rc,ln)
        
#         if ln>=50:
#             mns=nrm[-1]
#         else:
#             whichone=np.array(np.where(ln==narr)).flatten()
#             if len(whichone)==0:
#                 mns=np.empty(4)
#                 mns[:]=np.nan
#             else:
#                 mns=nrm[whichone[0]]


#         ztmp=(rc-mns[1])/mns[2]
#         sig[ns]=1.*(ztmp>mns[3])
#         zval[ns]=ztmp


#     return zval,sig


import numpy as np

def templates(bursts, seqs, nrm, ids_clust, min_ratio=2):
    """
    Generate templates for neural burst sequences based on clustering and quality criteria.

    This function computes a template sequence for each cluster of bursts. For each cluster, 
    it calculates a representative sequence (template) using either the mean (for 2D burst data) 
    or an averaged sequence (for 3D burst data). The template is then evaluated with the 
    `check_template` function to compute a quality metric (radius). Finally, clusters whose 
    quality ratio (within-cluster / across-cluster correlation) is below `min_ratio` are excluded.

    Parameters
    ----------
    bursts : array_like
        A 2D or 3D array of burst sequences. If 2D, each row represents a burst; if 3D, bursts 
        may include additional dimensions (e.g., time bins x neurons).
    seqs : list or array_like
        A list of individual sequences (e.g., neuron spike orderings) corresponding to the bursts.
    nrm : array_like
        An array containing normalization parameters (e.g., mean, standard deviation, and threshold)
        used for z-score computation in template quality checking.
    ids_clust : array_like
        An array of cluster IDs assigning each burst (or sequence) to a specific cluster.
    min_ratio : float, optional
        The minimum ratio (within-cluster / across-cluster correlation) required for a cluster's 
        template to be accepted. Clusters with a lower ratio are excluded (default is 2).

    Returns
    -------
    retval : dict
        Dictionary containing:
            - 'adj': List of adjustment (significance) values for each accepted cluster.
            - 'template': List of accepted template sequences.
            - 'clist': List of indices for bursts belonging to each accepted cluster.
            - 'radius': List of radius values (quality measure) for each accepted template.
            - 'seqs': Original input sequences (for reference).
            - 'ids_clust': Original cluster IDs (for reference).
            - 'bursts': Original burst data (for reference).
            - 'ratio': List of computed quality ratios for each cluster.
            - 'exclude': List of cluster indices that were excluded based on the quality ratio.
    """
    # Initialize the return dictionary with placeholders.
    retval = {
        'adj': [],
        'template': [],
        'clist': [],
        'radius': [],
        'seqs': [],
        'ids_clust': [],
        'bursts': [],
        'ratio': []
    }
    
    # Save the original input data for reference.
    retval['seqs'].append(seqs)         # Added by Hamed
    retval['ids_clust'].append(ids_clust) # Added by Hamed
    retval['bursts'].append(bursts)       # Added by Hamed

    # Process each cluster based on its identifier.
    for nc in range(max(ids_clust) + 1):
        # Find the indices of bursts belonging to the current cluster.
        clist = np.where(ids_clust == nc)[0]
        
        # Determine the template based on the dimensionality of the bursts data.
        if np.array(bursts).ndim == 2:
            # For 2D data: compute the mean burst across the cluster, then sort.
            mns = np.nanmean(np.array(bursts)[clist, :], axis=0)
            tmp = np.argsort(mns)
            # Remove any indices corresponding to NaN values.
            temp = tmp[~np.isnan(np.sort(mns))]
        elif np.array(bursts).ndim == 3:
            # For 3D data: compute an average sequence using the helper function.
            temp, _ = average_sequence(np.array(bursts)[clist, :, :])
        else:
            raise ValueError("Unsupported burst data dimensionality.")
            
        # Check the template quality against the provided sequences using normalization parameters.
        chck = check_template(seqs, temp, nrm)
        # Define the template's "radius" as the mean significance flag scaled by 30.
        radius = np.mean(chck[1]) * 30
        
        # Store the computed template and related metrics.
        retval['template'].append(temp)
        retval['clist'].append(clist)
        retval['radius'].append(radius)
        retval['adj'].append(chck[1])

    # Evaluate the quality of each cluster template using within- and across-cluster measures.
    crit = within_across(retval['adj'], ids_clust)
    retval.update({'exclude': []})
    
    # Compute a quality ratio for each cluster and mark clusters for exclusion if they do not meet the threshold.
    for nc in range(len(retval['radius'])):
        ratio = crit['within'][nc] / crit['across'][nc]
        retval['ratio'].append(ratio)
        if ratio < min_ratio:
            retval['exclude'].append(nc)
                               
    # Remove clusters that failed to meet the quality threshold.
    retval['template'] = [i for j, i in enumerate(retval['template']) if j not in retval['exclude']]
    retval['clist'] = [i for j, i in enumerate(retval['clist']) if j not in retval['exclude']]
    retval['radius'] = [i for j, i in enumerate(retval['radius']) if j not in retval['exclude']]
    retval['adj'] = [i for j, i in enumerate(retval['adj']) if j not in retval['exclude']]
    # Optionally, similar filtering can be applied to 'ratio' and 'ids_clust' if desired.

    return retval


def check_template(seqs, tmpl, nrm):
    """
    Evaluate a template against a set of sequences using rank correlation and compute z-scores.

    For each sequence in `seqs`, this function calculates the Spearman rank order correlation 
    with the template `tmpl` (using the previously defined `rankseq` function). Based on the 
    overlap length between the template and each sequence, appropriate normalization parameters 
    are selected from `nrm`. A z-score is then computed, and a binary significance flag is set 
    if the z-score exceeds a threshold.

    Parameters
    ----------
    seqs : list or array_like
        A list of sequences (e.g., neuron spike orders) to compare with the template.
    tmpl : array_like
        The template sequence against which each sequence is evaluated.
    nrm : array_like
        Normalization parameters. Each row is expected to contain reference values 
        (e.g., [?, mean, std, threshold]) for z-score calculation. The first column is used 
        to match the overlap length.

    Returns
    -------
    zval : numpy.ndarray
        Array of z-scores for each sequence based on the rank correlation with the template.
    sig : numpy.ndarray
        Binary array (0 or 1) indicating whether each sequence's z-score exceeds the threshold.
    """
    nseqs = len(seqs)
    # Flatten the template sequence for comparison.
    s1 = np.array(tmpl).flatten()

    # Extract the first column from nrm to be used for matching the overlap length.
    narr = np.array(nrm)[:, 0]
    
    sig = np.zeros(nseqs)
    zval = np.zeros(nseqs)
        
    for ns in range(nseqs):
        s2 = seqs[ns]
        rc, ln = rankseq(s1, s2)  # Compute rank correlation and overlap length
        
        # Choose normalization parameters based on the overlap length.
        if ln >= 50:
            mns = nrm[-1]  # Use the last row if sufficient overlap exists.
        else:
            # Find the matching normalization parameters for the given overlap length.
            whichone = np.array(np.where(ln == narr)).flatten()
            if len(whichone) == 0:
                mns = np.empty(4)
                mns[:] = np.nan
            else:
                mns = nrm[whichone[0]]

        # Calculate the z-score using the reference mean and standard deviation.
        ztmp = (rc - mns[1]) / mns[2]
        # Set the significance flag if the z-score exceeds the threshold.
        sig[ns] = 1.0 * (ztmp > mns[3])
        zval[ns] = ztmp

    return zval, sig


def average_sequence(burst_set):
    """
    Compute an average sequence template from a set of bursts.

    This function computes the mean burst across a set of bursts (averaging over the first axis) 
    and then calculates a weighted average sequence (coefficient sequence) based on burst intensities.
    The resulting sequence is generated by sorting the weighted coefficients and filtering out any 
    NaN values.

    Parameters
    ----------
    burst_set : array_like
        A 3D numpy array where bursts are arranged along the first dimension.

    Returns
    -------
    seq : numpy.ndarray
        An array representing the ordered indices of the average sequence template.
    cofseq : numpy.ndarray
        The weighted coefficient sequence computed from the burst data.
    """
    # Compute the mean burst over all bursts in the set.
    vec = np.mean(burst_set, axis=0)
    # Generate an index array corresponding to columns.
    itax = np.arange(vec.shape[1])
    # Sum each row to obtain total burst intensity.
    nvec = np.sum(vec, axis=1)
    # Compute the center of mass:

    cofseq = (itax @ vec.transpose()) / nvec
    # Sort indices based on the weighted coefficients.
    tmp = np.argsort(cofseq)
    # Remove indices corresponding to NaN coefficient values.
    seq = tmp[~np.isnan(np.sort(cofseq))]

    return seq, cofseq


def within_across(adj, ids_clust):
    """
    Compute within-cluster and across-cluster mean adjustments.

    For each cluster, this function calculates the mean adjustment (e.g., significance flag) 
    for sequences within the cluster (within-cluster) and for those outside the cluster 
    (across-cluster).

    Parameters
    ----------
    adj : list of array_like
        A list where each element corresponds to a cluster and contains adjustment values.
    ids_clust : array_like
        An array of cluster IDs assigning each sequence (or burst) to a cluster.

    Returns
    -------
    ret : dict
        Dictionary with keys:
            - 'within': List of mean adjustment values computed for sequences within each cluster.
            - 'across': List of mean adjustment values computed for sequences outside each cluster.
    """
    ret = {'within': [], 'across': []}
    # Loop over each cluster's adjustment values.
    for nc in range(len(adj)):
        # Get indices of sequences inside and outside the current cluster.
        idin = np.where(ids_clust == nc)[0]
        idout = np.where(~(ids_clust == nc))[0]
        # Compute the mean adjustments.
        within = np.mean(adj[nc][idin])
        across = np.mean(adj[nc][idout])
        ret['within'].append(within)
        ret['across'].append(across)

    return ret



import numpy as np
from scipy.signal import find_peaks

def binned_burst(dt, winlen, thr_burts, fs, timewins):
    """
    Bin the population data into time windows and detect bursts within each window. For the manuscript we did not perform binning)

    This function segments the input data matrix `dt` (cells x time) into smaller time 
    windows. If `timewins` is empty, it automatically generates time window boundaries based 
    on the window length `winlen`. For each time window, it calls the `popbursts` function to 
    detect bursts and extract the associated burst sequences. The results from each window 
    are then aggregated.

    Parameters
    ----------
    dt : np.ndarray
        2D array of neural data (cells x time) representing population activity.
    winlen : int or float
        Length of each time window (in samples) used to bin the data.
    thr_burts : float
        Threshold multiplier for burst detection used by the `popbursts` function.
    fs : float
        Sampling frequency of the data (in Hz).
    timewins : list or array-like
        List of time window boundaries (indices). If empty, time windows are automatically 
        generated based on the data length and `winlen`.

    Returns
    -------
    poprate : list
        Aggregated population firing rate over all time windows.
    id_peaks_trl : list
        Aggregated indices (relative to the full data) of detected burst peaks.
    bursts_tmp : list
        List of detected burst segments (submatrices) from each time window.
    seqs_tmp : list
        List of burst sequences (orderings of cell activations) derived from each burst.
    """
    # If no time window boundaries are provided, generate them automatically.
    if len(timewins) < 1:
        # Generate equally spaced boundaries across the number of time samples in dt.
        timewins = f2(np.ceil(np.linspace(0, dt.shape[1], int(np.ceil(dt.shape[1] / winlen)))))
        if dt.shape[1] < winlen:
            timewins = [0, dt.shape[1]]
    
    # Initialize lists to store the aggregated results.
    seqs_tmp = []         # Burst sequences
    bursts_tmp = []        # Burst segments (submatrices)
    spike_time = []        # (Optional) Spike times per burst (not used in this version)
    id_peaks_trl = []      # Peak indices per trial/window
    poprate = []           # Population firing rate over all windows
    lsdt = 0               # Cumulative time offset for each window
    raw_data = []          # Raw data segments (for debugging or further processing)
    Rasters = []           # Raster data (if needed)
    lentrials = []         # Trial lengths (for debugging or further processing)
    id_peaks_all = []      # All detected peak indices (not used later)
    spike_time_all = []    # Aggregated spike times (if available)
    
    dtlen = 0  # Cumulative length of processed data (in samples)
    i0 = 0     # Index offset for each window
    
    # Loop over each pair of consecutive time window boundaries.
    for twin in np.arange(len(timewins) - 1):
        # Extract the segment of data corresponding to the current time window.
        sdt = dt[:, timewins[twin]:timewins[twin+1]]
        # Check that the segment is valid (here, sum > -10000 acts as a basic validity check).
        if np.sum(sdt[:]) > -10000:
            # Call popbursts to detect bursts in the current time window.
            bursts_tmp_s, seqs_tmp_s, id_peaks_trl_s, poprate_s = popbursts(sdt, thr_burts, fs)
            
            # Store raw data from the current window (transposed for convenience).
            raw_data.extend(np.transpose(dt[:, timewins[twin]:timewins[twin+1]]))
            lentrials.extend(np.transpose(dt[:, timewins[twin]:timewins[twin+1]]))
            
            # Adjust the peak indices to the full-session time by adding the cumulative offset.
            id_peaks_all.extend([k + dtlen for k in id_peaks_trl_s])
            dtlen += sdt.shape[1]
            
            # Update the cumulative offset for the burst peaks.
            poprate.extend(poprate_s)
            id_peaks_trl.extend(np.array(id_peaks_trl_s) + lsdt)
            bursts_tmp.extend(bursts_tmp_s)
            seqs_tmp.extend(seqs_tmp_s)
            lsdt += sdt.shape[1]
            
        # (Optional) Merge spike times from all windows if available.
        if len(spike_time_all) > 0:
            spike_time_mrg = merge_spike_times(spike_time_all, dt.shape[0])
        else:
            spike_time_mrg = []
        
        # Update the offset for the next window.
        i0 = i0 + sdt.shape[1]
    
    return poprate, id_peaks_trl, bursts_tmp, seqs_tmp


def popbursts(mat, sig, fs, kwidth=0, minwidth=1):
    """
    Detect population bursts and extract burst sequences from a data matrix.

    This function computes a population rate by summing the activity across cells in the input 
    matrix `mat`. If a kernel width (`kwidth`) is provided (or adjusted), the population rate is 
    smoothed using a Gaussian kernel. A threshold for burst detection is then set based on the 
    mean and standard deviation of the population rate scaled by `sig`. The `find_peaks` function 
    identifies burst peaks, and for each peak, a burst segment (a submatrix) is extracted around 
    the peak. Optionally, the burst segment can be randomized in time. Finally, a "sequence" is 
    computed for each burst using a center-of-mass calculation on the burst segment.

    Parameters
    ----------
    mat : np.ndarray
        2D data matrix (cells x time) for which bursts are to be detected.
    sig : float
        Multiplier for the standard deviation to set the burst detection threshold.
    fs : float
        Sampling frequency of the data.
    kwidth : float, optional
        Kernel width for Gaussian smoothing of the population rate. If less than 1/(3*fs), it is 
        set to 1/(3*fs). Default is 0.
    minwidth : float, optional
        Minimum width (in samples) for a detected burst peak. Default is 1.

    Returns
    -------
    vec : list
        List of detected burst segments (submatrices extracted from the data matrix).
    seq : list
        List of sequences corresponding to each burst segment. Each sequence is derived from the 
        center-of-mass calculation of the burst.
    peaks : list
        List of peak indices (in samples) where bursts were detected.
    poprate : np.ndarray
        The population rate vector computed over the entire time axis of `mat`.
    """
    global Tspare  # Tspare should be defined elsewhere in your code (e.g., burst duration in seconds. We used 0.5 sec for the manuscript)
    random_time = False
    import random

    # If kwidth is too small, set it to 1/(3*fs) and compute poprate as the simple sum.
    if kwidth < 1/(3*fs):
        kwidth = 1/(3*fs)
        poprate = np.sum(mat, axis=0)
    else:
        # Define a time axis for the Gaussian kernel.
        tax = np.arange(-3 * kwidth, 3 * kwidth, 1/fs)
        # Convolve the summed activity with a Gaussian kernel.
        poprate = np.convolve(np.sum(mat, axis=0),
                              np.exp(-(tax**2) / (2 * kwidth**2)) / kwidth,
                              mode='same')
    
    # Set threshold for burst detection based on mean and standard deviation of poprate.
    thresh = np.mean(poprate) + sig * np.std(poprate)
    # 'spare' is the minimum separation (in samples) between detected bursts.
    spare = int(Tspare * fs)
    print('Burst length is ', Tspare)
    
    vec = []     # List to store burst segments (submatrices)
    # Use scipy.signal.find_peaks to detect peaks in the population rate.
    idpeaks, _ = find_peaks(poprate, height=thresh, width=(minwidth, spare * 10), distance=spare)
    
    peaks = []   # List to store the final accepted peak indices
    idprev = -1  # Variable to store the previous accepted peak index

    # Loop over each detected peak.
    for idpeak in idpeaks:
        # Check that the current peak is sufficiently separated from the previous one and that a 
        # burst segment can be extracted without going out of bounds.
        if (idpeak - idprev > spare) and (idpeak - int(spare/2) >= 0) and (idpeak + int(spare/2) + 1 <= mat.shape[1]):
            # Extract a burst segment centered at the peak.
            vtmp = mat[:, idpeak - int(spare/2) : idpeak + int(spare/2) + 1]

            # Optionally, randomize the order of time bins in the burst segment.
            if random_time:
                column_indices = np.arange(vtmp.shape[1])
                np.random.shuffle(column_indices)
                print('bursts are randomized!!!')
                vtmp = vtmp[:, column_indices]

            # Only accept burst segments with activity from at least 5 cells.
            if np.sum(np.sum(vtmp, axis=1) > 0) > 4:
                vec.append(vtmp)
                peaks.append(idpeak)  # Record the peak index (in samples)
                idprev = idpeak

    # If any bursts were detected, set up an index array for the time bins.
    if len(vec) > 0:
        itax = np.arange(vec[0].shape[1])
    else:
        itax = np.array([])

    seq = []  # List to store the computed sequence for each burst
    # For each burst segment, compute the "center-of-mass" based sequence.
    for nv in range(len(vec)):
        # Sum activity per cell within the burst segment.
        nvec = np.sum(vec[nv], axis=1)
        # Compute the weighted average index (center-of-mass) for each cell.
        cofseq = (itax @ vec[nv].transpose()) / nvec
        # Get the order of cell activation based on the center-of-mass.
        tmp = np.argsort(cofseq)
        # Remove any indices corresponding to NaN values (if any).
        seq.append(tmp[~np.isnan(np.sort(cofseq))])
    
    return vec, seq, peaks, poprate






# def popbursts(mat,sig,fs,kwidth=0,minwidth=1):
#     global Tspare
#     random_time=False
#     import random
#     #poprate=np.sum(mat,axis=0)
#     if kwidth<1/(3*fs):
#         kwidth=1/(3*fs)
#         poprate=np.sum(mat,axis=0)
#     else:
#         tax=np.arange(-3*kwidth,3*kwidth,1/fs)
#         poprate=np.convolve(np.sum(mat,axis=0),np.exp(-(tax**2)/2/kwidth**2)/kwidth,'same')
    
#     thresh=np.mean(poprate)+sig*np.std(poprate)
#     spare=int(Tspare*fs)
#     print('Burst length is ' ,Tspare)
    
#     #mask=(poprate>=thresh)*1.
#     #mask=1.*(np.diff(mask)==1)
#     #ids=np.where(mask>0)[0]
    
#     vec=[]
    
#     idpeaks, _ = find_peaks(poprate, height=thresh, width=(minwidth,spare*10), distance=spare)
    
#     peaks=[]
#     idprev=-1
#     #for n in range(len(ids)):
#     #
#     #    id0=ids[n]
#     #
#     #    if id0+spare<=mat.shape[1]:
#     #        idpeak=np.argmax(poprate[id0:id0+spare])+id0
#     #        if (idpeak-idprev>spare)*(idpeak-int(spare/2)>=0)*(idpeak+int(spare/2)+1<=mat.shape[1]):
#     #            vtmp=mat[:,idpeak-int(spare/2):idpeak+int(spare/2)+1]
#     #            if len(np.where(np.sum(vtmp,axis=1)>0)[0])>4:#mimimum 5 active cells
#     #                vec.append(vtmp)
#     #                peaks.append(idpeak/fs)
#     #                idprev=idpeak
    
#     for idpeak in idpeaks:
#         if (idpeak-idprev>spare)*(idpeak-int(spare/2)>=0)*(idpeak+int(spare/2)+1<=mat.shape[1]):
#             vtmp = mat[:,idpeak-int(spare/2):idpeak+int(spare/2)+1]

#             if random_time:
#                 column_indices = np.arange(np.shape(vtmp)[1])
#                 np.random.shuffle(column_indices)
#                 print('bursts are randomized!!!')
#                 # Use the shuffled index array to rearrange the columns
#                 vtmp = vtmp[:, column_indices]




#             if len(np.where(np.sum(vtmp,axis=1)>0)[0])>4:
#                 vec.append(vtmp)
#                 #peaks.append(idpeak/fs)

#                 peaks.append(idpeak)# Hamed Chaned

#                 idprev=idpeak
                
                
#     if len(vec)>0:
        
#         # if random_time==True:
#         #     itax=np.arange(vec[0].shape[1])
#         #     random.shuffle(itax)
#         #     print('bursts are randomized!!!')
#         # else:

#         itax=np.arange(vec[0].shape[1])



#     seq=[]
#     for nv in range(len(vec)):
#         nvec=np.sum(vec[nv],axis=1)
#         cofseq=(itax@vec[nv].transpose())/nvec
#         tmp=np.argsort(cofseq)
#         seq.append(tmp[~np.isnan(np.sort(cofseq))])
                   
      
    
    
    
#     return vec,seq,peaks,poprate








# def average_sequence(burst_set):
    
#     vec=np.mean(burst_set,axis=0)
#     itax=np.arange(vec.shape[1])
#     nvec=np.sum(vec,axis=1)
#     cofseq=(itax@vec.transpose())/nvec
#     tmp=np.argsort(cofseq)
#     seq=tmp[~np.isnan(np.sort(cofseq))]

#     return seq,cofseq


# def within_across(adj,ids_clust):

#     ret={'within':[], 'across':[]}
#     for nc in range(len(adj)):
#         idin = np.where((ids_clust==nc))[0]
#         idout = np.where(~(ids_clust==nc))[0]
#         within = np.mean(adj[nc][idin])
#         across = np.mean(adj[nc][idout])
#         ret['within'].append(within)
#         ret['across'].append(across)

#     return ret

# def templates(bursts,seqs,nrm,ids_clust,min_ratio=2):

#     retval={'adj':[], 'template':[], 'clist':[], 'radius':[],'seqs':[],'ids_clust':[], 'bursts':[], 'ratio':[]}
#     retval['seqs'].append(seqs)# added by Hamed
#     retval['ids_clust'].append(ids_clust)# added by Hamed
#     retval['bursts'].append(bursts)# added by Hamed

#     for nc in range(max(ids_clust)+1):
#         clist=(np.where(ids_clust==nc)[0])
#         if np.array(bursts).ndim==2:
#             mns = np.nanmean(np.array(bursts)[clist,:], axis=0)
#             tmp = np.argsort(mns)
#             temp = tmp[~np.isnan(np.sort(mns))]

#         elif np.array(bursts).ndim==3:
#             temp,dummy = average_sequence(np.array(bursts)[clist,:,:])
            
#         chck=check_template(seqs,temp,nrm)
#         radius=np.mean(chck[1])*30
#         retval['template'].append(temp)
#         retval['clist'].append(clist)
#         retval['radius'].append(radius)
#         retval['adj'].append(chck[1])


#     crit = within_across(retval['adj'],ids_clust)
#     retval.update({'exclude':[]})
#     for nc in range(len(retval['radius'])):
#         ratio=crit['within'][nc]/crit['across'][nc]
#         best_ratio=crit['within'][nc]
#         #retval['within_ratio'].append(best_ratio)
#         retval['ratio'].append(ratio)

#         #print(nc, ": ", ratio)
#         if ratio<min_ratio:
#             retval['exclude'].append(nc)
                               
                               
#     retval['template'] = [i for j, i in enumerate(retval['template']) if j not in retval['exclude']]# remove bad clusters
#     retval['clist'] = [i for j, i in enumerate(retval['clist']) if j not in retval['exclude']]# remove bad clusters
#     retval['radius'] = [i for j, i in enumerate(retval['radius']) if j not in retval['exclude']]# remove bad clusters
#     retval['adj'] = [i for j, i in enumerate(retval['adj']) if j not in retval['exclude']]# remove bad clusters
#     #retval['ratio'] = [i for j, i in enumerate(retval['ratio']) if j not in retval['exclude']]# remove bad clusters
#     #retval['ids_clust'] = [i for j, i in enumerate(retval['ids_clust']) if j not in retval['exclude']]# remove bad clusters


#     #
#     return retval





def graph(seqAll,nrm,temp_info=[],temp_infoD=[]):
    cmap=plt.get_cmap('Set3')
    options = {"alpha": 0.7}
    
    nclust=0
    if len(temp_info)>0:
        temp=temp_info['template']
        clist=temp_info['clist']
        radius=temp_info['radius']        

    nclust=len(radius)
    
    for nD in range(len(temp_infoD)):
        temp.extend(temp_infoD[nD]['template'])
        clist.extend(temp_infoD[nD]['clist'])
        radius.extend(temp_infoD[nD]['radius'])        


        
    lseqs=len(seqAll)-len(radius)
    dum1,dum2,dum3,bmat,zmat,dum=allmot(seqAll,nrm)
    #
    
    G=nx.Graph()
    for jm in range(bmat.shape[1]):
        for jn in range(bmat.shape[0]):
            if bmat[jn,jm]>0:
                G.add_edge(jn,jm,weight=zmat[jn,jm])

    pos = nx.spring_layout(G, seed=12647)  # positions for all nodes
    #
    allist=list(pos.keys())

    for nc in range(nclust):
        if np.sum(np.array(temp_info['exclude'])==nc):
            continue

        colrgb=np.array(cmap.colors[np.mod(nc,10)]).reshape(1,3)
        q=[]
        for ok in allist:
            if np.sum(clist[nc]==ok):
                q.append(ok)

        tmpnodelabel=lseqs+nc
        if np.sum(np.array(allist)==tmpnodelabel)>0:
            nx.draw_networkx_nodes(G, pos, nodelist=q, node_size=5*radius[nc],edgecolors=["black"],node_color=colrgb, **options)
            nx.draw_networkx_nodes(G, pos, nodelist=[tmpnodelabel], edgecolors=["red"], node_size=200,node_color=colrgb, **options)
            labels = {x: x-lseqs+1 for x in G.nodes if x >=lseqs}
            nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='b')

    nc=nclust
    colD=["red","black"]
    for nD in range(len(temp_infoD)):
        for ncloc in range(len(temp_infoD[nD]['radius'])):
            
            if np.sum(np.array(temp_infoD[nD]['exclude'])==ncloc):
                nc=nc+1
                continue


            tmpnodelabel=lseqs+nc
            nx.draw_networkx_nodes(G, pos, nodelist=[tmpnodelabel], edgecolors=[colD[nD]], node_size=50,node_color=[colD[1-nD]], **options)
            nc=nc+1

            
        
    #nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.05)

    return G,pos





# def cluster(bmat,zmat,params):
#     cmat=np.zeros_like(zmat)
#     cmat[~np.isnan(zmat)]=bmat[~np.isnan(zmat)]

#     if params['name']=='AHC':
#         fac=params['fac']
#         clnmbr=params['clnbr']
#         pdist=scich.distance.pdist(cmat)
#         lkg=scich.linkage(pdist, method='ward')
#         c_th=np.max(pdist)*fac
#         ids_clust = scich.fcluster(lkg,c_th,criterion='distance')-1
#         #ids_clust = scich.fcluster(lkg,clnmbr,criterion='maxclust')-1


#     #gmm = mixture.GaussianMixture( n_components=2, covariance_type="full" ).fit(cmat)
#     #ids_clust = gmm.predict(cmat)

#     ## estimate bandwidth for mean shift
#     #bandwidth = cluster2.estimate_bandwidth(cmat, quantile=.5)
#     #ms = cluster2.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(cmat)
#     #ids_clust = ms.labels_

#     #optics = OPTICS(    max_eps=.3).fit(cmat) 
#     #ids_clust=optics.labels_
#     elif params['name']=='DB':

#         DBSCAN_cluster = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(cmat) 
#         ids_clust= DBSCAN_cluster.labels_

#     #two_means = cluster2.MiniBatchKMeans(n_clusters = 2).fit(cmat)
#     #ids_clust= two_means.labels_

#     return ids_clust

from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
import numpy as np

def cluster(bmat, zmat, params):
    """
    Cluster detected sequences based on a similarity matrix.

    This function creates a clustering matrix from `bmat` by using the non-NaN entries 
    in `zmat` to filter out invalid values. Depending on the specified clustering method in 
    `params['name']`, it applies either Agglomerative Hierarchical Clustering (AHC) or DBSCAN.

    For AHC (Used for the manuscript):
      - A pairwise distance matrix is computed from the filtered clustering matrix.
      - Ward's linkage method is used to perform hierarchical clustering.
      - The clustering threshold is set as a fraction (`fac`) of the maximum distance in the pairwise distances.
      - Clusters are formed based on this threshold.

    For DBSCAN:
      - The DBSCAN algorithm is used directly on the filtered clustering matrix with the 
        provided `eps` (maximum neighborhood distance) and `min_samples` (minimum number of points).

    Parameters
    ----------
    bmat : array_like
        A matrix representing pairwise similarity or connectivity between sequences.
    zmat : array_like
        A matrix of the same shape as `bmat`, where entries are z-scored similarity values.
        NaN entries in `zmat` indicate invalid or missing comparisons and are used to filter `bmat`.
    params : dict
        Dictionary containing clustering parameters. Depending on the method:
          - For Agglomerative Hierarchical Clustering (AHC):
              * params['name'] should be 'AHC'
              * params['fac']: Factor to multiply the maximum pairwise distance to obtain the clustering threshold.
              * (Optional) params['clnbr']: Expected number of clusters (alternative criterion, currently commented out).
          - For DBSCAN:
              * params['name'] should be 'DB'
              * params['eps']: Maximum distance between two samples for them to be considered as neighbors.
              * params['min_samples']: Minimum number of samples required to form a cluster.

    Returns
    -------
    ids_clust : ndarray
        An array of cluster labels (starting at 0) assigned to each sequence.
    """
    # Create a clustering matrix (cmat) by keeping bmat values only where zmat is valid (not NaN)
    cmat = np.zeros_like(zmat)
    cmat[~np.isnan(zmat)] = bmat[~np.isnan(zmat)]

    # Apply clustering based on the selected method
    if params['name'] == 'AHC':
        # Extract parameters for hierarchical clustering
        fac = params['fac']
        # clnmbr = params['clnbr']  # Alternative: use a fixed number of clusters (commented out)
        
        # Compute the pairwise distance matrix from cmat.
        # pdist returns a condensed distance matrix.
        pdist_matrix = pdist(cmat)
        
        # Perform hierarchical/agglomerative clustering using Ward's method.
        linkage_matrix = sch.linkage(pdist_matrix, method='ward')
        
        # Determine a clustering threshold: a fraction of the maximum pairwise distance.
        threshold = np.max(pdist_matrix) * fac
        
        # Form clusters using the distance criterion.
        # Subtract 1 so that cluster IDs start at 0.
        ids_clust = sch.fcluster(linkage_matrix, threshold, criterion='distance') - 1
        
        # Alternative: use a fixed number of clusters (commented out)
        # ids_clust = sch.fcluster(linkage_matrix, clnmbr, criterion='maxclust') - 1

    elif params['name'] == 'DB':
        # Use DBSCAN clustering with the provided eps and min_samples parameters.
        dbscan_cluster = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(cmat)
        ids_clust = dbscan_cluster.labels_

    # Other clustering methods (e.g., Gaussian Mixture Models, MeanShift, OPTICS, KMeans) could be added here.
    # For example:
    # gmm = mixture.GaussianMixture(n_components=2, covariance_type="full").fit(cmat)
    # ids_clust = gmm.predict(cmat)

    return ids_clust




def plot_samples(templates,isort,samples,ax):

    ncells=len(templates[0])
    linsp=np.arange(1,ncells+1)

    nshift=0
    for nt in range(len(templates)):
        ax.plot(nshift+templates[nt][isort], linsp, 'r.') 
        nshift += 1
        
    for nsmpl in range(len(samples)):
        ax.plot(nshift+samples[nsmpl][isort],linsp, '.k')
        nshift += 1
        
    ax.set_xlabel('Seq. #')
    ax.set_ylabel('Cell #')
   

def t_from_mat(slist,fs):
    itax=np.arange(slist[0].shape[1])     
    
    cofseq=[]
    for ns in range(len(slist)):
        vec=slist[ns]
        nvec=np.sum(vec,axis=1)
        cofseq.append((itax@vec.transpose())/nvec/fs)
         
         
    return cofseq
    

def cluster_graph(vec,seq,nrm):
    repid,nsig,pval,bmat,zmat=allmot(seq,nrm);

    #clustering of motifs
    ids_clust=cluster(bmat,zmat,1.5)
       
    # compute templates
    temp_info=templates(vec,seq,nrm,ids_clust)
    

    seqAll=copy.copy(seq)
    for nc in range(len(temp_info['radius'])):    
        seqAll.append(temp_info['template'][nc])
        
    G, pos = graph(seqAll,nrm,temp_info)

    return G, pos


   
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def ranksum_pvalue(rates,conds):
    Ratesappend = rates    
    condnmbr = len(Ratesappend)
    pvals=np.zeros((condnmbr,condnmbr))
    condname=[]
    for i in range(condnmbr):
        for j in range(condnmbr):
            
            pvals[i,j] = ranksums(list(zip(*Ratesappend[i]))[0], list(zip(*Ratesappend[j]))[0])[1]
        
        condname.append( [name for name in conds if name in list(zip(*Ratesappend[i]))[1][0]][-1])
    return pvals,condname    




def deconv(signal,tau):
    f_sampling=20
   
    decon=np.diff(signal,axis=1)*f_sampling+(signal[:,0:-1]+signal[:,1:])/2/tau
    decon=decon/np.max(decon,axis=1).reshape(-1,1)# normalize to 1
    return decon




def velocity(x,y,fs):
    dx=np.diff(x)
    dy=np.diff(y)

    speed=np.zeros_like(x)

    v=np.zeros((2,len(x)))

    v[0,:-1]=dx*fs
    v[1,:-1]=dy*fs
    v[:,-1]=v[:,-2]

    ndur=int(0.3*fs)
    v[0,:]=np.convolve(v[0,:],np.ones(ndur)/ndur,'same')
    
    speed=np.sqrt(np.sum(v**2,axis=0))

    phi=np.angle(v[0,:]+v[1,:]*1j)/np.pi*180
            
    return v,speed,phi




def f(x):
    return int(x)
f2 = np.vectorize(f)




def binned_burst(dt,winlen,thr_burts,fs,timewins):

    if len(timewins)<1:# if the time windows for finding peaks are not privided(timewins=[])

        timewins=f2(np.ceil(np.linspace(0,dt.shape[1],int(np.ceil(dt.shape[1]/winlen)))))
        if dt.shape[1]<winlen:
            timewins=[0,dt.shape[1]]
        
    
    seqs_tmp=[]
    bursts_tmp=[]
    spike_time=[]
    id_peaks_trl=[]

    poprate=[]
    lsdt=0

    raw_data=[]
    Rasters=[]

    lentrials=[]
    id_peaks_all=[]
    spike_time_all=[]
    #spike_time_all= [[] for i in range(len(timewins)-1)]

    dtlen=0
    i0=0
    for twin in np.arange(len(timewins)-1):
        sdt=dt[:,timewins[twin]:timewins[twin+1]]
        if np.sum(sdt[:])>-10000:
            #bursts_tmp_s,seqs_tmp_s,ids_temp_s,raster_s,spike_time_s,id_peaks_trl_s,poprate_s = popbursts(sdt,thr_burts,fs)
            #bursts_tmp_s,seqs_tmp_s,ids_temp_s,raster_s,spike_time_s,id_peaks_trl_s,poprate_s = popbursts_new(sdt,thr_burts,fs)

            bursts_tmp_s,seqs_tmp_s,id_peaks_trl_s,poprate_s = popbursts(sdt,thr_burts,fs)


            
            raw_data.extend(np.transpose(dt[:,timewins[twin]:timewins[twin+1]]))
           # Rasters.extend(np.transpose(raster_s))
            
            lentrials.extend(np.transpose(dt[:,timewins[twin]:timewins[twin+1]]))
            

            
            id_peaks_all.extend([k+dtlen for k in id_peaks_trl_s])# the time of sequences for one condition
            dtlen += np.shape(dt[:,timewins[twin]:timewins[twin+1]])[1]
            #plt.plot(id_peaks_all,[len(x) for x in seqs])
            ncells=dt.shape[0]
            
            # if np.sum(spike_time_s[:])==0:
            #     continue# this should be removed 
            

            
            
            # if len([x for x in spike_time_s if len(x)>0])>0:# concat spiketime of all jumps and add the time of each jump
            #     spike_time_add=[]
            #     for spkt in spike_time_s:
            #         if spkt:
            #             spike_time_add.append([i+i0 for i in spkt])# each cell
            #         else:
            #             spike_time_add.append([])

            #     spike_time_all.append(np.array(spike_time_add))
            #     #spike_time_all=(np.array(spike_time_add))
                



                
                # if 0:
                #     spike_time_allt=np.transpose(spike_time_all)                        
                #     Spike_time=[[] for kk in range(ncells)]

                #     for nkk in range(ncells):                                                        
                #         my_list=[i for i in spike_time_allt[nkk] if i]
                #         flat_list = [num for sublist in my_list for num in sublist]                          
                #         Spike_time[nkk].append((flat_list))
                #     plt.figure()
                #     for lne in range(len(Spike_time)):
                #         if Spike_time[lne]:
                #             plt.vlines(np.array(Spike_time[lne]),ymin=lne,  ymax=lne+1)
            i0=i0+sdt.shape[1]


            
            poprate.extend(poprate_s)
            id_peaks_trl.extend(np.array(id_peaks_trl_s)+lsdt)
            #spike_time.extend(spike_time_s)  
            bursts_tmp.extend(bursts_tmp_s)
            seqs_tmp.extend(seqs_tmp_s)
            lsdt = lsdt+sdt.shape[1]

        if len(spike_time_all)>0:
            spike_time_mrg = merge_spike_times(spike_time_all,ncells)
        else:
            spike_time_mrg=[]
        
    return poprate,id_peaks_trl,bursts_tmp,seqs_tmp



def merge_spike_times(spike_time_all,ncells):
    spike_time_mrg=[[] for kk in range(ncells) ]

    for y in range(ncells):#neuron
        spike_time_tmp=[]
        for x in spike_time_all:# twin
            spike_time_tmp.extend(x[y])
        spike_time_mrg[y]=spike_time_tmp
    return(spike_time_mrg)




def find_correct_index(cond_trials,correct_trials):

    findex=np.zeros(len(cond_trials),dtype=bool)
    for i in range(len(cond_trials)):
        a=[]
        for j in range(len(correct_trials)):# check if the condition index are in correct trials
            a.append((cond_trials[i][0] >=correct_trials[j][0]) & (cond_trials[i][1] <=correct_trials[j][1]))
        if sum(a)>0:
            findex[i]=True
    return(findex)







def merge_clusters2(temp_info, nrm, seqs, bursts,plot_figure, min_ratio, z_thr):
    mask_badcluster = (temp_info['ids_clust'][0] == [temp_info['exclude']][0])

    ids_clust = temp_info['ids_clust'][0]
    ids_clust[mask_badcluster] = -1  # replace bad clusters with -1
    ids_clust = np.expand_dims(ids_clust, 0)
    temp_info['ids_clust'] = ids_clust

    temp_info2 = copy.deepcopy(temp_info)
    repid_temp, nsig_temp, pval_temp, bmat_temp, zmat_temp, corrmat_temp = allmot(
        temp_info['template'], nrm)
    zmat_temp_new = np.nan_to_num(zmat_temp, copy=True)
    bmat_temp_new = np.array(np.nan_to_num(
        bmat_temp, copy=True, nan=0.0, posinf=None, neginf=None))

    max_dix = np.array([(np.max(x), i, np.argmax(x))
                       for i, x in enumerate(np.triu(zmat_temp_new))])
    max_dix_sort = max_dix[np.argsort(np.array(max_dix)[:, 0])[::-1], :]
    vmax = max_dix_sort[0][0]
    vmin = max_dix_sort[-1][0]
    mxid = max_dix_sort[0]
    cl_id = 0
    #ids_clust = np.zeros_like(temp_info['ids_clust'])
    #ids_clust = np.copy(temp_info2['ids_clust'])
    for nt in range(len(temp_info['template'])):
        # if nt >4:
        #     break

        # apply zmat value=1 as a constraint for merging
        if (bmat_temp_new[mxid[1].astype(int)][mxid[2].astype(int)] == 1) & (zmat_temp_new[mxid[1].astype(int)][mxid[2].astype(int)] > z_thr):
            ids_clust = np.copy(temp_info2['ids_clust'])
            print(np.shape(bmat_temp_new))
            cluster_mix_mask = (np.squeeze(temp_info2['ids_clust']) == mxid[2]) | (np.squeeze(
                temp_info2['ids_clust']) == mxid[1])  # find index of seqs in two similar clusters
            # print(i,sum(cluster_mix_mask))
            #burst_mixed_clusters = np.array(np.squeeze(temp_info2['bursts']))[cluster_mix_mask,:,:]
            #mixed_temp,dummy = mot.average_sequence(np.squeeze(burst_mixed_clusters))
            # ids_clust[cluster_mix_mask]=mxid[1]*np.ones(sum(cluster_mix_mask))
            # temp_info_mixed['ids_clust'].extend(np.array(cl_id*np.ones(sum(cluster_mix_mask)),dtype=int))
            # temp_info_mixed['clist'].append(np.where(cluster_mix_mask)[0])# refers to sequences
            # temp_info_mixed['bursts'].extend(burst_mixed_clusters)
            # temp_info_mixed['template'].append(mixed_temp)# refers to neurons

            #print(nt, mxid[1],mxid[2])
            # temp_info2['ids_clust'][0][cluster_mix_mask]=cl_id*np.ones(sum(cluster_mix_mask))
            ids_clust[0][cluster_mix_mask] = mxid[2]
            u, ids_clust = np.unique(ids_clust[0], return_inverse=True)
            ids_clust[mask_badcluster] = -1  # keep index of bad clusters

            a = np.arange(0, np.max(ids_clust))
            # if index of a cluster is changed
            if len(np.where((np.array([sum(x == ids_clust) for x in a]) == 0))[0]):

                # find the cluster that is removed according to exclustion criteria and remove it from ids_clust
                b = np.where((np.array([sum(x == ids_clust) for x in a]) == 0))[
                    0][0]
                # modify ids_clust numbers
                ids_clust = np.where(ids_clust > b, ids_clust-1, 0)
                ids_clust[mask_badcluster] = -1

            del temp_info2
            temp_info2 = templates(
                bursts, seqs, nrm, ids_clust, min_ratio=min_ratio)
            repid_temp, nsig_temp, pval_temp, bmat_temp, zmat_temp, corrmat_temp = allmot(
                temp_info2['template'], nrm)
            print('exclude cluster is ',
                  temp_info2['exclude'], 'mixed clusters are ', mxid)
            #del ids_clust
            # print((temp_info2['ratio']))

            zmat_temp_new = np.nan_to_num(zmat_temp, copy=True)
            bmat_temp_new = np.array(np.nan_to_num(
                bmat_temp, copy=True, nan=0.0, posinf=None, neginf=None))


            if plot_figure:
                plt.figure()
                # plt.plot(ids_clust[0][cluster_mix_mask])
                # plt.plot(temp_info['ids_clust'][0][cluster_mix_mask])
                plt.subplot(121)
                plt.imshow(zmat_temp_new, vmax=vmax, vmin=vmin)
                plt.colorbar(fraction=0.046, pad=0.04)

                plt.subplot(122)
                plt.imshow(bmat_temp_new)

            max_dix = np.array([(np.max(x), i, np.argmax(x))
                               for i, x in enumerate(np.triu(zmat_temp_new))])
            max_dix_sort = max_dix[np.argsort(
                np.array(max_dix)[:, 0])[::-1], :]
            mxid = max_dix_sort[0]
            cl_id += 1

    return temp_info2, mask_badcluster





def find_correct_index(cond_trials,correct_trials):

    findex=np.zeros(len(cond_trials),dtype=bool)
    trlnbr=[]
    for i in range(len(cond_trials)):
        a=[]
        
        for j in range(len(correct_trials)):# check if the condition index are in correct trials
            a.append((cond_trials[i][0] >=correct_trials[j][0]) & (cond_trials[i][1] <=correct_trials[j][1]))
          
            
        
        if sum(a)>0:
            findex[i]=True
            trlnbr.extend(np.asarray(np.where(a))+1)
    return findex,trlnbr




def apply_masks(sess_info,Masks,cond_numbers,cond_name,sessin_numbers,odd_even,sess_name,trial_type,phase):

    run_data={'idpeaks_cells':[[] for _ in range(len(sess_info['Spike_times_cells']))]}

    if odd_even != None: 
        mask_odd= np.asarray(Masks['odd_even'])==odd_even# mask for odd or even trials(0/1)
        mask_odd_seqs= np.asarray(Masks['odd_even_seqs'])==odd_even# mask for sequences of odd or even trials(0/1)
        mask_odd_fr= np.asarray(Masks['odd_even_fr'])==odd_even

    else:
        mask_odd=np.ones(len(Masks['odd_even'])).astype(bool)
        mask_odd_seqs= np.ones(len(Masks['odd_even_seqs'])).astype(bool)# mask for sequences of odd or even trials(0/1)
        mask_odd_fr=  np.ones(len(Masks['odd_even_fr'])).astype(bool) 




    if phase != None:
        mask_phase= np.asarray(Masks['phases'])==phase# learning or learned
        mask_phase_seqs= np.asarray(Masks['bursts_phase'])==phase
        #mask_phase_cell= np.asarray(Masks['cell_phase'][celid])==phase
        mask_phase_cell= [ np.asarray(x) == phase for x in Masks['cell_phase'] ]


        mask_phase_fr= np.asarray(Masks['fr_phase'])==phase

        if phase==0:
            phase_name='learning'
        elif phase==1:
            phase_name='learned'
    else:
        mask_phase=np.ones(len(Masks['phases'])).astype(bool)# both learning and learned
        mask_phase_seqs=np.ones(len(Masks['bursts_phase'])).astype(bool)# both learning and learned
        #mask_phase_cell=np.ones(len(Masks['cell_phase'][celid])).astype(bool)# both learning and learned
        mask_phase_cell=[np.ones(len(x)).astype(bool) for x in Masks['cell_phase'] ]
        mask_phase_fr=np.ones(len(Masks['fr_phase'])).astype(bool)# both learning and learned


        phase_name='learning and learned'



    mask_correct= np.asarray(Masks['correct_failed'])==trial_type# mask for correct or failed trials(0/1)
    mask_correct_seqs= np.asarray(Masks['correct_failed_seqs'])==trial_type

    #sessin_number=2
    #celid=11

    mask_sess_burst=np.zeros_like(np.asarray(Masks['bursts_sess'])).astype(bool)
    #mask_sess_cell=np.zeros_like(np.asarray(Masks['cell_sess'][celid])).astype(bool)
    mask_sess_t=np.zeros_like(np.asarray(Masks['sessions'])).astype(bool)
    mask_sess_fr=np.zeros_like(np.asarray(Masks['fr_sess'])).astype(bool)
    sess_names=str()
    for sess_nbr in sessin_numbers:
        mask_sess_burst+= np.asarray(Masks['bursts_sess'])==sess_nbr
        #mask_sess_cell+= np.asarray( Masks['cell_sess'][celid])==sess_nbr
        mask_sess_t+= np.asarray(Masks['sessions'])==sess_nbr
        mask_sess_fr+= np.asarray(Masks['fr_sess'])==sess_nbr
        sess_name_=list(sess_name.keys())[sess_nbr]
        sess_names= sess_names+ ' and '  +sess_name_[:-4]

    mask_sess_cell=[ np.zeros_like(np.asarray(x)).astype(bool) for x in Masks['cell_phase'] ]

    for celid in range(len(mask_sess_cell)):
        for sess_nbr in sessin_numbers:
            mask_sess_cell[celid]+= np.asarray( Masks['cell_sess'][celid])==sess_nbr





    mask_cond_burst_side=np.zeros_like(np.asarray(Masks['bursts_cond'])).astype(bool)
    #mask_cond_cell_side=np.zeros_like(np.asarray(Masks['cell_cond'][celid])).astype(bool)
    mask_cond_t_side=np.zeros_like(np.asarray(Masks['conditions'])).astype(bool)
    mask_cond_fr=np.zeros_like(np.asarray(Masks['fr_cond'])).astype(bool)

    cond_names=str()
    for cond_nbr in cond_numbers:
        mask_cond_burst_side += np.asarray(Masks['bursts_cond'])==cond_nbr
        #mask_cond_cell_side += np.asarray(Masks['cell_cond'][celid])==cond_nbr
        mask_cond_t_side += np.asarray(Masks['conditions'])==cond_nbr
        mask_cond_fr += np.asarray(Masks['fr_cond'])==cond_nbr
        cond_name_=list(cond_name.keys())[cond_nbr]
        cond_names= cond_names+ ' and '  +cond_name_



    mask_cond_cell=[ np.asarray(x).astype(bool) for x in Masks['cell_cond'] ]
    for celid in range(len(mask_cond_cell)):
        for cond_nbr in cond_numbers:
            mask_cond_cell[celid] += np.asarray(Masks['cell_cond'][celid])==cond_nbr

    # cond_number=cond_number1 # 'outward_L_side'
    # mask_cond_burst_side= np.asarray(Masks['bursts_cond'])==cond_number
    # mask_cond_cell_side= np.asarray(Masks['cell_cond'][celid])==cond_number
    # mask_cond_t_side= np.asarray(Masks['conditions'])==cond_number
    # mask_cond_fr= np.asarray(Masks['fr_cond'])==cond_number
    mask_correct_fr= np.asarray(Masks['correct_failed_fr'])==trial_type
    # conndname1=list(cond_names.keys())[cond_number]
    # cond_number=cond_number2 # 'outward_L_center'
    # mask_cond_burst_center= np.asarray(Masks['bursts_cond'])==cond_number
    # mask_cond_cell_center= np.asarray(Masks['cell_cond'][celid])==cond_number
    # mask_cond_t_center= np.asarray(Masks['conditions'])==cond_number
    # conndname2=list(cond_names.keys())[cond_number]

    mask_cond_burst= mask_cond_burst_side
    #mask_cond_cell= mask_cond_cell_side
    mask_cond_t= mask_cond_t_side

    
    run_data['ids_clust']=np.asarray(sess_info['ids_clust'])[mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']]
    run_data['idpeaks']=np.asarray(sess_info['id_peaks'])[mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']]
    run_data['seqs']=np.asarray(sess_info['seqs'])[mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']]
    run_data['bursts']=np.asarray(sess_info['bursts'])[mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']]

    #run_data['idpeaks_cells']=np.asarray(sess_info['Spike_times_cells'][celid])[mask_sess_cell & mask_cond_cell & mask_phase_cell]

    for celid in range(len(sess_info['Spike_times_cells'])):
        run_data['idpeaks_cells'][celid] = np.asarray(sess_info['Spike_times_cells'][celid])[mask_sess_cell[celid] & mask_cond_cell[celid] & mask_phase_cell[celid]]

   
    run_data['trial_idx_mask']=np.asarray(sess_info['trial_idx_mask'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]

    run_data['t']=np.asarray(sess_info['t'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['poprate']=np.asarray(sess_info['pop_rate'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    cell_trace_sess1=np.asarray(sess_info['extract'][celid])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']] 

    run_data['trace_cells']=np.asarray([x[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']] for x in sess_info['extract']])# raw trace of all cells
    run_data['lin_pos']=np.asarray(sess_info['lin_pos'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['conditions']=np.asarray(Masks['conditions'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]


    run_data['x_loc']=np.asarray(sess_info['loc']['x'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase]
    run_data['y_loc']=np.asarray(sess_info['loc']['y'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase]
    run_data['speed']=np.asarray(sess_info['speed'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase]

 
    run_data['fr']=np.asarray(sess_info['fr'])[mask_sess_fr & mask_cond_fr &  mask_correct_fr & mask_phase_fr]

    t_all=np.asarray(sess_info['t'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase ]
    #spk_times=np.where(np.isin(run_data['t'],run_data['idpeaks']))# time of spike for population rate in the determined condtions
    
    spk_times=np.where(np.isin(t_all,run_data['idpeaks']))# time of spike for population rate in the determined condtions
    spk_times_cell= [np.where(np.isin(t_all,x)) for x in run_data['idpeaks_cells']]

    if 0:
        plt.figure()
        plt.plot(run_data['lin_pos'])
        if len (run_data['poprate'])>0:
            plt.plot(run_data['poprate']/np.max(run_data['poprate']))

        plt.eventplot(spk_times,lineoffsets=1,color='r')
        plt.eventplot(spk_times_cell[celid],lineoffsets=2,color='k')
        plt.plot(2+cell_trace_sess1/10)


    run_data['spike_idx']=spk_times
    run_data['spike_idx_cells']=spk_times_cell
    run_data['sess_name']=sess_names
    run_data['phase_name']=phase_name
    run_data['cond_name']=cond_names

    return(run_data)



def find_diffs(seq_rate_failed,pairs):
    diff_rate={}
    for pr in pairs.keys():
        for ppr in pairs[pr]:
            if 'learning' in ppr:
                learning_rate=seq_rate_failed[ppr]
            elif 'learned' in ppr:
                learned_rate=seq_rate_failed[ppr]
        
        diff_rt=learned_rate-learning_rate

        diff_rate[pr]=diff_rt
    return diff_rate


        
def smooth_signal(signal, window_size):
    """
    Smooth a signal using a moving average filter.

    Parameters:
    - signal: The input signal to be smoothed.
    - window_size: The size of the moving average window.

    Returns:
    - smoothed_signal: The smoothed signal with the same length as the input signal.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size should be an odd number.")

    # Pad the signal to handle edges
    pad_width = (window_size - 1) // 2
    padded_signal = np.pad(signal, pad_width, mode='edge')

    # Apply the moving average filter
    weights = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(padded_signal, weights, mode='valid')

    return smoothed_signal




def find_condition(input_nums, condition_dict):
    """
    Determine the condition label based on a list of numeric condition identifiers.

    This function uses a predefined mapping (for "side" and "center" conditions) to convert
    numeric identifiers into string labels (e.g., 'outward_L', 'inward_R'). It returns the
    label corresponding to the first matching identifier found in the input list.

    Parameters
    ----------
    input_nums : list or array-like
        A list of numeric identifiers representing potential conditions.
    condition_dict : dict
        A dictionary of condition names (reserved for future extensions; currently not used).

    Returns
    -------
    str or None
        The condition label corresponding to the first numeric identifier in input_nums that
        is found in the predefined mapping, or None if no match is found.
    """
    # Predefined mapping for keys that have both "side" and "center" suffixes.
    side_center_mapping = {
        3: 'outward_L',
        2: 'outward_L',
        5: 'outward_R',
        4: 'outward_R',
        8: 'inward_L',
        9: 'inward_L',
        10: 'inward_R',
        11: 'inward_R'
    }

    # Find the first numeric value in input_nums that is in the mapping.
    result = [num for num in input_nums if num in side_center_mapping]

    if not result:
        return None

    return side_center_mapping[result[0]]


def apply_masks_test(sess_info, Masks, cond_numbers, cond_name, sessin_numbers, odd_even, sess_name, trial_type, phase):
    """
    Apply masks to session data to extract data for a specific condition, session, phase, and trial type.

    This function filters the session information (in `sess_info`) using various boolean masks from
    the `Masks` dictionary. Data is selected based on multiple criteria including:
      - Trial type (correct or failed)
      - Phase (e.g., learning or learned)
      - Condition (e.g., outward, inward) specified by `cond_numbers` and `cond_name`
      - Session number (from `sessin_numbers` and `sess_name`)
      - Odd or even trial selection (if specified)
      - Speed threshold criteria (not used for the manuscript)

    The filtered data is then returned in a dictionary (`run_data`) containing information such as:
      - Cluster IDs, burst peak indices, sequences, burst events
      - Single-cell spike times and condition labels for cells
      - Trial indices, time vectors, population rates, raw cell traces, position data,
        spatial coordinates, speed, and firing rates
      - Session, phase, and condition names

    Parameters
    ----------
    sess_info : dict
        Dictionary containing session information data. Expected keys include:
          'Spike_times_cells', 'id_peaks', 'seqs', 'bursts', 'trial_idx_mask',
          't', 'pop_rate', 'extract', 'lin_pos', 'xloc', 'yloc', 'speed', 'fr', etc.
    Masks : dict
        Dictionary containing various boolean masks for filtering the session data. Expected keys include:
          'odd_even', 'odd_even_seqs', 'odd_even_fr', 'phases', 'bursts_phase', 'fr_phase', 
          'correct_failed', 'correct_failed_seqs', 'bursts_sess', 'sessions', 'fr_sess', 'cell_phase',
          'bursts_cond', 'conditions', 'fr_cond', 'cell_cond', 'cell_correct', 'correct_failed_fr',
          'speed_seq', 'speed'
    cond_numbers : list or array-like
        List of numeric condition identifiers to filter conditions (e.g., [3] for outward).
    cond_name : dict
        Dictionary mapping condition numbers to condition names.
    sessin_numbers : list or array-like
        List of session numbers to include.
    odd_even : int or None
        Specify 0 for odd or 1 for even trials; if None, all trials are included.
    sess_name : dict
        Dictionary mapping session numbers to session names.
    trial_type : int
        Trial type to filter by (1 for correct trials, 0 for failed trials).
    phase : int or None
        Phase of the session to filter by (0 for learning, 1 for learned). If None, includes both.

    Returns
    -------
    run_data : dict
        Dictionary containing filtered session data with keys including:
          'idpeaks_cells', 'mask_cond_fr_cells', 'ids_clust', 'idpeaks', 'seqs', 'bursts',
          'mask_cond_fr', 'trial_idx_mask', 't', 'poprate', 'trace_cells', 'lin_pos',
          'conditions', 'x_loc', 'y_loc', 'speed', 'fr', 'spike_idx', 'spike_idx_cells',
          'sess_name', 'phase_name', 'cond_name'.
    """
    # Initialize the output dictionary with empty lists for cell-specific data.
    run_data = {
        'idpeaks_cells': [[] for _ in range(len(sess_info['Spike_times_cells']))],
        'mask_cond_fr_cells': [[] for _ in range(len(sess_info['Spike_times_cells']))]
    }

    # ~~~~~~ Define Masks for Odd/Even Trials ~~~~~~
    if odd_even is not None:
        # Masks for odd/even trials (0: odd, 1: even) for different data streams.
        mask_odd = np.asarray(Masks['odd_even']) == odd_even
        mask_odd_seqs = np.asarray(Masks['odd_even_seqs']) == odd_even
        mask_odd_fr = np.asarray(Masks['odd_even_fr']) == odd_even
    else:
        # If odd_even is not specified, include all trials.
        mask_odd = np.ones(len(Masks['phases']), dtype=bool)
        mask_odd_seqs = np.ones(len(Masks['bursts_phase']), dtype=bool)
        mask_odd_fr = np.ones(len(Masks['fr_phase']), dtype=bool)

    # ~~~~~~ Define Masks for Phase (Learning vs. Learned) ~~~~~~
    if phase is not None:
        mask_phase = np.asarray(Masks['phases']) == phase
        mask_phase_seqs = np.asarray(Masks['bursts_phase']) == phase
        # For each cell, create a mask for the phase.
        mask_phase_cell = [np.asarray(x) == phase for x in Masks['cell_phase']]
        mask_phase_fr = np.asarray(Masks['fr_phase']) == phase

        # Define a human-readable phase name.
        phase_name = 'learning' if phase == 0 else 'learned'
    else:
        # Include both phases if phase is None.
        mask_phase = np.ones(len(Masks['phases']), dtype=bool)
        mask_phase_seqs = np.ones(len(Masks['bursts_phase']), dtype=bool)
        mask_phase_cell = [np.ones(len(x), dtype=bool) for x in Masks['cell_phase']]
        mask_phase_fr = np.ones(len(Masks['fr_phase']), dtype=bool)
        phase_name = 'learning and learned'

    # ~~~~~~ Define Masks for Correct/Failed Trials ~~~~~~
    mask_correct = np.asarray(Masks['correct_failed']) == trial_type
    mask_correct_seqs = np.asarray(Masks['correct_failed_seqs']) == trial_type

    # ~~~~~~ Define Masks for Session Number ~~~~~~
    mask_sess_burst = np.zeros_like(np.asarray(Masks['bursts_sess']), dtype=bool)
    mask_sess_t = np.zeros_like(np.asarray(Masks['sessions']), dtype=bool)
    mask_sess_fr = np.zeros_like(np.asarray(Masks['fr_sess']), dtype=bool)
    sess_names = str()
    # Loop through the desired session numbers to build session masks.
    for sess_nbr in sessin_numbers:
        mask_sess_burst |= np.asarray(Masks['bursts_sess']) == sess_nbr
        mask_sess_t |= np.asarray(Masks['sessions']) == sess_nbr
        mask_sess_fr |= np.asarray(Masks['fr_sess']) == sess_nbr
        # Extract the session name (remove file extension from key).
        sess_name_ = list(sess_name.keys())[sess_nbr]
        sess_names += ' and ' + sess_name_[:-4]

    # Build mask for each cell based on session number.
    mask_sess_cell = [np.zeros_like(np.asarray(x), dtype=bool) for x in Masks['cell_phase']]
    for celid in range(len(mask_sess_cell)):
        for sess_nbr in sessin_numbers:
            mask_sess_cell[celid] |= np.asarray(Masks['cell_sess'][celid]) == sess_nbr

    # ~~~~~~ Define Masks for Condition ~~~~~~
    mask_cond_burst_side = np.zeros_like(np.asarray(Masks['bursts_cond']), dtype=bool)
    mask_cond_t_side = np.zeros_like(np.asarray(Masks['conditions']), dtype=bool)
    mask_cond_fr = np.zeros_like(np.asarray(Masks['fr_cond']), dtype=bool)
    cond_names = str()
    # Loop through the desired condition numbers to build condition masks.
    for cond_nbr in cond_numbers:
        mask_cond_burst_side |= np.asarray(Masks['bursts_cond']) == cond_nbr
        mask_cond_t_side |= np.asarray(Masks['conditions']) == cond_nbr
        mask_cond_fr |= np.asarray(Masks['fr_cond']) == cond_nbr
        cond_name_ = list(cond_name.keys())[cond_nbr]
        cond_names += ' and ' + cond_name_

    # Build mask for each cell based on condition.
    mask_cond_cell = [np.zeros_like(x, dtype=bool) for x in Masks['cell_cond']]
    for celid in range(len(mask_cond_cell)):
        for cond_nbr in cond_numbers:
            mask_cond_cell[celid] |= np.asarray(Masks['cell_cond'][celid]) == cond_nbr

    mask_correct_fr = np.asarray(Masks['correct_failed_fr']) == trial_type

    # For burst conditions, use the side mask.
    mask_cond_burst = mask_cond_burst_side
    mask_cond_t = mask_cond_t_side

    # ~~~~~~ Apply Combined Masks to Session Data ~~~~~~
    # Use logical AND (&) to combine multiple masks for bursts and sequences.
    run_data = dict()
    run_data['ids_clust'] = np.asarray(sess_info['ids_clust'])[
        mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']
    ]
    run_data['idpeaks'] = np.asarray(sess_info['id_peaks'])[
        mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']
    ]
    run_data['seqs'] = np.asarray(sess_info['seqs'])[
        mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']
    ]
    run_data['bursts'] = np.asarray(sess_info['bursts'])[
        mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']
    ]
    run_data['mask_cond_fr'] = np.asarray(Masks['bursts_cond'])[
        mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']
    ]

    # Process cell-specific data: spike times and condition labels.
    for celid in range(len(sess_info['Spike_times_cells'])):
        msk_crct = np.asarray(Masks['cell_correct'][celid]) == bool(trial_type)
        run_data['idpeaks_cells'][celid] = np.asarray(sess_info['Spike_times_cells'][celid])[
            msk_crct & mask_sess_cell[celid] & mask_cond_cell[celid] & mask_phase_cell[celid]
        ]
        run_data['mask_cond_fr_cells'][celid] = np.asarray(Masks['cell_cond'][celid])[
            msk_crct & mask_sess_cell[celid] & mask_cond_cell[celid] & mask_phase_cell[celid]
        ]

    # Combine masks for trial indices, time, and population rate.
    run_data['trial_idx_mask'] = np.asarray(sess_info['trial_idx_mask'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    run_data['t'] = np.asarray(sess_info['t'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    run_data['poprate'] = np.asarray(sess_info['pop_rate'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    # Extract raw traces for all cells.
    run_data['trace_cells'] = np.asarray([
        x[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
        for x in sess_info['extract']
    ])
    run_data['lin_pos'] = np.asarray(sess_info['lin_pos'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    run_data['conditions'] = np.asarray(Masks['conditions'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    run_data['x_loc'] = np.asarray(sess_info['xloc'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    run_data['y_loc'] = np.asarray(sess_info['yloc'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    run_data['speed'] = np.asarray(sess_info['speed'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']
    ]
    run_data['fr'] = np.asarray(sess_info['fr'])[
        mask_sess_fr & mask_cond_fr & mask_correct_fr & mask_phase_fr
    ]

    # Get all time points for the selected trials.
    t_all = np.asarray(sess_info['t'])[
        mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase
    ]
    # Find spike times for the population.
    spk_times = np.where(np.isin(t_all, run_data['idpeaks']))
    # Find spike times for each cell.
    spk_times_cell = [np.where(np.isin(t_all, x)) for x in run_data['idpeaks_cells']]

    # (Optional) Plotting code for verification can be enabled here.
    if 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(run_data['lin_pos'])
        if len(run_data['poprate']) > 0:
            plt.plot(run_data['poprate'] / np.max(run_data['poprate']))
        plt.eventplot(spk_times, lineoffsets=1, color='r')
        plt.eventplot(spk_times_cell[celid], lineoffsets=2, color='k')
        plt.plot(2 + sess_info['extract'][celid] / 10)

    # Save spike index data and additional session/condition names.
    run_data['spike_idx'] = spk_times
    run_data['spike_idx_cells'] = spk_times_cell
    run_data['sess_name'] = sess_names
    run_data['phase_name'] = phase_name
    run_data['cond_name'] = cond_names

    return run_data





# def pc_faction_in_sequnce(Masks,sess_info,sig_pc_idx_ph,cond_names):
#     '''This function finds the precentage of the cells in a sequence that are place cells.

# ''' 
#     PC_frac_in_seq={}
#     tasks=['sampling','outward','reward','inward']

#     cnt=-1
#     for phs in range(2):# learned 
#         PC_frac_in_seq_corr={}
        
#         for correct in range(2):# correct trials


#             #fig, ax = plt.subplots(3, 1, figsize=(7, 10))

#             cnt=cnt+1
#             if phs==1:
#                 mode='learned'
#             else:
#                 mode='learning'

#             if correct==1:
#                 typoftrial='correct_trials'
#             else:
#                 typoftrial='failed_trials'

#             sig_pc_idx=sig_pc_idx_ph[mode]# the indices of the si/pc/tc cells from learning or learned

#             ph_mask=np.asarray(Masks['bursts_phase'])==phs
#             correct_mask=np.asarray(Masks['correct_failed_seqs'])==correct


#             cond_seqs={}
#             pc_ratio={}
#             seq_len={}
#             pc_ratio2={}



#             for itsk, tsk in enumerate(tasks):
#                 mskcnd=np.zeros_like(Masks['bursts_cond']).astype(bool)
#                 #mskcnd=np.zeros_like(Masks['bursts_cond'],type=bool)
#                 for icond, condname_r in enumerate(cond_names):
#                     if tsk in condname_r:
#                         #print(cond_names[condname_r])
#                         mskcnd+=(np.asarray(Masks['bursts_cond'])==cond_names[condname_r])
#                 cond_seqs[tsk]=np.asarray(sess_info['seqs'])[mskcnd & correct_mask & ph_mask] 


#             #selected_seqs = np.asarray(sess_info['seqs'])[(np.asarray(Masks['bursts_cond'])==8)|(np.asarray(Masks['bursts_cond'])==9)|(np.asarray(Masks['bursts_cond'])==10)|(np.asarray(Masks['bursts_cond'])==11)]


#                 title=tsk
#                 pc_seq_lengh=np.zeros(len(cond_seqs[tsk]))# precentage of place cells that are contibuted in a sequence

#                 pc_seq_ratio=np.zeros(len(cond_seqs[tsk]))# precentage of place cells that are contibuted in a sequence
#                 seq_pc_ratio=np.zeros(len(cond_seqs[tsk]))# precentage of sequences that are place cells 
#                 if len(cond_seqs[tsk])>0:
#                     len_seq_max=np.max([len(x) for x in cond_seqs[tsk]])
#                 for iseq,seq in enumerate(cond_seqs[tsk]):
#                     pc_seq_ratio[iseq]=(np.sum(np.isin(seq,sig_pc_idx))/len(sig_pc_idx[0]))# how many precent of the place cells conributed in this sequence
#                     seq_pc_ratio[iseq]=(np.sum(np.isin(seq,sig_pc_idx))/len(seq))# how many precents of the cells in this sequence are place cells
#                     pc_seq_lengh[iseq]=(len(seq))

#                 pc_ratio[tsk]=seq_pc_ratio
#                 seq_len[tsk]=pc_seq_lengh
#                 pc_ratio2[tsk]=pc_seq_ratio


#                 # PC_frac_in_seq[tsk]=pc_seq_lengh
#                 # PC_frac_in_seq[tsk]=pc_seq_ratio
#             PC_frac_in_seq_corr[typoftrial]= pc_ratio  
        
#         PC_frac_in_seq[mode]=PC_frac_in_seq_corr
#     return PC_frac_in_seq

def pc_faction_in_sequnce(Masks, sess_info, sig_pc_idx_ph, cond_names):
    """
    Compute the fraction of place cells present in sequences for each task, phase, and trial type.

    This function calculates two types of ratios for sequences:
      1. The fraction of all identified place cells (from sig_pc_idx_ph) that appear in a given sequence.
      2. The fraction of cells in a sequence that are identified as place cells.
      
    For each phase (learning and learned) and for each trial type (correct and failed), the function:
      - Applies phase and trial masks to select the appropriate sequences.
      - Divides sequences into tasks based on condition masks (e.g., sampling, outward, reward, inward).
      - Computes the percentage of place cells in each sequence (i.e., the ratio of place cells in the 
        sequence relative to the total number of cells in that sequence).

    Parameters
    ----------
    Masks : dict
        Dictionary containing various boolean masks. Expected keys include:
          - 'bursts_phase': mask for the phase (0 for learning, 1 for learned)
          - 'correct_failed_seqs': mask for trial type (0 for failed, 1 for correct)
          - 'bursts_cond': mask for condition numbers for each burst
    sess_info : dict
        Dictionary containing session information. It must include:
          - 'seqs': a list/array of sequences, where each sequence is an array of cell indices.
    sig_pc_idx_ph : dict
        Dictionary with keys 'learning' and 'learned'. Each entry contains an array (or list of arrays)
        of indices corresponding to cells identified as place cells in that phase.
    cond_names : dict
        Dictionary mapping condition names to numeric codes. These codes are used to filter sequences
        based on the task (e.g., {'sampling': code1, 'outward': code2, ...}).

    Returns
    -------
    PC_frac_in_seq : dict
        A nested dictionary where:
          - Outer keys are phases ('learning' or 'learned').
          - Each value is another dictionary with keys 'correct_trials' and 'failed_trials'.
          - These, in turn, are dictionaries mapping task names ('sampling', 'outward', 'reward', 'inward')
            to arrays. Each array contains, for each sequence of that task, the percentage of cells in the
            sequence that are place cells.
    """
    # Initialize output dictionary for the percentages
    PC_frac_in_seq = {}
    # Define tasks to be evaluated
    tasks = ['sampling', 'outward', 'reward', 'inward']

    cnt = -1  # Counter (optional, can be used for debugging/plotting)
    
    # Loop over the two phases (0: learning, 1: learned)
    for phs in range(2):
        # Create a temporary dictionary for the current phase
        PC_frac_in_seq_corr = {}
        
        # Loop over trial types: 0 for failed and 1 for correct
        for correct in range(2):
            cnt = cnt + 1  # Update counter (if needed for debugging)

            # Determine mode string based on phase
            mode = 'learned' if phs == 1 else 'learning'
            # Determine trial type string based on correctness
            typoftrial = 'correct_trials' if correct == 1 else 'failed_trials'

            # Get the indices of place cells for the current phase
            sig_pc_idx = sig_pc_idx_ph[mode]

            # Build phase and trial masks
            ph_mask = np.asarray(Masks['bursts_phase']) == phs
            correct_mask = np.asarray(Masks['correct_failed_seqs']) == correct

            # Dictionaries to store condition-specific sequences and computed ratios
            cond_seqs = {}
            pc_ratio = {}    # Fraction of cells in the sequence that are place cells
            seq_len = {}     # Length of each sequence (optional, for further analysis)
            pc_ratio2 = {}   # Fraction of all place cells that appear in the sequence (alternative metric)

            # Loop over each task (e.g., sampling, outward, etc.)
            for itsk, tsk in enumerate(tasks):
                # Initialize a mask for the current condition/task
                mskcnd = np.zeros_like(Masks['bursts_cond'], dtype=bool)
                # Loop over condition names to add to the mask if they match the task
                for icond, condname_r in enumerate(cond_names):
                    if tsk in condname_r:
                        # Add the mask corresponding to the current condition number
                        mskcnd |= (np.asarray(Masks['bursts_cond']) == cond_names[condname_r])
                # Select sequences that satisfy the condition, trial, and phase masks
                cond_seqs[tsk] = np.asarray(sess_info['seqs'])[mskcnd & correct_mask & ph_mask]

                # Initialize arrays to hold computed ratios for each sequence in the task
                pc_seq_lengh = np.zeros(len(cond_seqs[tsk]))   # Sequence lengths
                pc_seq_ratio = np.zeros(len(cond_seqs[tsk]))     # Ratio: (place cells / total place cells)
                seq_pc_ratio = np.zeros(len(cond_seqs[tsk]))     # Ratio: (place cells in sequence / sequence length)

                # Determine the maximum sequence length for current task (optional)
                if len(cond_seqs[tsk]) > 0:
                    len_seq_max = np.max([len(x) for x in cond_seqs[tsk]])
                
                # Process each sequence
                for iseq, seq in enumerate(cond_seqs[tsk]):
                    # Ratio of place cells (from sig_pc_idx) that are present in the sequence relative to total place cells
                    pc_seq_ratio[iseq] = (np.sum(np.isin(seq, sig_pc_idx)) / len(sig_pc_idx[0]))
                    # Ratio of cells in the sequence that are identified as place cells
                    seq_pc_ratio[iseq] = (np.sum(np.isin(seq, sig_pc_idx)) / len(seq))
                    # Store the sequence length
                    pc_seq_lengh[iseq] = len(seq)

                # Save computed ratios for the task.
                # Here we choose to store the fraction of sequence cells that are place cells.
                pc_ratio[tsk] = seq_pc_ratio
                seq_len[tsk] = pc_seq_lengh
                pc_ratio2[tsk] = pc_seq_ratio  # Alternatively, store the ratio of all place cells found

            # Store the results for the current trial type (correct or failed)
            PC_frac_in_seq_corr[typoftrial] = pc_ratio

        # Store the results for the current phase (learning or learned)
        PC_frac_in_seq[mode] = PC_frac_in_seq_corr

    return PC_frac_in_seq


import scipy.stats as stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy.ndimage import gaussian_filter1d 









def plot_kl_distributions_ss(js_divergence_ss,p_value_corr_js_,name,type='Correct'):
    # plot kl distrinutions
    plt.figure(figsize=(6, 4))
    # Define the bins for both histograms
    bins = np.linspace(0, max(max(js_divergence_ss[type]), max(js_divergence_ss[type+'_sh'])), 20)
    # Determine the maximum y limit for both plots
    max_y = max(
        max(np.histogram(js_divergence_ss[type], bins=bins)[0]),
        max(np.histogram(js_divergence_ss[type+'_sh'], bins=bins)[0])
    )+2

    plt.subplot(1, 2, 1)
    sns.histplot(js_divergence_ss[type], bins=bins, kde=False, color='blue', label='Original')
    plt.title(name[:-2])
    plt.xlabel(name[-2:])
    plt.ylabel('Frequency')
    plt.legend()
    plt.ylim(0, max_y)

    #plt.subplot(1, 2, 2)
    sns.histplot(js_divergence_ss[type+'_sh'], bins=bins, kde=False, color='orange', label='Shuffled')
    #plt.title('Learned vs Learning shuffled')
    plt.xlabel(name[-2:])
    plt.ylabel('Occurrence')
    plt.ylim
    plt.legend(fontsize=10,loc='upper right')
    plt.ylim(0, max_y)

    plt.suptitle(f'pval={p_value_corr_js_:.3f}', x=0.405, fontsize=14)
    plt.tight_layout()
    plt.show()



    
def calculate_divergences(condition1_data, condition2_data, epsilon=1e-10):
    # Convert to DataFrame for easier manipulation
    df_condition1 = pd.DataFrame(condition1_data)
    df_condition2 = pd.DataFrame(condition2_data)

    # Create a common set of clusters
    all_clusters = sorted(set(df_condition1['cluster_number']).union(set(df_condition2['cluster_number'])))

    # Align the cluster counts to the common set of clusters
    def align_counts(df, all_clusters):
        aligned_counts = []
        for cluster in all_clusters:
            if cluster in df['cluster_number'].values:
                aligned_counts.append(df.loc[df['cluster_number'] == cluster, 'clsuster_counts'].values[0])
            else:
                aligned_counts.append(0)
        return np.array(aligned_counts, dtype=float)  # Convert to float

    aligned_counts1 = align_counts(df_condition1, all_clusters)
    aligned_counts2 = align_counts(df_condition2, all_clusters)

    # Add a small value to avoid zero probabilities (smoothing)
    aligned_counts1 += epsilon
    aligned_counts2 += epsilon

    # Normalize the counts to get probability distributions
    def normalize_counts(counts):
        total = np.sum(counts)
        return counts / total

    condition1_prob = normalize_counts(aligned_counts1)
    condition2_prob = normalize_counts(aligned_counts2)

    # Calculate Jensen-Shannon Divergence
    js_divergence = jensenshannon(condition1_prob, condition2_prob, base=2)

    # Calculate Kullback-Leibler Divergence
    kl_divergence = entropy(condition1_prob, condition2_prob)
    if 0:
        # Plot histograms for visual comparison
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(all_clusters, aligned_counts1, alpha=0.7, label='Condition 1', color='blue')
        plt.title('Condition 1 Histogram')
        plt.xlabel('Cluster Number')
        plt.ylabel('Counts')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.bar(all_clusters, aligned_counts2, alpha=0.7, label='Condition 2', color='orange')
        plt.title('Condition 2 Histogram')
        plt.xlabel('Cluster Number')
        plt.ylabel('Counts')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return js_divergence, kl_divergence

def get_pval(orignial,shuffled):
    pvalue= 1- np.sum(orignial>np.mean(shuffled))/len(orignial)
    if pvalue==0:
        pvalue=1/len(orignial)
    return pvalue





def plot_seqs(ti,vec,fs):
    # thsi is for plotting
    fig=plt.figure(figsize=(6,6))
    templates={}
    tsamples=[]
    for nt in  range(len(ti['template'])):
        if nt >2:
            break
        # if len(np.where(np.array(ti['exclude'])==nt)[0])>0:
        #     continue
        clist=(np.where(ti['ids_clust']==nt)[0])
        #print(nt)
        #clist=ti['ids_clust'][nt]
        samples=np.array(vec)[clist,:,:]
        dummy,template = average_sequence(samples)
        templates[nt]=template/fs

        samples=np.array(vec)[clist,:,:]
        tsamples.append(t_from_mat(list(samples),fs))

    dy=0.2
    ctr=0
    for nt in templates.keys():
        ax=fig.add_axes([0.1,0.85-ctr*dy,0.8,0.1])
        ctr += 1
        isort=np.argsort(templates[nt])
        print(nt,np.array(templates[nt]).shape)
        #mot.plot_samples(templates,isort,tsamples[nt][0:10],ax,1,nc=int(nt))
        seq_nbr=len(tsamples[nt])
        plot_samples2(templates,isort,tsamples[nt][0:10],seq_nbr,ax,1,nc=int(nt))
        #hide_spines(ax=ax)
    #plt.show(block=0)




def plot_samples2(templates,isort,samples,seq_nbr,ax,fac=1,nc=-1):
    cmap = plt.get_cmap('tab10')
    ncells=len(templates[list(templates.keys())[0]])
    linsp = np.arange(1,ncells+1)
    custom_colors = [

        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
        '#66c2a5',  # Teal
        '#fc8d62',  # Coral
    ]
    nshift=0
    for idx, tmpl in enumerate(templates.values()):

        
        # Dynamically pick color for the current row
        if idx < len(custom_colors):
            color = custom_colors[idx]  # Use custom color
        else:
            color = custom_colors.colors[np.mod(idx, 10)]  # Fallback to colormap colors

        #print(templates, tmpl)
        if nc==-1:
            ax.plot(nshift+tmpl[isort]*fac, linsp, '.r', markersize=.3) 
        else:
            colrgb=np.array(cmap.colors[np.mod(nc,10)]).reshape(1,3)
            #ax.plot(nshift+tmpl[isort]*fac, linsp, '.',color=color,  markersize=.5) 
            ax.scatter(nshift+tmpl[isort]*fac, linsp,color=color,edgecolor='none',s=10) 

        nshift += .61
        
    for nsmpl in range(len(samples)):
        #ax.plot(nshift+samples[nsmpl][isort]*fac,linsp,'o',color= 'k', markersize=1)
        ax.scatter(nshift + samples[nsmpl][isort] * fac, linsp, s=10, c='k' ,edgecolor='none')

        #ax.text(3.1,len(isort)+len(isort)/20,'seq # = '+str(seq_nbr),fontsize=16)

        nshift +=.61
        
    ax.set_xlabel('Seq. #',fontsize=16)
    ax.set_ylabel('Cluster '+str(nc)+' \nSort seq#',fontsize=16)




def plot_sequences_new(mat, seq, vec, idpeaks, sig, fs, speed_trl, poprate, xlim):
    # Calculate threshold
    thresh = np.mean(poprate) + sig * np.std(poprate)
    
    # Setting up plotting options
    sorted = True
    Plot_entire_recording = True
    sequencenmbr = 10
    
    if Plot_entire_recording:
        dta = mat
        colors = 'k'
        linewidth = 1
        fsize = 18

    # Normalize the data for plotting
    khers = [(x - np.min(x)) / (np.max(x) - np.min(x)) for i, x in enumerate(dta)]
    khers = np.nan_to_num(khers, copy=True, nan=0.0)
    clidx = [i for i, x in enumerate(dta)]
    ticks = np.linspace(0, np.shape(khers)[1] - 1, 5)
    ticklabels = ["{:0.0f}".format(i) for i in ticks / fs]

    # Plotting the first figure
    with plt.rc_context({'font.size': fsize}):
        fig = plt.figure(figsize=(3, 8))
        ax0 = fig.add_axes([0.1, 0.1, 0.4, 0.7])
        ax = fig.add_axes([0.1, 0.8, 0.4, 0.1])

        for peak in idpeaks:
            peak_value = poprate[peak]  # Get the value of poprate at the specific peak index
            # First line with thinner width and a dark gray color for a clean look
            ax.vlines(peak, ymin=-1, ymax=peak_value + 10, linewidth=1, linestyles='dashed', color='dimgray', alpha=0.8)
            
            # Second line with thicker width, using a publication-friendly blue color
            ax.vlines(peak, ymin=0, ymax=peak_value + 1, linewidth=6, color='grey', alpha=0.3)
        # Plot population rate and peaks
        #ax.vlines(idpeaks, ymin=-1, ymax=max(poprate + 1), linewidth=1, color='k', alpha=0.8)
        #ax.vlines(idpeaks, ymin=0, ymax=max(poprate + 1), linewidth=10, color='y', alpha=0.5)
        ax.plot(poprate, linewidth=3,color='steelblue')
        ax.hlines(thresh, xmin=0, xmax=len(poprate), color='r', linestyles='dashed')
        ax.set_ylabel('ΔF/F0')
        ax.set_title('Detected bursts')
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot cell traces
        ax0.set_ylim(0, len(clidx))
        ax0.vlines(idpeaks, ymin=0, ymax=len(clidx), linewidth=1, linestyle='dashed',color='dimgray', alpha=0.8)
        ax0.vlines(idpeaks, ymin=0, ymax=len(clidx), linewidth=6, color='grey', alpha=0.3)
        i = 0
        for kk, ii in enumerate(khers):
            ax0.plot(ii + i, color='k', linewidth=linewidth)
            i += 1

        # Set up ticks and labels
        ax0.set_xticks(ticks, ticklabels)
        ax0.set_xlim(xlim)
        ax.set_xlim(xlim)
        ax0.set_xlabel('Time')
        ax0.set_ylabel('Cell #')

    # Plotting the second figure
    with plt.rc_context({'font.size': fsize}):
        fig2 = plt.figure(figsize=(3, 8))
        ax1 = fig2.add_axes([0.1, 0.1, 0.4, 0.7])
        ax2 = fig2.add_axes([0.65, 0.1, 0.4, 0.7])

        # Plot sorted data
        dta = vec[sequencenmbr][seq[sequencenmbr]]
        colors = 'dimgray'
        ttitle = 'Sorted'
        linewidth = 3
        fsize = 20
        khers = [(x - np.min(x)) / (np.max(x) - np.min(x)) for i, x in enumerate(dta) if np.nansum(x) != 0]
        clidx = [i for i, x in enumerate(dta) if np.nansum(x) != 0]
        i = 0
        for kk, ii in enumerate(khers):
            ax2.plot(ii + i, color=colors, linewidth=linewidth)
            i += 1
        ax2.set_title(ttitle)
        ax2.set_yticks(np.arange(len(seq[sequencenmbr])), seq[sequencenmbr])

        # Plot unsorted data
        dta = vec[sequencenmbr]
        colors = 'dimgray'
        ttitle = 'Unsorted'
        khers = [(x - np.min(x)) / (np.max(x) - np.min(x)) for i, x in enumerate(dta) if np.nansum(x) != 0]
        clidx = [i for i, x in enumerate(dta) if np.nansum(x) != 0]
        i = 0
        for kk, ii in enumerate(khers):
            ax1.plot(ii + i, color=colors, linewidth=linewidth)
            i += 1
        ax1.set_title(ttitle)
        ax1.set_yticks(np.arange(len(seq[sequencenmbr])), np.sort(seq[sequencenmbr]))

        ticks = np.linspace(0, np.shape(khers)[1] - 1, 3)
        ticklabels = ["{:0.2f}".format(i) for i in ticks / fs]
        ax1.set_xticks(ticks, ticklabels, fontsize=14)
        ax1.set_xlabel('time')
        ax1.set_ylabel('Cell index')
        ax2.set_xticks(ticks, ticklabels, fontsize=14)
        ax2.set_xlabel('Time')

    # Return figures and axes
    return fig, fig2, ax, ax0, ax1, ax2





################## Read_and_cluster_functions################################################################   


def process_sessions(task, dates, folders, datafolder, signal_type, use_uniqueindex=True):
    """
    Process session data to compute trial rates, identify cells active across all sessions,
    and plot the correct trial rate for each session.

    Parameters:
        task (list of str): List of task names.
        dates (list of str): List of session dates.
        folders (list of str): List of animal (folder) identifiers.
        datafolder (str): Base directory where data is stored.
        signal_type (str): String appended to filenames to indicate the signal type.
        use_uniqueindex (bool): If True, process and filter cells active in all sessions.

    Returns:
        dict: A dictionary containing:
            - 'trial_number': (currently unused) trial information.
            - 'correctrate': List of tuples (filename, correct trial rate).
            - 'failedrate': (currently reset to an empty list at the end) failed trial rate info.
            - 'true_cell_idx': List of cell indices from metadata.
            - 'mask_commonids': Boolean array marking cells active across all sessions.
            - 'ncells': Number of cells active in every session.
            - 'session_mask': Dictionary mapping session names to an index.
    """
    trial_number = {}
    correctrate = []
    failedrate = []
    true_cell_idx = []

    if use_uniqueindex:
        # Loop over each task, date, and folder to process the sessions
        for t, current_task in enumerate(task):
            print(current_task)
            for date in dates:
                for fol in folders:
                    # Choose filename based on the task type.
                    if current_task == 'sleep_learned_after':
                        # Be careful: if t==0 then task[t-1] may not exist.
                        if t == 0:
                            print("Warning: 'sleep_learned_after' encountered as the first task. Skipping...")
                            continue
                        filename = f"{date}_gcamp6f{fol}_{task[t-1]}.mat"
                    else:
                        filename = f"{date}_gcamp6f{fol}_{current_task}.mat"
                    
                    filepath = os.path.join(datafolder, fol, filename)
                    if os.path.exists(filepath):
                        # Generate an alternative filename (not used further in this snippet)
                        if current_task == 'sleep_learned_after':
                            base_filename = f"{date}_gcamp6f{fol}_{current_task}.mat"[:-4]
                            filenames = [f"{base_filename}_{signal_type}"]
                        else:
                            base_filename = filename[:-4]
                            filenames = [f"{base_filename}_{signal_type}"]

                        data = loadmat(filepath)
                        print(os.path.join(fol, filename))

                        # For specific tasks, calculate trial rates.
                        if current_task in ['task_learning', 'task_learned']:
                            num_failed = len(data['EvtT']['failed_trials'])
                            num_correct = len(data['EvtT']['correct_trials'])
                            total_trials = num_failed + num_correct
                            # Avoid division by zero
                            if total_trials > 0:
                                failed_rate = num_failed / total_trials
                                correct_rate = num_correct / total_trials
                            else:
                                failed_rate = 0
                                correct_rate = 0

                            failedrate.append((filename, failed_rate))
                            correctrate.append((filename, correct_rate))

                        # Save cell index data from metadata.
                        true_cell_idx.append(data['metadata']['CellRegCellID'])

    # Extract session names and correct rates for plotting.
    session_names = [item[0] for item in correctrate]
    correct_values = [item[1] for item in correctrate]

    # Plot the correct trial rate.
    plt.figure()
    plt.bar(session_names, correct_values)
    plt.xticks(rotation=90)
    # Note: 'fol' here is from the innermost loop; adjust if needed.
    plt.title(f"{fol} correct trial rate")
    plt.hlines(0.7, -1, len(session_names) + 1, colors='k', linestyles='dashed')
    plt.hlines(0.5, -1, len(session_names) + 1, colors='r', linestyles='dashed')
    plt.xlim([-1, len(session_names)])
    plt.tight_layout()
    plt.show()

    # Create a dictionary to map each session name to an index.
    session_mask = {name: idx for idx, name in enumerate(session_names)}

    # Identify cells that are active in all sessions.
    # Transpose the true_cell_idx list so that each row corresponds to one cell across sessions.
    true_cell_idx_array = np.array(true_cell_idx).T
    common_cell_indices = [i for i in range(true_cell_idx_array.shape[0])
                           if -1 not in true_cell_idx_array[i, :]]
    
    mask_commonids = np.zeros(true_cell_idx_array.shape[0], dtype=bool)
    mask_commonids[common_cell_indices] = True
    ncells = len(common_cell_indices)

    # Reset failedrate (if this is intended in your workflow)
    failedrate = []

    print('Number of active cells during tasks:', ncells)

    # Return collected results.
    return {
        'trial_number': trial_number,
        'correctrate': correctrate,
        'failedrate': failedrate,
        'true_cell_idx': true_cell_idx,
        'mask_commonids': mask_commonids,
        'ncells': ncells,
        'session_mask': session_mask,
        'newidx':common_cell_indices
    }





def process_animal_data(task, dates, folders, datafolder, signal_type, burst_len, tau,
                        place_cells, significant_pc, newidx, winlen, thr_burts, session_mask, skel):
    """
    Process and concatenate data from each animal/session.
    
    Parameters:
        task (list of str): Task names.
        dates (list of str): Session dates.
        folders (list of str): Animal identifiers (folder names).
        datafolder (str): Base folder where data is stored.
        signal_type (str): Suffix for the signal (e.g., 'deconv').
        burst_len (str): Burst length (used in descriptor string).
        tau (float): Time constant for deconvolution.
        place_cells (bool): Whether to analyze only significant place cells.
        significant_pc (array-like): Boolean index for significant place cells.
        newidx (array-like): Index of cells active in all sessions.
        winlen (int/float): Window length for burst/binning analysis.
        thr_burts (float): Threshold for burst detection.
        session_mask (dict): Dictionary mapping session filenames to indices.
        skel: Skeleton or other parameters required by the linearization function.
        
    Returns:
        data_all_sessions (dict): Dictionary with each session’s processed trial data.
        spks (list): List of spike data (one element per session).
    """
    # Initialize accumulators and counters
    idtr_f = [0]
    idtr_c = [0]
    trial_offset = 0
    trial_tmp = [0]
    trial_counter = 0
    trial_idx_mask_all = [0]
    trial_idx_mask = []
    spks = []
    data_all_sessions = {}
    global_i0 = 0  # keeps track of the overall time index across sessions
    total_trial_count = 0

    # Loop over each combination of task, date, and animal folder.
    for t, current_task in enumerate(task):
        for date in dates:
            for fol in folders:
                # Build filename based on the current task
                if current_task == 'sleep_learned_after':
                    # Be cautious: if this is the first task (t == 0) then task[t-1] is invalid.
                    if t == 0:
                        print("Warning: 'sleep_learned_after' encountered as first task. Skipping.")
                        continue
                    filename = f"{date}_gcamp6f{fol}_{task[t-1]}.mat"
                else:
                    filename = f"{date}_gcamp6f{fol}_{current_task}.mat"
                
                # Construct the full path and check that it exists
                file_path = os.path.join(datafolder, fol, filename)
                if not os.path.exists(file_path):
                    continue
                
                # Build an alternate filename (if needed later)
                if current_task == 'sleep_learned_after':
                    alt_filename = f"{date}_gcamp6f{fol}_{current_task}.mat"
                    alt_filename = alt_filename[:-4] + '_' + signal_type
                    filenames = [alt_filename]
                else:
                    alt_filename = filename[:-4] + '_' + signal_type
                    filenames = [alt_filename]
                
                descriptor = f"No_chunk_{burst_len}_{signal_type}"
                print(f"Processing: {fol}/{filename} ({current_task})")
                
                # Load the data from file
                data = loadmat(file_path)
                
                # --- Preprocess the session signal ---
                # Deconvolve the raw traces and store them in the data structure.
                decsig = deconv(data['STMx']['traces'], tau)
                data['STMx']['deconv'] = decsig
                
                # Retrieve metadata and compute sampling frequency.
                metadata = data['metadata']
                fs_str = metadata['recordingmethod']['sampling_frequency']
                fs = float(fs_str[0:2])
                if fs_str == 'kHz':
                    fs *= 1000

                # Select cells active in all sessions using the provided index.
                true_cell_idx3 = np.array(metadata['CellRegCellID'])
                STMx1 = data['STMx'][signal_type][true_cell_idx3[newidx]]
                
                # (This counter can be used for trial numbering)
                trial_counter = np.max(trial_idx_mask_all) + 1
                
                # Select cells based on whether we are analyzing only place cells.
                if place_cells:
                    STMx2 = STMx1[significant_pc]
                    spks.append(STMx1[significant_pc])
                    ncells = len(STMx2)
                else:
                    STMx2 = STMx1
                    spks.append(STMx1)
                    ncells = len(newidx)
                
                # --- Compute population activity and spike times ---
                poprate, id_peaks, bursts, seqs = binned_burst(STMx2, winlen, thr_burts, fs, timewins=[])
                
                # For each cell, use find_peaks to compute spike times.
                Spike_times = []
                for cell_signal in STMx2:
                    peaks, _ = find_peaks(cell_signal, height=0, width=1, distance=2)
                    Spike_times.append(peaks)
                
                # --- Process location and velocity if not a sleep task ---
                if 'sleep' not in current_task:
                    xloc = np.expand_dims(np.array(data['EvtT']['x']), axis=0)
                    yloc = np.expand_dims(np.array(data['EvtT']['y']), axis=0)
                    EvtT = np.concatenate((xloc, yloc))
                    v_all, speed_all, phi_all = velocity(EvtT[0], EvtT[1], fs)
                
                # --- Process trial data for task learning sessions ---
                if current_task in ['task_learned', 'task_learning']:
                    t_stamps = np.arange(STMx2.shape[1])
                    xloc = np.expand_dims(np.array(data['EvtT']['x']), axis=0)
                    yloc = np.expand_dims(np.array(data['EvtT']['y']), axis=0)
                    EvtT = np.concatenate((xloc, yloc))
                    v_all, speeds, phi_all = velocity(EvtT[0], EvtT[1], fs)
                    
                    # Reshape trial timestamps for correct and failed trials.
                    selected_corrects = np.array(data['EvtT']['correct_trials'])
                    correct_trials = selected_corrects.reshape(-1, 2)
                    selected_false = np.array(data['EvtT']['failed_trials'])
                    false_trials = selected_false.reshape(-1, 2)
                    
                    # Stack all trial indices.
                    trial_indices = np.vstack((correct_trials, false_trials))
                    for it, tl in enumerate(trial_indices):
                        trial_idx_mask.extend(it * np.ones(int(tl[1] - tl[0])))
                    
                    tracks = (data['EvtT']['x'], data['EvtT']['y'])
                    sess_data = {}
                    
                    # Update trial offset (if needed for concatenation).
                    trial_offset += trial_tmp[-1]
                    Left_trial_number = int(len(data['EvtT']['sampling_L']) / 2)
                    Right_trial_number = int(len(data['EvtT']['sampling_R']) / 2)
                    L_R = {'L': Left_trial_number, 'R': Right_trial_number}
                    
                    # Define the keys and a template dictionary for storing trial data.
                    keys = [
                        'sampling_L', 'sampling_R', 'outward_L',  
                        'outward_R', 'reward_L', 'reward_R', 
                        'inward_L', 'inward_R', 
                        'correct_trials', 'loc'
                    ]
                    data_dict_template = {
                        'trial_data': [],
                        'pop_rate': [],
                        'xloc': [],
                        'yloc': [],
                        'fr': [],
                        'seq_mask': [],
                        'id_peaks': [],
                        'bursts': [],
                        'speed': [],
                        'passid': [],
                        'lin_pos': [],
                        'loc': data['EvtT'],
                        'corr': data['EvtT']['correct_trials'],
                        'Spike_times_cells': [],
                        't': [],
                        'extract': [],
                        'seqs': [],
                        'odd_even_mask_seqs': [],
                        'fr_cell': [],
                        'correct_trial_idx_mask': [],
                        'failed_trial_idx_mask': [],
                        'correct_trial_idx_mask_fr': [],
                        'correct_trial_idx_mask_burst': [],
                        'failed_trial_idx_mask_fr': [],
                        'failed_trial_idx_mask_burst': [],
                        'trial_idx_mask': [],
                        'trial_idx_mask2': [],
                        'trial_numbers': [],
                        'binary_spike': [],
                        'binary_spike_cells': [],
                        'correct_failed_mask': [],
                        'correct_failed_bursts_mask': [],
                        'correct_failed_fr_mask': []
                    }
                    
                    # Create a dictionary for each condition key.
                    main_dict = {key: copy.deepcopy(data_dict_template) for key in keys}
                    
                    # Iterate over left and right trial conditions.
                    for side in L_R:
                        for trial_index in range(L_R[side]):
                            # Process each condition within the event structure.
                            for condition in data['EvtT']:
                                # Check that the condition name matches the expected format.
                                if (side in condition) and (
                                    ('x' not in condition) and 
                                    ('y' not in condition) and 
                                    ('correct_trials' not in condition) and 
                                    ('failed_trials' not in condition) and 
                                    ('trial_list' not in condition) and 
                                    ('sampling' not in condition)
                                    or ('sampling_L' in condition) or ('sampling_R' in condition)
                                ):
                                    # Initialize a dictionary for this condition.
                                    sess_data[condition] = {
                                        'trial_data': [],
                                        'pop_rate': [],
                                        'xloc': [],
                                        'yloc': [],
                                        'fr': [],
                                        'seq_mask': [],
                                        'id_peaks': [],
                                        'bursts': [],
                                        'speed': [],
                                        'passid': [],
                                        'lin_pos': [],
                                        'loc': data['EvtT'],
                                        'corr': data['EvtT']['correct_trials'],
                                        'Spike_times_cells': [],
                                        't': [],
                                        'extract': [],
                                        'seqs': [],
                                        'odd_even_mask_seqs': [],
                                        'fr_cell': [],
                                        'correct_trial_idx_mask': [],
                                        'failed_trial_idx_mask': [],
                                        'correct_trial_idx_mask_fr': [],
                                        'correct_trial_idx_mask_burst': [],
                                        'failed_trial_idx_mask_fr': [],
                                        'failed_trial_idx_mask_burst': [],
                                        'trial_idx_mask': [],
                                        'trial_idx_mask2': [],
                                        'trial_numbers': []
                                    }
                                    
                                    cond_name = f"{current_task}_{condition}_{date}"
                                    
                                    # Reshape the condition’s trial timestamps.
                                    selected_trials = np.array(data['EvtT'][condition])
                                    cond_trials = selected_trials.reshape(-1, 2)
                                    
                                    # Find correct and failed trial indices for this condition.
                                    correxindex, idtr_c = find_correct_index(cond_trials, correct_trials)
                                    falseindex, idtr_f = find_correct_index(cond_trials, false_trials)
                                    
                                    Cortr = [(trial, 'correct') for trial in cond_trials[correxindex]]
                                    Failtr = [(trial, 'failed') for trial in cond_trials[falseindex]]
                                    if (len(Failtr) > 0) and (len(Cortr) > 0):
                                        Alltr = np.vstack((Cortr, Failtr))
                                    elif (len(Failtr) == 0) and (len(Cortr) > 0):
                                        Alltr = Cortr
                                    elif (len(Cortr) == 0) and (len(Failtr) > 0):
                                        Alltr = Failtr
                                    else:
                                        continue  # No valid trials in this condition.
                                    
                                    trial_info = Alltr[trial_index]
                                    x_trial = trial_info[0]   # [start, end] indices for this trial
                                    trial_type = trial_info[1]
                                    if x_trial[1] > x_trial[0]:
                                        # (Optional) Find a matching index in the overall trial indices.
                                        matching_index = None
                                        for idx, trial_range in enumerate(trial_indices):
                                            if (x_trial[0] >= trial_range[0]) and (x_trial[1] <= trial_range[1]):
                                                matching_index = idx + 1
                                                break
                                        
                                        # Create a mask for peaks within the trial window.
                                        mask = (id_peaks >= x_trial[0]) & (id_peaks < x_trial[1])
                                        cond_key = condition
                                        if 'outward' in condition:
                                            cond_key = f"outward_{side}"
                                        elif 'inward' in condition:
                                            cond_key = f"inward_{side}"
                                        
                                        # Create a binary spike vector for the population.
                                        binary_spike = np.zeros(len(poprate[x_trial[0]:x_trial[1]]))
                                        binary_spike[np.asarray(id_peaks)[mask] - x_trial[0]] = 1
                                        main_dict[cond_key]['binary_spike'].extend(binary_spike)
                                        
                                        # Append the trial data.
                                        main_dict[cond_key]['trial_data'].append(
                                            np.array(STMx2[:, x_trial[0]:x_trial[1]])
                                        )
                                        main_dict[cond_key]['pop_rate'].extend(poprate[x_trial[0]:x_trial[1]])
                                        main_dict[cond_key]['xloc'].extend(data['EvtT']['x'][x_trial[0]:x_trial[1]])
                                        main_dict[cond_key]['yloc'].extend(data['EvtT']['y'][x_trial[0]:x_trial[1]])
                                        main_dict[cond_key]['fr'].append(np.sum(mask) / (x_trial[1] - x_trial[0]))
                                        main_dict[cond_key]['seq_mask'].extend(
                                            session_mask[filename] * np.ones(len(np.asarray(seqs)[mask]))
                                        )
                                        main_dict[cond_key]['id_peaks'].extend(
                                            np.asarray(id_peaks)[mask] - x_trial[0] + global_i0
                                        )
                                        main_dict[cond_key]['bursts'].extend(np.asarray(bursts)[mask])
                                        main_dict[cond_key]['speed'].extend(speeds[x_trial[0]:x_trial[1]])
                                        main_dict[cond_key]['passid'].append([x_trial[0], x_trial[1]])
                                        main_dict[cond_key]['t'].extend(t_stamps[x_trial[0]:x_trial[1]] - x_trial[0])
                                        main_dict[cond_key]['extract'].extend(
                                            np.transpose(np.array(STMx2[:, x_trial[0]:x_trial[1]]))
                                        )
                                        main_dict[cond_key]['seqs'].extend(np.asarray(seqs)[mask])
                                        main_dict[cond_key]['trial_numbers'].extend(
                                            trial_index * np.ones(len(poprate[x_trial[0]:x_trial[1]])) + total_trial_count
                                        )
                                        
                                        # Compute binary spike vectors for individual cells.
                                        cell_spike_binary = np.zeros((len(Spike_times), len(poprate[x_trial[0]:x_trial[1]])))
                                        for cell_idx, spikes_arr in enumerate(Spike_times):
                                            mask_n = (np.asarray(spikes_arr) >= x_trial[0]) & (np.asarray(spikes_arr) < x_trial[1])
                                            cell_spike_binary[cell_idx, np.asarray(spikes_arr)[mask_n] - x_trial[0]] = 1
                                        main_dict[cond_key]['binary_spike_cells'].extend(
                                            np.transpose(cell_spike_binary)
                                        )
                                        
                                        # Set masks for correct vs. failed trials.
                                        if trial_type == 'correct':
                                            main_dict[cond_key]['correct_failed_mask'].extend(
                                                np.ones(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                            main_dict[cond_key]['correct_failed_bursts_mask'].extend(
                                                np.ones(len(np.asarray(seqs)[mask])).astype(int)
                                            )
                                            main_dict[cond_key]['correct_failed_fr_mask'].extend(
                                                np.ones(1).astype(int)
                                            )
                                            main_dict[cond_key]['correct_trial_idx_mask'].extend(
                                                1 * np.ones(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                            main_dict[cond_key]['correct_trial_idx_mask_fr'].extend(
                                                1 * np.ones(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                            main_dict[cond_key]['correct_trial_idx_mask_burst'].extend(
                                                1 * np.ones(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                        elif trial_type == 'failed':
                                            main_dict[cond_key]['correct_failed_mask'].extend(
                                                np.zeros(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                            main_dict[cond_key]['correct_failed_bursts_mask'].extend(
                                                np.zeros(len(np.asarray(seqs)[mask])).astype(int)
                                            )
                                            main_dict[cond_key]['correct_failed_fr_mask'].extend(
                                                np.zeros(1).astype(int)
                                            )
                                            main_dict[cond_key]['failed_trial_idx_mask'].extend(
                                                0 * np.ones(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                            main_dict[cond_key]['failed_trial_idx_mask_fr'].extend(
                                                0 * np.ones(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                            main_dict[cond_key]['failed_trial_idx_mask_burst'].extend(
                                                0 * np.ones(len(poprate[x_trial[0]:x_trial[1]]))
                                            )
                                        
                                        # Process spike times for each cell relative to the trial.
                                        trial_spikes_allcells = []
                                        for spikes_arr in Spike_times:
                                            trial_spikes = np.asarray(spikes_arr)[
                                                (np.asarray(spikes_arr) >= x_trial[0]) &
                                                (np.asarray(spikes_arr) < x_trial[1])
                                            ]
                                            trial_spikes = trial_spikes - x_trial[0] + global_i0 if trial_spikes.size > 0 else trial_spikes
                                            trial_spikes_allcells.append(trial_spikes)
                                        main_dict[cond_key]['Spike_times_cells'].append(trial_spikes_allcells)
                                        
                                        # Compute linearized position (using your custom function).
                                        ind1, ind2 = x_trial[0], x_trial[1]
                                        if condition.split('_')[1] == 'L':
                                            lin_pos = linearize_2d_track_single_run(tracks, ind1, ind2, skel, is_left=True)
                                        elif condition.split('_')[1] == 'R':
                                            lin_pos = linearize_2d_track_single_run(tracks, ind1, ind2, skel, is_left=False)
                                        else:
                                            lin_pos = []
                                        main_dict[cond_key]['lin_pos'].extend(lin_pos)
                                        
                                        # Update the global time index offset.
                                        global_i0 += len(poprate[x_trial[0]:x_trial[1]])
                            total_trial_count += trial_index
                        
                    # Save the processed trial data for this session.
                    data_all_sessions[filename] = main_dict
                    data_all_sessions[filename].update({'correct_trials': data['EvtT']['correct_trials']})
                    loc = (data['EvtT']['x'], data['EvtT']['y'])
                    data_all_sessions[filename].update({'loc': loc})
    
    return data_all_sessions, spks








def generate_masks(data_all_sessions, session_mask,  ncells):
    """
    Generate masks and aggregate session/condition information for further analysis.

    Parameters:
        data_all_sessions (dict): Dictionary containing processed session data.
                                  Each key is a session name and its value is a dict
                                  with condition keys (e.g. 'sampling_L', 'reward_R', etc.)
                                  and associated trial/feature data.
        session_mask (dict): Dictionary mapping session names to unique session indices.

        ncells (int): The number of cells (used to initialize cell‐wise mask lists).

    Returns:
        Masks (dict): Dictionary containing various mask arrays for conditions, sessions, phases,
                      bursts, and trial indices.
        sess_info (dict): Dictionary aggregating session‐level information.
        cond_info (dict): Dictionary aggregating condition‐level information.
    """
    # ----------------------------
    # Define condition names and codes
    # ----------------------------
    conds = [
        'sampling_L', 'sampling_R',
        'outward_L', 'outward_R',
        'reward_L', 'reward_R',
        'inward_L', 'inward_R'
    ]
    cond_number = {cond: idx for idx, cond in enumerate(conds)}
    # (cond_names is identical to cond_number if needed)
    cond_names = cond_number

    # ----------------------------
    # Initialize aggregated dictionaries and lists
    # ----------------------------
    # (all_data is not used later, but you might want to keep it for reference.)
    all_data = {
        'trial_data': [], 'pop_rate': [], 'xloc': [], 'yloc': [],
        'fr': [], 'seq_mask': [], 'id_peaks': [], 'bursts': [],
        'speed': [], 'passid': [], 'lin_pos': [],
        
        'Spike_times_cells': [], 't': [], 'extract': [], 'seqs': []
    }
    
    # For condition-wise aggregated info.
    cond_info = {}
    
    # Initialize cell‐wise mask lists (one list per cell)
    cell_mask_cond    = [[] for _ in range(ncells)]
    cell_mask_sess    = [[] for _ in range(ncells)]
    cell_mask_phase   = [[] for _ in range(ncells)]
    cell_mask_correct = [[] for _ in range(ncells)]
    
    # Initialize the Masks dictionary with many fields.
    Masks = {
        'conditions': [],
        'sessions': [],
        'phases': [],
        'bursts_cond': [],
        'bursts_sess': [],
        'bursts_phase': [],
        'odd_even': [],
        'odd_even_seqs': [],
        'cell_cond': cell_mask_cond,
        'cell_sess': cell_mask_sess,
        'cell_phase': cell_mask_phase,
        'cell_correct': cell_mask_correct,
        'correct_failed': [],
        'correct_failed_seqs': [],
        'correct_failed_fr': [],
        'fr_phase': [],
        'fr_cond': [],
        'fr_sess': [],
        'odd_even_fr': [],
        'correct_trial_idx_mask': [0],
        'failed_trial_idx_mask': [0],
        'trial_idx_mask': [],
        'correct_trial_idx_mask_fr': [],
        'correct_trial_idx_mask_burst': [],
        'failed_trial_idx_mask_fr': [],
        'failed_trial_idx_mask_burst': [],
        'trial_number': [],
        'Spike_binary_cells': [],
        'Spike_binary': []
    }
    
    # Features to pool (from each condition) into the session-level info.
    features = ['pop_rate', 'xloc', 'yloc', 'fr', 'seq_mask', 'speed', 'lin_pos', 'bursts', 'seqs']
    
    # Initialize the aggregated session info dictionary.
    sess_info = {
        'trial_data': [], 'pop_rate': [], 'xloc': [], 'yloc': [],
        'fr': [], 'seq_mask': [], 'id_peaks': [], 'bursts': [],
        'speed': [], 'passid': [], 'lin_pos': [],
        
        'Spike_times_cells': [[] for _ in range(ncells)],
        't': [], 'extract': [], 'seqs': [],
        'trial_idx_mask': []
    }
    
    # These lists will accumulate binary spike data across sessions.
    binary_spike_all = []
    binary_spike_all_cells = []
    
    # Initialize a cumulative time offset variable.
    time_offset = 0

    # ----------------------------
    # Loop over each session and condition to aggregate data and create masks
    # ----------------------------
    for sess_name, sess_data in data_all_sessions.items():
        # You can also create session-specific arrays if needed:
        odd_even_mask_sess = []  # (if you use odd-even masks later)
        sess_trace = []          # (for tracing session signals)

        for condname in conds:
            # Initialize a fresh dictionary for this condition.
            cond_info[condname] = {
                'trial_data': [], 'pop_rate': [], 'xloc': [], 'yloc': [],
                'fr': [], 'seq_mask': [], 'id_peaks': [], 'bursts': [],
                'speed': [], 'passid': [], 'lin_pos': [],
                
                'Spike_times_cells': [], 't': [], 'extract': [], 'trial_number': []
            }
            # (The original code had an if‐statement checking that 'correct_trials'
            # or 'loc' are not in the condition name; since our conds list does not include
            # these, we process every condition.)
            if len(sess_data[condname]['t']) > 0:
                # t0: the first timestamp in the current condition’s data.
                t0 = sess_data[condname]['t'][0]

                # Extend condition info with id_peaks and population rate.
                cond_info[condname]['id_peaks'].extend(sess_data[condname]['id_peaks'])
                cond_info[condname]['pop_rate'].extend(
                    np.asarray(sess_data[condname]['pop_rate'])
                )

                # For each feature, add the pooled data from this condition.
                for fname in features:
                    sess_info[fname].extend(sess_data[condname][fname])
                
                # Accumulate binary spike data.
                binary_spike_all.extend(sess_data[condname]['binary_spike'])
                binary_spike_all_cells.extend(sess_data[condname]['binary_spike_cells'])
                
                # Aggregate the 'extract' data. We assume that the extract data is stored
                # as a matrix that needs to be transposed before concatenation.
                extract_data = np.array(sess_data[condname]['extract']).T
                if sess_info['extract'] == []:
                    sess_info['extract'] = extract_data
                else:
                    sess_info['extract'] = np.hstack((sess_info['extract'], extract_data))
                
                # Determine the phase indicator from the session name.
                # (For example, set phi = 0 for 'learning' and phi = 1 for 'learned'.)
                if 'learning' in sess_name:
                    phi = 0
                elif 'learned' in sess_name:
                    phi = 1
                else:
                    phi = np.nan  # or any default value
                
                # For each cell, use the binary spike data to extend cell-specific masks.
                binary_cells = np.array(sess_data[condname]['binary_spike_cells'])
                # Transpose so that we iterate cell by cell.
                for i, cell_binary in enumerate(binary_cells.T):
                    n_spikes = int(np.sum(cell_binary == 1))
                    Masks['cell_cond'][i].extend(
                        cond_number[condname] * np.ones(n_spikes)
                    )
                    Masks['cell_sess'][i].extend(
                        session_mask[sess_name] * np.ones(n_spikes)
                    )
                    Masks['cell_phase'][i].extend(
                        phi * np.ones(n_spikes)
                    )
                    # Extend the cell's "correct" mask using the condition's correct_failed_mask.
                    # (Ensure that cell_binary is interpreted as boolean.)
                    correct_mask = np.asarray(
                        sess_data[condname]['correct_failed_mask']
                    )[cell_binary.astype(bool)].astype(bool)
                    Masks['cell_correct'][i].extend(correct_mask)
                
                # Update the cumulative time offset using the current condition's t vector.
                time_offset += -sess_data[condname]['t'][0] + sess_data[condname]['t'][-1]
                
                # Extend Masks with various trial and session-level data.
                Masks['correct_failed'].extend(sess_data[condname]['correct_failed_mask'])
                Masks['correct_failed_seqs'].extend(sess_data[condname]['correct_failed_bursts_mask'])
                Masks['trial_idx_mask'].extend(np.asarray(sess_data[condname]['trial_numbers']))
                # Adjust the trial index masks by the last value so far.
                Masks['correct_trial_idx_mask'].extend(
                    np.asarray(sess_data[condname]['correct_trial_idx_mask']) + Masks['correct_trial_idx_mask'][-1]
                )
                Masks['failed_trial_idx_mask'].extend(
                    np.asarray(sess_data[condname]['failed_trial_idx_mask']) + Masks['failed_trial_idx_mask'][-1]
                )
                Masks['correct_failed_fr'].extend(sess_data[condname]['correct_failed_fr_mask'])
                Masks['fr_phase'].extend(phi * np.ones(len(sess_data[condname]['fr'])))
                Masks['fr_cond'].extend(
                    cond_number[condname] * np.ones(len(sess_data[condname]['fr']))
                )
                Masks['fr_sess'].extend(
                    session_mask[sess_name] * np.ones(len(sess_data[condname]['fr']))
                )
                Masks['correct_trial_idx_mask_fr'].extend(sess_data[condname]['correct_trial_idx_mask_fr'])
                Masks['correct_trial_idx_mask_burst'].extend(sess_data[condname]['correct_trial_idx_mask_burst'])
                Masks['failed_trial_idx_mask_fr'].extend(sess_data[condname]['failed_trial_idx_mask_fr'])
                Masks['failed_trial_idx_mask_burst'].extend(sess_data[condname]['failed_trial_idx_mask_burst'])
                Masks['phases'].extend(phi * np.ones(len(sess_data[condname]['pop_rate'])))
                
                # Extend burst-related masks.
                Masks['bursts_cond'].extend(
                    cond_number[condname] * np.ones(len(sess_data[condname]['bursts']))
                )
                Masks['bursts_sess'].extend(
                    session_mask[sess_name] * np.ones(len(sess_data[condname]['bursts']))
                )
                Masks['bursts_phase'].extend(
                    phi * np.ones(len(sess_data[condname]['bursts']))
                )
                
                # Extend masks for session and condition information.
                Masks['sessions'].extend(
                    session_mask[sess_name] * np.ones(len(sess_data[condname]['pop_rate']))
                )
                Masks['conditions'].extend(
                    cond_number[condname] * np.ones(len(sess_data[condname]['pop_rate']))
                )
                Masks['trial_number'].extend(
                    np.asarray(sess_data[condname]['trial_numbers'])
                )
    
    # ----------------------------
    # Finalize session-level info using the accumulated binary spike data.
    # ----------------------------
    sess_info['id_peaks'] = np.where(np.asarray(binary_spike_all))[0]
    sess_info['trial_idx_mask'] = Masks['trial_idx_mask']
    sess_info['Spike_times_cells'] = [
        np.where(arr)[0] for arr in np.array(binary_spike_all_cells).T
    ]
    sess_info['Spike_binary_cells'] = np.array(binary_spike_all_cells).T
    sess_info['Spike_binary'] = binary_spike_all
    sess_info['t'] = np.arange(len(binary_spike_all_cells))
    
    return Masks, sess_info, cond_info