import numpy as np
from scipy import spatial

### Helper functions.
def separate_in(indices,not_in):
    ''' Separate 2 periods of spike-like (point) data.
    '''    
   
    not_in_range=[]
    for n in range(len(not_in[0::2])):
        not_in_range.extend(range(not_in[0::2][n],not_in[1::2][n],1))# I added the +1 (hamed)
    not_in_range=set(not_in_range)
  
    ind=[]
    for n in range(len(indices)):
        if indices[n] in not_in_range:
            ind.append(indices[n])
   
    return ind

def separate_in_2d_array(data,manual_in):

    start=manual_in[0::2]
    end=manual_in[1::2]
    
    extracted=[]
    for i in range(len(data)):
        l1=len(start)
        l2=len(end)
        
        for n in range(np.min([l1,l2])):
            extracted.extend(data[i][int(start[n]):int(end[n])])
    extracted=np.reshape(extracted,[len(data),-1])
    return extracted


def linearize_2d_track_single_run(track,start,end,skel,is_left=True):
    '''

    Parameters
    ----------
    track : 2d array (x and y over time)
        Behavioural tracking.
    start , end : list of int
        start-end time points of behavioural epochs.
    skel : dict
        Skeleton of this mosue.
    is_left : Boolean, optional
        True: left, False: right. The default is True.

    Returns
    -------
    lin_pos : 1d array
        Linearized position of the run.

    '''
    if is_left==True:
        c=skel['skeleton left'] 
        total_length=skel['length left']
        x_real=track[0][start:end]
        y_real=track[1][start:end]
        lin_pos=[]
        for n in range(len(x_real)):
            first_ind=[x_real[n],y_real[n]]
            distance,index = spatial.KDTree(c).query(first_ind)
            lin_pos.append(index/total_length)
        
    else:
        c=skel['skeleton right'] 
        total_length=skel['length right']
        x_real=track[0][start:end]
        y_real=track[1][start:end]
        lin_pos=[]
        for n in range(len(x_real)):
            first_ind=[x_real[n],y_real[n]]
            distance,index = spatial.KDTree(c).query(first_ind)
            lin_pos.append(index/total_length)
    
        
    
    return lin_pos