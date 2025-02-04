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

Tspare=.5




def extract_seq(times):
    stimes=np.sort(times)
    ids=np.argsort(times)
    return ids[np.where(~np.isnan(stimes))[0]], ids
    
def rankseq(s1,s2):

    #compute rank order correlation between sequences

    # set things straight
    s1=np.array(s1).flatten()
    s2=np.array(s2).flatten()
    l1=len(s1)
    l2=len(s2)
    
    #difference matrix
    d=np.ones((l1,1))*s2 - (np.ones((l2,1))*s1).transpose()
  
    # binary identity matrix
    d=(d==0)
  

    # make s0 the shorter sequence
    s=s1
    s0=s2
    l0=l2
    ln=l1
    if l1<l2:
        s=s2;
        s0=s1;
        l0=l1
        ln=l2
        d=d.transpose()


 
        
    #compute cell overlap (neurons contained in both)
    minseq=s[np.where(np.sum(d,axis=1)>0)[0]];
    lm=len(minseq)
  
    # delete neurons from the shorter sequence that are not in the minimal
    # sequence
    #
    
    d0=np.ones((l0,1))*minseq - (np.ones((lm,1))*s0).transpose()
    d0=(d0==0)
    s0=s0[np.sum(d0,axis=1)>0]
    l0=len(s0)
  
  
    #find ordinal rank in the shorter sequence
    dd=np.ones((lm,1))*s0 - (np.ones((l0,1))*minseq).transpose()
  
    #compute spearmans r
    if len(dd)>1:
        ids=np.argmin(np.abs(dd),axis=0)
        
        rc = np.corrcoef(np.arange(len(ids)),ids)[0,1]
        ln=len(ids)
    else:
        rc=np.nan;
        ln=np.nan
  
   
    
    return rc, ln



def shuffle(narr):

    nrep=10000
    
    ret=[]
    for n in narr:

        s2=np.arange(n)

        rval=np.zeros(nrep)
        for m in range(nrep):

            s1=np.random.permutation(n)

            rval[m],dummy=rankseq(s1,s2)

        c=np.mean(rval)
        sd=np.std(rval)
        prctl=np.quantile(rval,.95)
        
        ret.append([n, c, sd, prctl])

    return ret

    
def allmot(seqs,nrm):

    nseqs=len(seqs)

    narr=np.array(nrm)[:,0]

    corrmat=np.zeros((nseqs,nseqs))
    zmat=np.zeros((nseqs,nseqs))
    bmat=np.zeros((nseqs,nseqs))
    pval=np.zeros(nseqs)
    nsig=np.zeros(nseqs)
    
    for ns in range(nseqs):

        s1=seqs[ns]

        zmat[ns,ns]=np.nan
        bmat[ns,ns]=np.nan
        for ms in range(ns+1,nseqs):

            

            s2=seqs[ms]

            rc,ln=rankseq(s1,s2)
            

            if ln>=50:
                mns=nrm[-1]
            else:
                whichone=np.array(np.where(ln==narr)).flatten()
                if len(whichone)==0:
                    mns=np.empty(4)
                    mns[:]=np.nan
                else:
                    mns=nrm[whichone[0]]
                    
                    
            ztmp=(rc-mns[1])/mns[2]
            #print(ns,mns,ztmp)
            corrmat[ns,ms]=rc
            corrmat[ms,ns]=rc

            zmat[ns,ms]=ztmp
            zmat[ms,ns]=ztmp
            bmat[ns,ms]=1.*(ztmp>mns[3])
            bmat[ms,ns]=1.*(ztmp>mns[3])

        nsig[ns] = np.nansum(bmat[ns,:])
        pval[ns] = 1-binom.cdf(nsig[ns],nseqs-1,.05)# i will change pvalue from .05 to 0.01 for merging(hamed 02.02.2023)


    rep_index = nsig/np.std(nsig)


    return rep_index, nsig, pval, bmat, zmat, corrmat


def check_template(seqs,tmpl,nrm):

    nseqs=len(seqs)
    s1=np.array(tmpl).flatten()

    narr=np.array(nrm)[:,0]    
    sig=np.zeros(nseqs)
    zval=np.zeros(nseqs)
        
    for ns in range(nseqs):
    
        s2=seqs[ns]
        rc,ln=rankseq(s1,s2)
        #print(rc,ln)
        
        if ln>=50:
            mns=nrm[-1]
        else:
            whichone=np.array(np.where(ln==narr)).flatten()
            if len(whichone)==0:
                mns=np.empty(4)
                mns[:]=np.nan
            else:
                mns=nrm[whichone[0]]


        ztmp=(rc-mns[1])/mns[2]
        sig[ns]=1.*(ztmp>mns[3])
        zval[ns]=ztmp


    return zval,sig


def spiketimes_to_mat(st,fs):
    ncells=len(st)
    Tmax=0
    for n in range(ncells):
        #print(n, np.max(st[n]))
        if len(st[n])>0:
            Tmax=np.max([Tmax, np.max(st[n])])

    nbins=int(np.ceil(Tmax*fs))
    mat=np.zeros((ncells,nbins))
    #
    for n in range(ncells):
        if len(st[n])>0:
            ids=np.floor(np.array(st[n])*fs).astype(int)
            mat[n,ids]=1
    
    return mat









def popbursts(mat,sig,fs,kwidth=0,minwidth=1):
    global Tspare
    random_time=False
    import random
    #poprate=np.sum(mat,axis=0)
    if kwidth<1/(3*fs):
        kwidth=1/(3*fs)
        poprate=np.sum(mat,axis=0)
    else:
        tax=np.arange(-3*kwidth,3*kwidth,1/fs)
        poprate=np.convolve(np.sum(mat,axis=0),np.exp(-(tax**2)/2/kwidth**2)/kwidth,'same')
    
    thresh=np.mean(poprate)+sig*np.std(poprate)
    spare=int(Tspare*fs)
    print('Burst length is ' ,Tspare)
    
    #mask=(poprate>=thresh)*1.
    #mask=1.*(np.diff(mask)==1)
    #ids=np.where(mask>0)[0]
    
    vec=[]
    
    idpeaks, _ = find_peaks(poprate, height=thresh, width=(minwidth,spare*10), distance=spare)
    
    peaks=[]
    idprev=-1
    #for n in range(len(ids)):
    #
    #    id0=ids[n]
    #
    #    if id0+spare<=mat.shape[1]:
    #        idpeak=np.argmax(poprate[id0:id0+spare])+id0
    #        if (idpeak-idprev>spare)*(idpeak-int(spare/2)>=0)*(idpeak+int(spare/2)+1<=mat.shape[1]):
    #            vtmp=mat[:,idpeak-int(spare/2):idpeak+int(spare/2)+1]
    #            if len(np.where(np.sum(vtmp,axis=1)>0)[0])>4:#mimimum 5 active cells
    #                vec.append(vtmp)
    #                peaks.append(idpeak/fs)
    #                idprev=idpeak
    
    for idpeak in idpeaks:
        if (idpeak-idprev>spare)*(idpeak-int(spare/2)>=0)*(idpeak+int(spare/2)+1<=mat.shape[1]):
            vtmp = mat[:,idpeak-int(spare/2):idpeak+int(spare/2)+1]

            if random_time:
                column_indices = np.arange(np.shape(vtmp)[1])
                np.random.shuffle(column_indices)
                print('bursts are randomized!!!')
                # Use the shuffled index array to rearrange the columns
                vtmp = vtmp[:, column_indices]




            if len(np.where(np.sum(vtmp,axis=1)>0)[0])>4:
                vec.append(vtmp)
                #peaks.append(idpeak/fs)

                peaks.append(idpeak)# Hamed Chaned

                idprev=idpeak
                
                
    if len(vec)>0:
        
        # if random_time==True:
        #     itax=np.arange(vec[0].shape[1])
        #     random.shuffle(itax)
        #     print('bursts are randomized!!!')
        # else:

        itax=np.arange(vec[0].shape[1])



    seq=[]
    for nv in range(len(vec)):
        nvec=np.sum(vec[nv],axis=1)
        cofseq=(itax@vec[nv].transpose())/nvec
        tmp=np.argsort(cofseq)
        seq.append(tmp[~np.isnan(np.sort(cofseq))])
                   
      
    
    
    
    return vec,seq,peaks,poprate








def average_sequence(burst_set):
    
    vec=np.mean(burst_set,axis=0)
    itax=np.arange(vec.shape[1])
    nvec=np.sum(vec,axis=1)
    cofseq=(itax@vec.transpose())/nvec
    tmp=np.argsort(cofseq)
    seq=tmp[~np.isnan(np.sort(cofseq))]

    return seq,cofseq


def within_across(adj,ids_clust):

    ret={'within':[], 'across':[]}
    for nc in range(len(adj)):
        idin = np.where((ids_clust==nc))[0]
        idout = np.where(~(ids_clust==nc))[0]
        within = np.mean(adj[nc][idin])
        across = np.mean(adj[nc][idout])
        ret['within'].append(within)
        ret['across'].append(across)

    return ret

def templates(bursts,seqs,nrm,ids_clust,min_ratio=2):

    retval={'adj':[], 'template':[], 'clist':[], 'radius':[],'seqs':[],'ids_clust':[], 'bursts':[], 'ratio':[]}
    retval['seqs'].append(seqs)# added by Hamed
    retval['ids_clust'].append(ids_clust)# added by Hamed
    retval['bursts'].append(bursts)# added by Hamed

    for nc in range(max(ids_clust)+1):
        clist=(np.where(ids_clust==nc)[0])
        if np.array(bursts).ndim==2:
            mns = np.nanmean(np.array(bursts)[clist,:], axis=0)
            tmp = np.argsort(mns)
            temp = tmp[~np.isnan(np.sort(mns))]

        elif np.array(bursts).ndim==3:
            temp,dummy = average_sequence(np.array(bursts)[clist,:,:])
            
        chck=check_template(seqs,temp,nrm)
        radius=np.mean(chck[1])*30
        retval['template'].append(temp)
        retval['clist'].append(clist)
        retval['radius'].append(radius)
        retval['adj'].append(chck[1])


    crit = within_across(retval['adj'],ids_clust)
    retval.update({'exclude':[]})
    for nc in range(len(retval['radius'])):
        ratio=crit['within'][nc]/crit['across'][nc]
        best_ratio=crit['within'][nc]
        #retval['within_ratio'].append(best_ratio)
        retval['ratio'].append(ratio)

        #print(nc, ": ", ratio)
        if ratio<min_ratio:
            retval['exclude'].append(nc)
                               
                               
    retval['template'] = [i for j, i in enumerate(retval['template']) if j not in retval['exclude']]# remove bad clusters
    retval['clist'] = [i for j, i in enumerate(retval['clist']) if j not in retval['exclude']]# remove bad clusters
    retval['radius'] = [i for j, i in enumerate(retval['radius']) if j not in retval['exclude']]# remove bad clusters
    retval['adj'] = [i for j, i in enumerate(retval['adj']) if j not in retval['exclude']]# remove bad clusters
    #retval['ratio'] = [i for j, i in enumerate(retval['ratio']) if j not in retval['exclude']]# remove bad clusters
    #retval['ids_clust'] = [i for j, i in enumerate(retval['ids_clust']) if j not in retval['exclude']]# remove bad clusters


    #
    return retval





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





def cluster(bmat,zmat,params):
    cmat=np.zeros_like(zmat)
    cmat[~np.isnan(zmat)]=bmat[~np.isnan(zmat)]

    if params['name']=='AHC':
        fac=params['fac']
        clnmbr=params['clnbr']
        pdist=scich.distance.pdist(cmat)
        lkg=scich.linkage(pdist, method='ward')
        c_th=np.max(pdist)*fac
        ids_clust = scich.fcluster(lkg,c_th,criterion='distance')-1
        #ids_clust = scich.fcluster(lkg,clnmbr,criterion='maxclust')-1


    #gmm = mixture.GaussianMixture( n_components=2, covariance_type="full" ).fit(cmat)
    #ids_clust = gmm.predict(cmat)

    ## estimate bandwidth for mean shift
    #bandwidth = cluster2.estimate_bandwidth(cmat, quantile=.5)
    #ms = cluster2.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(cmat)
    #ids_clust = ms.labels_

    #optics = OPTICS(    max_eps=.3).fit(cmat) 
    #ids_clust=optics.labels_
    elif params['name']=='DB':

        DBSCAN_cluster = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(cmat) 
        ids_clust= DBSCAN_cluster.labels_

    #two_means = cluster2.MiniBatchKMeans(n_clusters = 2).fit(cmat)
    #ids_clust= two_means.labels_

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





def find_condition(input_nums,condition_dict):


    # Create a mapping for keys that have both "side" and "center" suffixes
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

    result = [num for num in input_nums if num in side_center_mapping]

    if not result:
        return None

    return side_center_mapping[result[0]]





def plot_samples2(templates,isort,samples,ax,fac=1,nc=-1):
    cmap=plt.get_cmap('Set3')
    ncells=len(templates[list(templates.keys())[0]])
    linsp=np.arange(1,ncells+1)

    nshift=0
    for tmpl in templates.values():
        #print(templates, tmpl)
        if nc==-1:
            ax.plot(nshift+tmpl[isort]*fac, linsp, '.r') 
        else:
            colrgb=np.array(cmap.colors[np.mod(nc,12)]).reshape(1,3)
            ax.plot(nshift+tmpl[isort]*fac, linsp, '.', color=colrgb) 
        nshift += .21
        
    for nsmpl in range(len(samples)):
        ax.plot(nshift+samples[nsmpl][isort]*fac,linsp, '.k')
        nshift +=.21
        
    ax.set_xlabel('Seq. #')
    ax.set_ylabel('Sort seq#')

    





def subsampling_perms(labindices,traintarget): 

    """
    subsampling_perms
    ______________________
    
    This is a function that shuffles the data indices so that 
    different parts of the dataset are represented within each
    subsampling.
    
    Input:  
            - a list holding the indices of the specific label.
    
            - an integer equal to the amount of patterns needed.                  
            
    Output: 
            - an array holding the indices for the test set.
    
            - an array holding the indices for the train set.
              
    """    
    train_perms = []
    test_perms = []
    # Get the first permutation
    lperm_i = list(np.random.permutation(labindices)) 
    train_perms = lperm_i[0:traintarget]
    test_perms = lperm_i[traintarget:]  

    return train_perms, test_perms
def train_test_model(x_dt,y_lb,subsamplings):

    """
    train_test_model
    ______________________
    
    This is a function that decodes the data using the model of choise
    A 2-fold subsampling procedure is implemented.
    The label representation/contribution is kept within each fold.
    
    Input:  
            - a numpy array holding the dataset. 
            - a numpy array holding the labels.
            - an integer equal to the dataset dimensionality.
            - the learning rate and the amount of epochs for the model.
            - an integer value equal to the amount of subsamplings requested.
            - the type of model requested.
            
            Note : an easily additional input could be the option to have
            ballanced label contribution  within subsamplings.
            Uppon request I can implement that. 
            
    Output: 
            - a list holding two arrays one with the mean ccr values 
              of all subsamplings and one holding a ccr value for 
              each pattern calculated over the subsampling process.               
                  
    """
    
    # Get the indices per label and their contribution % in train test sets
    label0_indices = [ind for ind,x in enumerate(y_lb) if x==0]
    label1_indices = [ind for ind,x in enumerate(y_lb) if x==1]    
    # Get the size of the training and testing sets
    train_size = int(np.ceil(x_dt.shape[0]*.8))
    # Get the ratio of each label in the original set
    label0_ratio = len(label0_indices)/(len(label0_indices)+len(label1_indices))
    label1_ratio = len(label1_indices)/(len(label0_indices)+len(label1_indices))
    # Get the target size for each label for the training-testing set

    minlen=np.min((len(label0_indices),len(label1_indices)))
    label0_train_target = int((minlen)/2)#int(np.ceil(train_size*label0_ratio))
    label1_train_target = int((minlen)/2)#int(np.ceil(train_size*label1_ratio))
    
    # Innitiallize an array that will hold the ccrs
    ccrs = []
    # Initiallize two arrays that will hold the ccr per bin
    correct = np.zeros((subsamplings,x_dt.shape[0]))
    used    = np.zeros((subsamplings,x_dt.shape[0])) 
    # Loop over the # subsamplings 
    for nsub in range(subsamplings):
        
        # Generate n distinct permutations of the indices per label to be used in each subsampling
        train0_perms,test0_perms = subsampling_perms(label0_indices,label0_train_target)
        train1_perms,test1_perms = subsampling_perms(label1_indices,label1_train_target)
        # Unite the indices from each label
        #ind_train = train0_perms + train1_perms
        #ind_test  = test0_perms + test1_perms      
        ind_train = train0_perms[:label0_train_target] + train1_perms [:label0_train_target] 
        ind_test  = test0_perms[:label0_train_target] + test1_perms[:label0_train_target] 
        # Pick the data needed for the subsampling
        y_train = np.copy(y_lb[ind_train])
        y_test  = np.copy(y_lb[ind_test])
        X_train = np.copy(x_dt[ind_train,:])
        X_test  = np.copy(x_dt[ind_test,:])  
        
        if np.sum(X_train)==0:
                X_train[0]=0.00000001
              
        # if len(np.unique(y_train))<2:
        #     continue  
        
        #         continue
        # if np.sum(np.asarray(X_train)[np.asarray(y_train==0)])==0:
        #       continue
        # if np.sum(np.asarray(X_train)[np.asarray(y_train==1)])==0:
        #       continue
        # Create an instance of the LDA model
        lda = LinearDiscriminantAnalysis(solver='lsqr')
        #print('len labels=',len(np.unique(y_train)),np.sum(X_test))
        # Fit the model to the training data
        lda.fit(X_train, y_train)

        # Predict the class labels for the test set

        y_pred = lda.predict(X_test)

        # Calculate the accuracy of the model
        #accuracy = balanced_accuracy_score(y_test, y_pred)
        #print("Accuracy: "+cod, accuracy)

        
        cnt=0
        ynt=0
        for iy in range(len(y_pred)):
                if y_pred[iy]==1 and y_test[iy]==1:    
                        cnt+=1
                elif y_pred[iy]==0 and y_test[iy]==0:    
                        ynt+=1
        # print('correct',cnt/sum(y_test))
        # print('failed',ynt)


        ccrs.append(np.sum(y_test==y_pred) / len(y_test))
    return ccrs




def apply_masks_test(sess_info,Masks,cond_numbers,cond_name,sessin_numbers,odd_even,sess_name,trial_type,phase):

    run_data={'idpeaks_cells':[[] for _ in range(len(sess_info['Spike_times_cells']))],'mask_cond_fr_cells':[[] for _ in range(len(sess_info['Spike_times_cells']))]}


    if odd_even != None: 
        mask_odd= np.asarray(Masks['odd_even'])==odd_even# mask for odd or even trials(0/1)
        mask_odd_seqs= np.asarray(Masks['odd_even_seqs'])==odd_even# mask for sequences of odd or even trials(0/1)
        mask_odd_fr= np.asarray(Masks['odd_even_fr'])==odd_even

    else:
        mask_odd=np.ones(len(Masks['phases'])).astype(bool)# phases should be replaced with real mask of odd or even. its just temproray as i dont use odd even now
        mask_odd_seqs= np.ones(len(Masks['bursts_phase'])).astype(bool)# mask for sequences of odd or even trials(0/1)
        mask_odd_fr=  np.ones(len(Masks['fr_phase'])).astype(bool) 




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



    mask_cond_cell=[ np.zeros_like(x).astype(bool) for x in Masks['cell_cond'] ]
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


    #run_data['mask_cond_t']= np.asarray(Masks['conditions'])[mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']]

    #run_data['mask_cond_fr']=np.asarray(Masks['fr_cond'])[mask_sess_fr & mask_cond_fr &  mask_correct_fr & mask_phase_fr]
    run_data['mask_cond_fr']=np.asarray(Masks['bursts_cond'])[mask_sess_burst & mask_cond_burst & mask_odd_seqs & mask_correct_seqs & mask_phase_seqs & Masks['speed_seq']]

    #run_data['idpeaks_cells']=np.asarray(sess_info['Spike_times_cells'][celid])[mask_sess_cell & mask_cond_cell & mask_phase_cell]

    for celid in range(len(sess_info['Spike_times_cells'])):
        msk_crct=np.asarray(Masks['cell_correct'][celid])==bool(trial_type)
        run_data['idpeaks_cells'][celid] = np.asarray(sess_info['Spike_times_cells'][celid])[msk_crct  & mask_sess_cell[celid] & mask_cond_cell[celid] & mask_phase_cell[celid]]
        run_data['mask_cond_fr_cells'][celid] = np.asarray(Masks['cell_cond'][celid])[msk_crct  & mask_sess_cell[celid] & mask_cond_cell[celid] & mask_phase_cell[celid]]




   
    run_data['trial_idx_mask']=np.asarray(sess_info['trial_idx_mask'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]

    run_data['t']=np.asarray(sess_info['t'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['poprate']=np.asarray(sess_info['pop_rate'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    cell_trace_sess1=np.asarray(sess_info['extract'][celid])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']] 

    run_data['trace_cells']=np.asarray([x[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']] for x in sess_info['extract']])# raw trace of all cells

    run_data['lin_pos']=np.asarray(sess_info['lin_pos'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]

    run_data['conditions']=np.asarray(Masks['conditions'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]


    run_data['x_loc']=np.asarray(sess_info['xloc'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['y_loc']=np.asarray(sess_info['yloc'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['speed']=np.asarray(sess_info['speed'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]

    

    run_data['fr']=np.asarray(sess_info['fr'])[mask_sess_fr & mask_cond_fr &  mask_correct_fr & mask_phase_fr]


    t_all=np.asarray(sess_info['t'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase ]

    spk_times=np.where(np.isin(t_all,run_data['idpeaks']))# time of spike for population rate in the determined condtions
    
    #spk_times=t_all[np.isin(t_all,run_data['idpeaks'])]

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




def pc_faction_in_sequnce(Masks,sess_info,sig_pc_idx_ph,cond_names):
    '''This function finds the precentage of the cells in a sequence that are place cells.

''' 
    PC_frac_in_seq={}
    tasks=['sampling','outward','reward','inward']

    cnt=-1
    for phs in range(2):# learned 
        PC_frac_in_seq_corr={}
        
        for correct in range(2):# correct trials


            #fig, ax = plt.subplots(3, 1, figsize=(7, 10))

            cnt=cnt+1
            if phs==1:
                mode='learned'
            else:
                mode='learning'

            if correct==1:
                typoftrial='correct_trials'
            else:
                typoftrial='failed_trials'

            sig_pc_idx=sig_pc_idx_ph[mode]# the indices of the si/pc/tc cells from learning or learned

            ph_mask=np.asarray(Masks['bursts_phase'])==phs
            correct_mask=np.asarray(Masks['correct_failed_seqs'])==correct


            cond_seqs={}
            pc_ratio={}
            seq_len={}
            pc_ratio2={}



            for itsk, tsk in enumerate(tasks):
                mskcnd=np.zeros_like(Masks['bursts_cond']).astype(bool)
                #mskcnd=np.zeros_like(Masks['bursts_cond'],type=bool)
                for icond, condname_r in enumerate(cond_names):
                    if tsk in condname_r:
                        #print(cond_names[condname_r])
                        mskcnd+=(np.asarray(Masks['bursts_cond'])==cond_names[condname_r])
                cond_seqs[tsk]=np.asarray(sess_info['seqs'])[mskcnd & correct_mask & ph_mask] 


            #selected_seqs = np.asarray(sess_info['seqs'])[(np.asarray(Masks['bursts_cond'])==8)|(np.asarray(Masks['bursts_cond'])==9)|(np.asarray(Masks['bursts_cond'])==10)|(np.asarray(Masks['bursts_cond'])==11)]


                title=tsk
                pc_seq_lengh=np.zeros(len(cond_seqs[tsk]))# precentage of place cells that are contibuted in a sequence

                pc_seq_ratio=np.zeros(len(cond_seqs[tsk]))# precentage of place cells that are contibuted in a sequence
                seq_pc_ratio=np.zeros(len(cond_seqs[tsk]))# precentage of sequences that are place cells 
                if len(cond_seqs[tsk])>0:
                    len_seq_max=np.max([len(x) for x in cond_seqs[tsk]])
                for iseq,seq in enumerate(cond_seqs[tsk]):
                    pc_seq_ratio[iseq]=(np.sum(np.isin(seq,sig_pc_idx))/len(sig_pc_idx[0]))# how many precent of the place cells conributed in this sequence
                    seq_pc_ratio[iseq]=(np.sum(np.isin(seq,sig_pc_idx))/len(seq))# how many precents of the cells in this sequence are place cells
                    pc_seq_lengh[iseq]=(len(seq))

                pc_ratio[tsk]=seq_pc_ratio
                seq_len[tsk]=pc_seq_lengh
                pc_ratio2[tsk]=pc_seq_ratio


                # PC_frac_in_seq[tsk]=pc_seq_lengh
                # PC_frac_in_seq[tsk]=pc_seq_ratio
            PC_frac_in_seq_corr[typoftrial]= pc_ratio  
        
        PC_frac_in_seq[mode]=PC_frac_in_seq_corr
    return PC_frac_in_seq



import scipy.stats as stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy.ndimage import gaussian_filter1d 









def plot_kl_distributions_ss(js_divergence_ss,p_value_corr_js_,name,type='Correct'):
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
