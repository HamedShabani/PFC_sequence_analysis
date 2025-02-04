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





















def popbursts_old(mat,sig,fs):
    global Tspare
    
    from scipy.signal import find_peaks
    poprate=np.sum(mat,axis=0)
    thresh=np.mean(poprate)+sig*np.std(poprate)
    spare=int(Tspare*fs)
    
    mask=(poprate>=thresh)*1.
    mask=1.*(np.diff(mask)==1)
    ids=np.where(mask>0)[0]
    
    vec=[]
    idprev=-1
    idpeaks=[]


    for n in range(len(ids)):

        id0=ids[n]

        if id0+spare<=mat.shape[1]:
            idpeak=np.argmax(poprate[id0:id0+spare])+id0
            if (idpeak-idprev>spare)*(idpeak-int(spare/2)>=0)*(idpeak+int(spare/2)+1<=mat.shape[1]):
                vec.append(mat[:,idpeak-int(spare/2):idpeak+int(spare/2)+1])
                idprev=idpeak
                idpeaks.append(idpeak)
            elif (idpeak<=int(spare/2)):# peaks in the beggining  smaller than spare time
                continue # ignore the events shorter than 5 time stamps (in the beginig)
                #vec.append(mat[:,idpeak:idpeak+int(spare)+1])
                #vec.append(mat[:,idpeak:idpeak+int(spare)])

                idprev=idpeak
                idpeaks.append(idpeak)
                
    if len(vec)>0:
        itax=np.arange(vec[0].shape[1])

    seq=[]
    spike_time=[[] for kk in range(np.shape(mat)[0]) ]
    raster = np.zeros((np.shape(mat)[0], np.shape(mat)[1]))
    raster2 = np.zeros((np.shape(mat)[0], len(vec)))


    for nv in range(len(vec)):
        #itax=np.arange(vec[nv].shape[1])# hamed added
        nvec=np.sum(vec[nv],axis=1)
        cofseq=(itax@vec[nv].transpose())/nvec
        tmp=np.argsort(cofseq)
        seq.append(tmp[~np.isnan(np.sort(cofseq))])

        for kk in seq[nv]:
            spike_time[kk].append(idpeaks[nv])

        raster[seq[nv],idpeaks[nv]] = 1
        raster2[seq[nv],nv] = 1




                           
    return vec,seq,ids,raster2,spike_time,idpeaks,poprate
                       



def plot_sequences(mat,seq,vec,idpeaks,sig,fs,speed_trl,poprate):
    #poprate=np.sum(mat,axis=0)
    thresh=np.mean(poprate)+sig*np.std(poprate)
    import seaborn as sns
    if 1:
        sorted=True
        Plot_entrire_recording=True
        sequencnmbr=0
        
        if Plot_entrire_recording:
           dta=mat
           colors='k' 
           linewidth=1 
           fsize=18    
        # elif sorted==True:
        #     dta = vec[sequencnmbr][seq[sequencnmbr]]
        #     colors='r'
        #     ttitle='sorted'
        #     linewidth=3  
        #     fsize=20  
        # else:
        #     dta= vec[sequencnmbr]
        #     colors='b'
        #     ttitle='unsorted'
        #     linewidth=3   
        #     fsize=20 
        


        #khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) if np.nansum(x)!=0]
        khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) ]
        khers=np.nan_to_num(khers, copy=True, nan=0.0)

        #clidx=[i for i, x in enumerate(dta) if np.nansum(x)!=0]
        clidx=[i for i, x in enumerate(dta) ]

        ticks = np.linspace(0,np.shape(khers)[1]-1,5)
       
        ticklabels = ["{:0.0f}".format(i) for i in ticks/fs]

        palette = sns.color_palette("hls", np.shape(khers)[-1])
        with plt.rc_context({'font.size': fsize}):
            fig=plt.figure(figsize=(30,10))
            if Plot_entrire_recording:
                ax0=fig.add_axes([0.1,0.1,0.4,.7])

                ax=fig.add_axes([0.1,.8,0.4,.1])
                ax.vlines(idpeaks,ymin=-1,ymax=max(poprate+10),linewidth=1,color='k',alpha=.8)
                ax.vlines(idpeaks,ymin=0,ymax=max(poprate+10),linewidth=10,color='g',alpha=.3)
                ax.plot(poprate,linewidth=5)
                ax.hlines(thresh,xmin=0,xmax=len(poprate),color='r',linestyles= 'dashed')
                ax.set_xlim([0,len(poprate)])
                ax.set_ylabel('ΔF/F0')
                #ax.set_title('Detected bursts of active cells')
                ax.set_title('Detected bursts')

                ax.set_xticks([])
                ax.set_yticks([])


                ax0.set_ylim(0,len(clidx))
                ax0.set_xlim([0,len(poprate)])

                # ax3=fig.add_axes([0.1,.9,0.4,.1])
                # ax3.plot(speed_trl,linewidth=4,color='green')
                # plt.xlim([0,len(poprate)])
                # ax3.set_ylabel('Speed Cm/sec')

                ax0.vlines(idpeaks,ymin=0,ymax=len(clidx),linewidth=1,color='k',alpha=.8)
                ax0.vlines(idpeaks,ymin=0,ymax=len(clidx),linewidth=10,color='g',alpha=.3)
                i=0
                for kk,ii in enumerate(khers):
                    ax0.plot(ii+i,color='k', linewidth=linewidth)

                    i=i+1



            # if sorted == True:
            #     plt.yticks(np.arange(len(seq[sequencnmbr])),seq[sequencnmbr])
            # else:
            #     plt.yticks(np.arange(len(clidx)),clidx)
            ax0.set_xticks(ticks,ticklabels)

            ax0.set_xlabel('time s')
            ax0.set_ylabel('Cell #')
            #ax2.set_title(ttitle) 







            with plt.rc_context({'font.size': fsize}):
                fig2=plt.figure(figsize=(5,10))
                ax1=fig2.add_axes([0.1,0.1,0.4,.7])
                ax2=fig2.add_axes([0.65,0.1,0.4,.7])

                dta = vec[sequencnmbr][seq[sequencnmbr]]
                colors='r'
                ttitle='sorted'
                linewidth=3  
                fsize=20  
                khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) if np.nansum(x)!=0]
                clidx=[i for i, x in enumerate(dta) if np.nansum(x)!=0]

                i=0
                for kk,ii in enumerate(khers):
                        ax2.plot(ii+i,color=colors, linewidth=linewidth)
                        i=i+1

                ax2.set_title(ttitle)
                ax2.set_yticks(np.arange(len(seq[sequencnmbr])),seq[sequencnmbr])



                dta= vec[sequencnmbr]
                colors='b'
                ttitle='unsorted'
                khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) if np.nansum(x)!=0]
                clidx=[i for i, x in enumerate(dta) if np.nansum(x)!=0]
                i=0
                for kk,ii in enumerate(khers):
                        ax1.plot(ii+i,color=colors, linewidth=linewidth)
                        i=i+1

                ax1.set_title(ttitle)
                ax1.set_yticks(np.arange(len(seq[sequencnmbr])),np.sort(seq[sequencnmbr]))
                
                ticks = np.linspace(0,np.shape(khers)[1]-1,3)
                ticklabels = ["{:0.2f}".format(i) for i in ticks/fs]
                ax1.set_xticks(ticks,ticklabels,fontsize=14)
                ax1.set_xlabel('time s')
                ax1.set_ylabel('Cell index')

                ax2.set_xticks(ticks,ticklabels,fontsize=14)
                ax2.set_xlabel('time s')
                #ax2.set_ylabel('Cell #')
                #plt.yticks(np.arange(len(seq[sequencnmbr])),seq[sequencnmbr])
                #plt.xticks(np.arange(0,len(poprate),50),np.arange(0,len(poprate),50)/fs)


                
            
    if 0:
        maxlen=np.max([len(x) for x in seq])
        sorted_seq =[seq[x] for x in np.argsort(ids_clust)]
        seq_equal=([np.concatenate((x,np.ones(maxlen-len(x))*np.nan)) for x in seq])
        plt.figure(figsize=(20,20))
        sns.heatmap(seq_equal,annot=True,fmt=".0f")
    #return (ax3,poprate,ticks,ticklabels)


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


def tsne_plot(bursts,ids_clust,temp_info,filenames,savefolder):
    rbursts=(np.reshape(bursts,(np.shape(bursts)[0],np.shape(bursts)[1]*np.shape(bursts)[2])))
    #if temp_info['within_ratio']:
        #best_template_idx = np.argmax(temp_info['within_ratio'])

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(rbursts)



    #for rn in range(len (temp_info['exclude'])):
    #    ids_clust=ids_clust[~np.array(temp_info['exclude'][rn]==ids_clust)]
    
    fcls=np.array(list(map(int, ids_clust)))

    n_flat_clusters_ = len(np.unique(ids_clust))
    show_order_ = np.unique(fcls)[::-1]
    n_flat_clusters = np.unique(fcls).shape[0]

    perplexity = 5

    model = TSNE(n_components=2, random_state=0, perplexity=perplexity,n_iter=1000, learning_rate=20)#,init='pca')
                
    
    proj = model.fit_transform(rbursts) 

    mpl.rcParams["figure.titlesize"] = 'large'
    fig ,ax=plt.subplots(1,3,figsize=(18,12))
    ax[0].scatter(proj[:,0],proj[:,1],s=66,lw=0,c=show_order_[fcls],
                    cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
    ax[0].set_title('Tsne')


    ax[1].scatter(principalComponents[:,0],principalComponents[:,1],s=66,lw=0,c=show_order_[fcls],
                cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
    ax[1].set_title('PCA')
    fig.suptitle(filenames)

    img = ax[2].imshow(rbursts[np.argsort(ids_clust)],aspect='auto',cmap='gray_r',vmin=-1,vmax=10)
    ax[2].set_title('Bursts')
    ax[2].set_xlabel('Bursts (concat of all neurons)')
    ax[2].set_ylabel('Sequence number')
    hlines = np.where(np.diff(np.asarray(ids_clust)[np.argsort(ids_clust)]))[0]
    #ax[2].set_yticks(hlines)
    #ax[2].set_yticklabels(ids_clust[1+hlines])   
    ax[2].hlines(hlines, xmin=0,xmax=np.shape(rbursts)[1]-10,color='k')
    #best_temp_sorted_idx = np.where(np.sort(ids_clust) == best_template_idx)
    #ax[2].vlines(0, ymin=best_temp_sorted_idx[0][0],ymax=best_temp_sorted_idx[0][-1],color='r',linestyles='dashed',linewidth=5,
    #    antialiased=False)

    fig.colorbar(img,  orientation='vertical')

    


    plt.savefig(savefolder+'Tsne'+filenames+'.png')
    #plt.savefig(savefolder+'Tsne'+filenames+'.svg')

    plt.show(block=0)

    plt.figure(figsize=(4,3))
    [plt.plot(np.mean(np.mean(np.array(bursts)[fcls==clstr],axis=0),axis=0)) for clstr in show_order_ ]

    return(proj,n_flat_clusters_,show_order_,n_flat_clusters,fcls)


   
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



def plot_ranksumpvalue(pvals, condname,fig):

    #plt.figure().canvas.draw()
    plt.imshow(pvals<.05)
    plt.xticks( np.arange(len(condname)), condname, rotation='vertical')
    plt.yticks( np.arange(len(condname)), condname)
    plt.title('Ranksum p-value(0.05)')
    return fig


def plot_sequence_number(session_templantes,conds,savefolder,fs,foldername):
    #tasks = ['arena','sampling','outward','inward','reward','sleep']
    tasks=['arena','failed_trials','correct_trials','sleep']
    AA=[]
    BB=[]
    Rates=[]
    Counts=[]
    Namesofsession=[]
    AAcnt = []
    BBcnt = []
    colornmbr = []
    colornmbr2 = []
    Seqnmbr = []
    Seqrates = []
    Seqratename = []
    Ratesappend=[]
    Ratesappend_sig=[]
    for conname in conds:
       
        rate = [(fs*np.size(session_templantes[namekey]['seqs'])/np.array(session_templantes[namekey]['duration']),namekey) for namekey in session_templantes.keys() if conname in(namekey) if np.size(session_templantes[namekey]['seqs'])>0]
        count = [(np.size(session_templantes[namekey]['seqs']),namekey) for namekey in session_templantes.keys() if conname in(namekey) if np.size(session_templantes[namekey]['seqs'])>0]
        Rates.extend(rate)
        Counts.extend(count)
        if rate:
            Ratesappend.append(rate)
            

        namesofsession = [namekey  for namekey in session_templantes.keys() if conname in(namekey) if np.size(session_templantes[namekey]['seqs'])>0]
        colornmbr.append(len(namesofsession))        
        Namesofsession.extend(namesofsession)



        seqnmbr = [(np.mean(session_templantes[keyname]['sequencerate']), keyname) for keyname in session_templantes.keys() if conname in(keyname) if np.size(session_templantes[keyname]['sequencerate'])>0]
        seqrates = [(np.sum(session_templantes[keyname]['sequencerate'])/np.array(session_templantes[keyname]['duration']), keyname) for keyname in session_templantes.keys() if conname in(keyname) if np.size(session_templantes[keyname]['sequencerate'])>0 ]
        seqratename = [keyname for keyname in session_templantes.keys() if conname in(keyname) if np.size(session_templantes[keyname]['sequencerate'])>0]
        Seqnmbr.extend(seqnmbr)
        Seqrates.extend(seqrates)
        Seqratename.extend(seqratename)
        colornmbr2.append(len(seqratename))    
        if seqrates:
            Ratesappend_sig.append(seqrates)
                

        
    Tsk_rate=[]
    for tsk in tasks:# rate for defined tasks
        tsk_rate = [(fs*np.size(session_templantes[namekey]['seqs'])/np.array(session_templantes[namekey]['duration']),tsk) for namekey in session_templantes.keys() if tsk in(namekey) if np.size(session_templantes[namekey]['seqs'])>0]
        Tsk_rate.append(tsk_rate)

    Tsk_rate_sig=[]
    for tsk in tasks:# rate for defined tasks
        tsk_rate_sig = [(fs*np.size(session_templantes[namekey]['sequencerate'])/np.array(session_templantes[namekey]['duration']),tsk) for namekey in session_templantes.keys() if tsk in(namekey) if np.size(session_templantes[namekey]['sequencerate'])>0]
        Tsk_rate_sig.append(tsk_rate_sig)





    pvals_conds,condname_conds = ranksum_pvalue(Ratesappend, conds)
    pvals_tasks,condname_tasks= ranksum_pvalue(Tsk_rate, tasks)

    pvals_conds_sig,condname_conds_sig = ranksum_pvalue(Ratesappend_sig, conds)
    pvals_tasks_sig,condname_tasks_sig= ranksum_pvalue(Tsk_rate_sig, tasks)


    fig = plt.figure(figsize=(10,8))
    #plt.show(block=False)

    #ax=fig.add_axes([0.1,0.1,0.5,0.5])
    ax=fig.add_subplot(221)

    fig = plot_ranksumpvalue(pvals_conds, condname_conds,fig)
    ax.set_title('Ranksum p-value map for conds (0.05)')
    ax.set_xticks(color='w',ticks=[])
    #ax=fig.add_axes([0.5,0.1,1,1])
    ax=fig.add_subplot(222)

    fig =plot_ranksumpvalue(pvals_tasks, condname_tasks,fig)
    ax.set_title('Ranksum p-value map for tasks (0.05)')
    ax.set_xticks(color='w',ticks=[])



    ax=fig.add_subplot(223)
    fig = plot_ranksumpvalue(pvals_conds_sig, condname_conds_sig,fig)
    ax.set_title(' p-value map for significant conds (0.05)')
    
    #ax=fig.add_axes([0.5,0.1,1,1])
    ax=fig.add_subplot(224)
    fig =plot_ranksumpvalue(pvals_tasks_sig, condname_tasks_sig,fig)
    ax.set_title(' p-value map for significant tasks (0.05)')

    plt.suptitle(foldername)
    plt.savefig(savefolder+'Ranksum pvalue '+foldername +'.png',dpi=300, bbox_inches = "tight")
    plt.savefig(savefolder+'Ranksum pvalue '+foldername +'.pdf',dpi=300, bbox_inches = "tight")
    plt.show()






    Tasknames=[]
    for tsk in tasks:# rate for defined tasks
        tasknames=[tsk for names in list(zip(*Seqrates))[1] if tsk in names]
        Tasknames.extend(tasknames)

    df2=pd.DataFrame({"rate":list(zip(*Seqrates))[0],
                        "name":Tasknames})
    
    sns.swarmplot(data = df2, y = "rate", x = "name")
    plt.xticks(rotation='vertical')
    plt.title(' Rate of significant seqs ' )






    palette = sns.color_palette("tab10", len(colornmbr))
    colorall = [ (palette[i],)*clr for i,clr in enumerate(colornmbr)]
    colorall = sum(colorall, ())
    

    palette2 = sns.color_palette("tab10", len(colornmbr2))
    colorall2 = [ (palette2[i],)*clr for i,clr in enumerate(colornmbr2)]
    colorall2 = sum(colorall2, ())
    with plt.rc_context({'font.size': 12}):
            
        plt.figure(figsize=(2*len(colornmbr),np.round(len(colornmbr)/2)))
        plt.subplot(223)
        barlist =  plt.bar( list(zip(*Rates))[1],list(zip(*Rates))[0])
        for item, color in zip(barlist, colorall):
            item.set_color(color)
        # plt.text(item.xy[0],item.get_height(),'*')


        plt.xticks(rotation='vertical')
        plt.ylabel('Hz')
        plt.title(' Sequence rate ' )

        plt.subplot(221)
        barlist =  plt.bar( list(zip(*Counts))[1],list(zip(*Counts))[0])
        for item, color in zip(barlist, colorall):
            item.set_color(color)
        plt.ylabel('#')
        plt.title(' Sequence Number ' )
        plt.xticks([''])

        plt.subplot(224)
        barlist =  plt.bar(list(zip(*Seqrates))[1],list(zip(*Seqrates))[0])
        for item, color in zip(barlist, colorall2):
            item.set_color(color)
        plt.xticks(rotation='vertical')
        plt.ylabel('Hz')
        plt.title('Sequence rate for significant sequences (sig seq)/time ' + foldername[0:3])

        plt.subplot(222)
        barlist =  plt.bar(list(zip(*Seqnmbr))[1],list(zip(*Seqnmbr))[0])
        for item, color in zip(barlist, colorall2):
            item.set_color(color)
        plt.ylabel('%')
        plt.title('Sequence average for significant sequences (sig seqs)/(total seqs) ' + foldername[0:3])
        plt.xticks([''])
        # aa=[[conname ,fs*np.size(session_templantes[namekey]['seqs'])/np.array(session_templantes[namekey]['duration'])] for namekey in session_templantes.keys() if conname in(namekey) ]
        # bb=[[conname ,np.size(session_templantes[namekey]['seqs'])] for namekey in session_templantes.keys() if conname in(namekey) ]

        # if len(aa)>0:
        #     seqnmbr=0
        #     for i in aa:
        #         if np.array(i[1]).size:
        #             seqnmbr  += np.array(i[1])

        #     AA.append(seqnmbr/len(aa))
        #     BB.append(aa[0][0])

        # if len(bb)>0:
        #     seqnmbr_cnt=0
        #     for ii in bb:
        #         if np.array(ii[1]).size:
        #             seqnmbr_cnt  += np.array(ii[1])

        #     AAcnt.append(seqnmbr_cnt/len(bb))
        #     BBcnt.append(aa[0][0])

        # plt.figure(figsize=(10,5))
        # plt.subplot(212)
        # plt.bar(BB,AA)
        # plt.xticks(rotation='vertical')
        # plt.ylabel('Hz')
        # plt.title(' Sequence rate ' + foldername)

        # plt.subplot(211)
        # plt.bar(BBcnt,AAcnt)
        # plt.ylabel('#')
        # plt.title(' Sequence Number ' + foldername)
        # plt.xticks([''])

        plt.savefig(savefolder+'Sequence rate '+foldername +'.png',dpi=300, bbox_inches = "tight")
        plt.savefig(savefolder+'Sequence rate '+foldername +'.pdf',dpi=300, bbox_inches = "tight")
        plt.show()


    

# def extract_data(data,learn_data,condition='correct_trials',signal_type='transients',task_name='learning',remove_failed_trials='False'): # extract the specified index from selected trials and returns it
#     #thr_burts=1# this line should be relocated
#     class sessionsdata():
#         def __init__(self, name, data,loc,trlidx,trial_data,id_peaks):
#             self.name = name
#             self.data = data
#             self.loc = loc
#             self.trlidx=trlidx
#             self.trial_data=trial_data
#             self.id_peaks=id_peaks

#     if len(data['EvtT'][condition])>0:
#         newidx=np.argsort(data['metadata']['CellRegCellID'][data['metadata']['CellRegCellID']>-1])
#         STMx=np.array(data['STMx'][signal_type])

#         STMx1 = np.zeros_like(STMx)
#         STMx1[newidx] = np.array(STMx)# whole recording
#         #STMx1 = np.array(STMx[newidx])
#         fs_str = data['metadata']['recordingmethod']['sampling_frequency']
#         fs=float(fs_str[0:2])
#         if fs_str =='kHz':
#             fs *= 1000
#         #winlen=fs*3# 3 second
#         poprate,id_peaks,bursts,seqs,spike_times,rasters = binned_burst(STMx1,winlen,thr_burts,fs,)


#         selected_trials = np.array(data['EvtT'][condition])      
#         correct_trials = np.array(selected_trials.reshape(int(len(selected_trials)/2), 2))
        
#         if remove_failed_trials=='True':
            
#             failed = np.array([x in (data['EvtT']['failed_trials'][0::2]) for x in data['EvtT'][condition][0::2]])
#             correct_trials= correct_trials[~failed]
            
#         trial_data=[]
#         extract=[]
#         extract_xloc=[]
#         extract_yloc=[]
#         for x in correct_trials:
#             trial_data.append(np.array(STMx1[:,x[0]:x[1]]))# valid trials
#             extract.extend(np.transpose(np.array(STMx1[:,x[0]:x[1]])))# valid trials
#             extract_xloc.extend(data['EvtT']['x'][x[0]:x[1]])
#             extract_yloc.extend(data['EvtT']['y'][x[0]:x[1]])

#         extract=np.array(np.transpose(extract)) 

#         #learn_data[condition] = extract # put data from selected trials

#         xloc= np.expand_dims(np.array(data['EvtT']['x']), axis=0)
#         yloc= np.expand_dims(np.array(data['EvtT']['y']), axis=0)
#         EvtT=np.concatenate((xloc,yloc))

#         EvtT2 = np.concatenate((np.expand_dims(np.array(extract_xloc), axis=0),np.expand_dims(np.array(extract_yloc), axis=0)))
        


#         learn_data[condition] =   sessionsdata(condition,extract,EvtT2,correct_trials,trial_data,id_peaks)

#         if 0:
#             fig=plt.figure()
#             ax=fig.add_axes([0.1,0.1,0.4,0.8])
#             ax.plot(EvtT[0],EvtT[1], '-k.')
#             ax.plot(EvtT2[0],EvtT2[1], 'b.')
#             ax.set_title(condition)
#             plt.show(block=0)
#         #plt.imshow(extract,aspect='auto',cmap='tab20')
#         #plt.title(task_name+ '  '+ condition)
#         #plt.show()
#     else:
#         extract=[]
#         EvtT2=[]
#         trial_data=[]
#         correct_trials=[]
#         id_peaks=[]
#         learn_data[condition] = sessionsdata(condition,extract,EvtT2,correct_trials,trial_data,id_peaks)




#     return learn_data


def deconv(signal,tau):
    f_sampling=20
   
    decon=np.diff(signal,axis=1)*f_sampling+(signal[:,0:-1]+signal[:,1:])/2/tau
    decon=decon/np.max(decon,axis=1).reshape(-1,1)# normalize to 1
    return decon


def swarmplot_conds(Tsk_rate_all_extend,tasks,fig,markersizes):
    Tasknames=[]
    for tsk in tasks:# rate for defined tasks
        tasknames=[tsk for names in list(zip(*Tsk_rate_all_extend))[1] if tsk in names]
        Tasknames.extend(tasknames)

    df2=pd.DataFrame({"duration s":np.array(list(zip(*Tsk_rate_all_extend))[0]),
                        "name":list(zip(*Tsk_rate_all_extend))[1]})

    sns.swarmplot(data = df2, y = "duration s", x = "name", hue="name",size=markersizes)
    plt.xticks(rotation='vertical')  








    return fig






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


def seq_rate(temdata,session_templantes,nrm,fs=20):
    seqdata = session_templantes['seqs'][0]
    
    if seqdata:
        lengest_temp_idx1 = np.argmax([len(temps)  for temps in temdata['template']])


        
        #print('idx of longest template= ',lengest_temp_idx1, 'idx highest within_ratio= ',lengest_temp_idx )
        #lengest_temp_idx = np.argmax(temdata['within_ratio'])# get the template with highest in/out ratio instead of longest template
        
        noisy_temp_idx = np.argmin(temdata['within_ratio'])
        #print('Arg of longest template= ',lengest_temp_idx1, '       Arg of best temp=',lengest_temp_idx1  )
        #chck = check_template(seqdata,temdata['template'][noisy_temp_idx],nrm)
        #seqerate=([fs*np.sum(chck[1])/session_templantes['duration']])
        
        
        if len(temdata['within_ratio'])==1:
            chck = check_template(seqdata,temdata['template'][noisy_temp_idx],nrm)
            seqerate = [fs*np.sum(chck[1])/session_templantes['duration']]# rate Hz
            seqratio = [np.mean(chck[1])]# precentage %

        else:
            seqerate=[]
            seqratio=[]
            for chks in range(len(temdata['within_ratio'])):
                if chks != noisy_temp_idx:
                    chck = check_template(seqdata,temdata['template'][chks],nrm)
                    seqerate = [fs*np.sum(chck[1])/session_templantes['duration']]# rate Hz
                    seqratio.extend([np.mean(chck[1])])#precentage %
    else:
        seqerate=  np.nan
        seqratio=np.nan
    return seqerate,seqratio






def plot_speed_location(EvtT,Id_peaks_trl,Speed_sess,X_loc_sess,Y_loc_sess,poprate_all_tril,Poprate,id_peaks_all_trl,speed_all_trl,dt_all_trl,fs,Figs,names,titleax,savefolder):


    X_dist=[]
    ax1s = Figs['ax1']
    ax2s = Figs['ax2']

    ax3s = Figs['ax3']
    ax4s = Figs['ax4']
    ax5s = Figs['ax5']
    ax6s = Figs['ax6']

    ax7s = Figs['ax7']
    ax8s = Figs['ax8']
    ax9s = Figs['ax9']
    ax10s = Figs['ax10']


    with plt.rc_context({'font.size': 20}):                                        #plt.suptitle(' Speed vs location '+fol+'_'+names+'_'+ signal_type)
        ax1s.plot(EvtT[0],EvtT[1], '-.',color='silver')
        for pl in range(len(Speed_sess)):    
            ax1s.plot(X_loc_sess[pl],Y_loc_sess[pl], '.',color='grey',alpha=.3)
            if ('reward' not in names) & ('sampling'not in names) :
                ax1s.plot3D(X_loc_sess[pl],Y_loc_sess[pl],Speed_sess[pl],'--',linewidth=1,alpha=.4,color='green')
            if len(Id_peaks_trl[pl])>0:
                ax1s.plot3D(X_loc_sess[pl][Id_peaks_trl[pl]],Y_loc_sess[pl][Id_peaks_trl[pl]],np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]],"X",color='k',markersize=5)
        
        ax1s.set_title(titleax)
        ax1s.view_init(elev=15, azim=0)
        ax5s.plot(EvtT[0],EvtT[1], '-.',color='silver')
        for pl in range(len(Speed_sess)):    
            ax5s.plot(X_loc_sess[pl],Y_loc_sess[pl], '.',color='grey',alpha=.3)
            #ax5.plot3D(X_loc_sess[pl],Y_loc_sess[pl],Speed_sess[pl],'--',linewidth=1,alpha=.4,color='navy')
            if len(Id_peaks_trl[pl])>0:
                ax5s.plot3D(X_loc_sess[pl][Id_peaks_trl[pl]],Y_loc_sess[pl][Id_peaks_trl[pl]],np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]],"X",color='k',markersize=5)
        ax5s.view_init(elev=90, azim=-90)

        for pl in range(len(Speed_sess)):    
            ax2s.plot(Speed_sess[pl],X_loc_sess[pl], '--',color='grey',alpha=.5)
            if len(Id_peaks_trl[pl])>0:
                ax2s.plot(np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]],X_loc_sess[pl][Id_peaks_trl[pl]],'X',color='red',alpha=.5)
            #ax2.plot3D(Speed_sess[pl],X_loc_sess[pl],Y_loc_sess[pl],'--',linewidth=1,alpha=.4,color='green')
            #ax2.plot3D(np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]],X_loc_sess[pl][Id_peaks_trl[pl]],Y_loc_sess[pl][Id_peaks_trl[pl]],"X",color='k',markersize=5)
        ax2s.set_ylabel('x loc')

                                                

        for pl in range(len(Speed_sess)):    
            ax3s.plot(Speed_sess[pl],Y_loc_sess[pl], '--',color='grey',alpha=.5)
            if len(Id_peaks_trl[pl])>0:
                ax3s.plot(np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]],Y_loc_sess[pl][Id_peaks_trl[pl]],'X',color='red',alpha=.5)
        ax3s.set_xlabel('speed')
        ax3s.set_ylabel('y loc')

        for pl in range(len(Poprate)):    
            if len (Poprate[pl])>0:
                ax10s.plot(Speed_sess[pl],Poprate[pl], '--',color='grey',alpha=.5)
            if len(Id_peaks_trl[pl])>0:
                ax10s.plot(np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]],np.array(Poprate[pl])[Id_peaks_trl[pl]],'X',color='red',alpha=.5)
            #ax2.plot3D(Speed_sess[pl],X_loc_sess[pl],Y_loc_sess[pl],'--',linewidth=1,alpha=.4,color='green')
            #ax2.plot3D(np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]],X_loc_sess[pl][Id_peaks_trl[pl]],Y_loc_sess[pl][Id_peaks_trl[pl]],"X",color='k',markersize=5)
        ax10s.set_ylabel('Avg ΔF/F0')

        all_evnt_speed=[] 
        for pl in range(len(Speed_sess)):
            if len(Id_peaks_trl[pl])>0:
                all_evnt_speed.extend(np.ones(len(Id_peaks_trl[pl]))*Speed_sess[pl][Id_peaks_trl[pl]])
                X_dist.extend(X_loc_sess[pl][Id_peaks_trl[pl]])

        if len(all_evnt_speed)>0:
            ax4s.hist(all_evnt_speed,bins=25)        
            ax4s.set_xlabel('speed')
            ax4s.set_ylabel('histogram')
            ax4s.set_xlim([0,np.max(speed_all_trl)])


        #thresh=np.mean(poprate_all_tril)+thr_burts*np.std(poprate_all_tril)

        khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(np.transpose(dt_all_trl)) if np.nansum(x)!=0]
        ticks = np.linspace(0,np.shape(khers)[1]-1,5)
        ticklabels = ["{:0.0f}".format(i) for i in ticks/fs]
        ax6s.vlines(np.array(id_peaks_all_trl),ymin=-1,ymax=max(np.array(poprate_all_tril)+10),linewidth=1,color='grey',alpha=.7)
        #plt.vlines(id_peaks_all_trl,ymin=0,ymax=max(np.array(poprate_all_tril)+10),linewidth=10,color='g',alpha=.3)
        ax6s.plot(poprate_all_tril,linewidth=1)
        #plt.hlines(thresh,xmin=0,xmax=len(poprate_all_tril),color='r',linestyles= 'dashed')
        ax6s.set_xlim([0,len(poprate_all_tril)])
        ax6s.set_ylabel('Avg ΔF/F0')
        ax6s.set_xlabel('time s')
        ax6s.set_xticks(ticks,ticklabels)

        ax7s.plot(speed_all_trl,linewidth=1,color='green')
        ax7s.vlines(np.array(id_peaks_all_trl),ymin=-1,ymax=max(np.array(speed_all_trl)),linewidth=1,color='grey',alpha=.7)
        ax7s.set_xlim([0,len(poprate_all_tril)])
        ax7s.set_ylabel('Speed')
        ax7s.set_xticks([])



        ax8s.plot(np.concatenate(Y_loc_sess).ravel(),linewidth=1,color='k')
        ax8s.vlines(np.array(id_peaks_all_trl),ymin=np.min(np.array(np.concatenate(Y_loc_sess).ravel())),ymax=max(np.array(np.concatenate(Y_loc_sess).ravel())),linewidth=1,color='grey',alpha=.7)
        ax8s.set_xlim([0,len(poprate_all_tril)])
        ax8s.set_ylabel('y loc')
        ax8s.set_xticks([])

        ax9s.plot(np.concatenate(X_loc_sess).ravel(),linewidth=1,color='k')
        ax9s.vlines(np.array(id_peaks_all_trl),ymin=np.min(np.array(np.concatenate(X_loc_sess).ravel())),ymax=max(np.concatenate(X_loc_sess).ravel()),linewidth=1,color='grey',alpha=.7)
        ax9s.set_xlim([0,len(poprate_all_tril)])
        ax9s.set_ylabel('xloc')
        ax9s.set_xticks([])
        

        plt.savefig(savefolder+'Speed '+titleax+'.png')
        return ax9s



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


























def binned_burst_old(dt,winlen,thr_burts,fs,timewins):

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
            bursts_tmp_s,seqs_tmp_s,ids_temp_s,raster_s,spike_time_s,id_peaks_trl_s,poprate_s = popbursts_new(sdt,thr_burts,fs)






            #print(np.shape(spike_time_s),spike_time_s)
            # plt.figure(figsize=(3,5))
            # plt.vlines(id_peaks_trl_s,np.min(poprate_s),np.max(poprate_s),color='k')
            # plt.plot(poprate_s)


            #if len(poprate_s>0):
                #speed_trl=speed_all[timewins[twin]:timewins[twin+1]]
                #Ax3,popratess,ticks,ticklabels=plot_sequences(sdt,seqs_tmp_s,bursts_tmp_s,id_peaks_trl_s,thr_burts,fs,speed_trl,poprate_s)
                #Ax3.set_title('Short data Find peaks'+names)

            
            raw_data.extend(np.transpose(dt[:,timewins[twin]:timewins[twin+1]]))
            Rasters.extend(np.transpose(raster_s))
            
            lentrials.extend(np.transpose(dt[:,timewins[twin]:timewins[twin+1]]))
            

            
            id_peaks_all.extend([k+dtlen for k in id_peaks_trl_s])# the time of sequences for one condition
            dtlen += np.shape(dt[:,timewins[twin]:timewins[twin+1]])[1]
            #plt.plot(id_peaks_all,[len(x) for x in seqs])
            ncells=dt.shape[0]
            
            # if np.sum(spike_time_s[:])==0:
            #     continue# this should be removed 
            

            
            
            if len([x for x in spike_time_s if len(x)>0])>0:# concat spiketime of all jumps and add the time of each jump
                spike_time_add=[]
                for spkt in spike_time_s:
                    if spkt:
                        spike_time_add.append([i+i0 for i in spkt])# each cell
                    else:
                        spike_time_add.append([])

                spike_time_all.append(np.array(spike_time_add))
                #spike_time_all=(np.array(spike_time_add))
                



                
                if 0:
                    spike_time_allt=np.transpose(spike_time_all)                        
                    Spike_time=[[] for kk in range(ncells)]

                    for nkk in range(ncells):                                                        
                        my_list=[i for i in spike_time_allt[nkk] if i]
                        flat_list = [num for sublist in my_list for num in sublist]                          
                        Spike_time[nkk].append((flat_list))
                    plt.figure()
                    for lne in range(len(Spike_time)):
                        if Spike_time[lne]:
                            plt.vlines(np.array(Spike_time[lne]),ymin=lne,  ymax=lne+1)
            i0=i0+sdt.shape[1]


            


        # if id_peaks_all and 'reward' in filenames[idt] and plot_sequenc_rate:    
        #     #binlen=np.ceil(id_peaks_all[-1]/10)# # of samples=> 20*50ms=1sec
        #     binlen=200# # of samples=> 20*50ms=1sec

        #     print(binlen)
        #     stppoint= int(np.ceil((id_peaks_all[-1]/binlen))*binlen)
        #     allen= np.linspace(binlen,stppoint,int(stppoint/binlen))
        #     rate_wise=[]
        #     for alleni in allen:# compute binned sequence rate 
                
        #         rate_wise.append(len([x for x in id_peaks_all if ((x >(alleni-binlen)) & (x <alleni)) ])/binlen)
        #     plt.figure()
        #     plt.plot(allen/fs,rate_wise, linewidth=5)
        #     plt.xlabel('time s')                            
        #     plt.ylabel('Rate Hz')

        #     plt.title('Binned sequence rate '+fol+'_'+filenames[idt]+'_'+ signal_type+'_binsize='+str(binlen))
        #     plt.savefig(savefolder+'Binned sequence rate '+fol+'_'+filenames[idt]+'_'+ signal_type+'.png')



            
            
            
            
            
            poprate.extend(poprate_s)
            id_peaks_trl.extend(np.array(id_peaks_trl_s)+lsdt)
            spike_time.extend(spike_time_s)  
            bursts_tmp.extend(bursts_tmp_s)
            seqs_tmp.extend(seqs_tmp_s)
            lsdt = lsdt+sdt.shape[1]

        if len(spike_time_all)>0:
            spike_time_mrg = merge_spike_times(spike_time_all,ncells)
        else:
            spike_time_mrg=[]
        
    return poprate,id_peaks_trl,bursts_tmp,seqs_tmp,spike_time_mrg,Rasters

from scipy.signal import find_peaks

def popbursts_new(mat,sig,fs,kwidth=0,minwidth=1):
    global Tspare

    #poprate=np.sum(mat,axis=0)
    if kwidth<1/(3*fs):
        kwidth=1/(3*fs)
        poprate=np.sum(mat,axis=0)
    else:
        tax=np.arange(-3*kwidth,3*kwidth,1/fs)

        poprate=np.convolve(np.sum(mat,axis=0),np.exp(-(tax**2)/2/kwidth**2)/kwidth,'same')

    thresh=np.mean(poprate)+sig*np.std(poprate)
    spare=int(Tspare*fs)


    vec=[]

    idpeaks, _ = find_peaks(poprate, height=thresh, width=(minwidth,spare*10), distance=spare)

    peaks=[]
    idprev=-1
    for idpeak in idpeaks:
        if (idpeak-idprev>spare)*(idpeak-int(spare/2)>=0)*(idpeak+int(spare/2)+1<=mat.shape[1]):
            vtmp=mat[:,idpeak-int(spare/2):idpeak+int(spare/2)+1]
            if len(np.where(np.sum(vtmp,axis=1)>0)[0])>4:
                vec.append(vtmp)
                peaks.append(idpeak)
                idprev=idpeak


    if len(vec)>0:
        itax=np.arange(vec[0].shape[1])

    spike_time=[[] for kk in range(np.shape(mat)[0]) ]

    raster2 = np.zeros((np.shape(mat)[0], len(vec)))
    seq=[]
    for nv in range(len(vec)):
        nvec=np.sum(vec[nv],axis=1)
        cofseq=(itax@vec[nv].transpose())/nvec
        tmp=np.argsort(cofseq)
        seq.append(tmp[~np.isnan(np.sort(cofseq))])


        for kk in seq[nv]:
            spike_time[kk].append(idpeaks[nv])

        raster2[seq[nv],nv] = 1

    return vec,seq,idpeaks,raster2,spike_time,peaks,poprate








def merge_spike_times(spike_time_all,ncells):
    spike_time_mrg=[[] for kk in range(ncells) ]

    for y in range(ncells):#neuron
        spike_time_tmp=[]
        for x in spike_time_all:# twin
            spike_time_tmp.extend(x[y])
        spike_time_mrg[y]=spike_time_tmp
    return(spike_time_mrg)

def concat_spike_times(spike_time_all,spike_times,i0):

    if len([x for x in spike_times if len(x)>0])>0:# concat spiketime of all jumps and add the time of each jump
        spike_time_add=[]
        for spkt in spike_times:
            if spkt:
                spike_time_add.append([i+i0 for i in spkt])# each cell
            else:
                spike_time_add.append([])

        spike_time_all.append(spike_time_add)
    return(spike_time_all)

def plot_raster(Spike_time,datainfo):  
    ncells=np.shape(Spike_time)[0]
    for lne in range(len(Spike_time)):
        plt.vlines(Spike_time[lne],ymin=lne,  ymax=lne+1)
    plt.ylim([0,ncells])
    plt.title(datainfo)

def plot_rasterplots(spike_time_all_merged,datainfo,Rasters,dt_all_trl,ids_clust,fs,savefolder):
    ncells=np.shape(spike_time_all_merged)[0] 
    spike_time_all=np.transpose(spike_time_all_merged)   
                        
    Spike_time=[[] for kk in range(ncells)]

    for nkk in range(ncells):                                                        
        my_list=[i for i in spike_time_all[nkk] if i]
        flat_list = [num for num in my_list]                          
        Spike_time[nkk].append((flat_list))

    


    Spike_time_raster=[]
    Rasters_sort=np.array(Rasters)[np.argsort(ids_clust).astype(int)]
    Rasters_sort=np.transpose(Rasters_sort)
    Rasters=np.transpose(Rasters)
    for clidx in Rasters:
        Spike_time_raster.append(np.where(clidx))
    Spike_time_raster_sorted=[]    
    for clidx in Rasters_sort:
        Spike_time_raster_sorted.append(np.where(clidx))

    #plt.rcParams.update({'font.size': 20})

    t_raw_data=np.transpose(dt_all_trl)
    n_raw_data=[x/np.max(x) for x in dt_all_trl]
    ticks = np.linspace(0,np.shape(n_raw_data)[0],4)
    ticklabels = ["{:0.0f}".format(i) for i in ticks/fs]
    with plt.rc_context({'font.size': 20}):
        Spike_time_t =[np.array(xx)/fs for x in Spike_time for xx in x ]
        plt.figure(figsize=(20,10))
        plt.suptitle(datainfo+'loc corrected')
        plt.subplot(142) 
        plot_raster(Spike_time_t,datainfo='Detected bursts')
        plt.xlabel('time s')
        plt.yticks([])
        #plt.xticks(ticks,ticklabels)
        plt.subplot(143) 
        plot_raster(Spike_time_raster,datainfo='Unsorted sequences')
        plt.yticks([])
        plt.xlabel('seq#')
        plt.subplot(144) 
        plot_raster(Spike_time_raster_sorted,datainfo='Sorted sequences')
        plt.vlines(np.where(np.diff(np.sort(ids_clust))),ymin=0,ymax=np.shape(Spike_time_raster_sorted)[0],colors='red')
        plt.xlabel('seq#')
        plt.yticks([])
        plt.subplot(141) 


        plt.imshow(np.transpose(n_raw_data) ,origin = 'lower',aspect='auto',cmap='Blues',interpolation='None')  
        plt.xticks(ticks,ticklabels)
        plt.ylim([0,ncells])
        plt.xlabel('time s')
        plt.ylabel('cell#')
        plt.title('Raw data')
        plt.savefig(savefolder+'New Spikeraster loc corrected'+datainfo+'_'+'.png')
        plt.savefig(savefolder+'Neew Spikeraster loc corrected'+datainfo+'_'+'.svg')








def plot_speed_location_alltrials(Behav_all_sess,titleax,savefolder,Figx):
    ax6x = Figx['ax1']
    ax7x = Figx['ax2']

    ax8x = Figx['ax3']
    ax9x = Figx['ax4']
    
    with plt.rc_context({'font.size': 20}):# plot all data


        lspd=0
        Speed_sess_flat=[]
        Id_sess_flat=[]
        lidpeak=0
        lidpeak_all=[]

        for kk in np.arange(len(Behav_all_sess['idpeaks'])):# data of all trials all sessions
            #ax5x.plot(x_loc_sess_all_trl[kk],y_loc_sess_all_trl[kk], '.',color='grey',alpha=.1)
            #ax5x.plot3D(x_loc_sess_all_trl[kk],y_loc_sess_all_trl[kk],speed_sess_all_trl[kk],'--',linewidth=1,alpha=.4,color='green')
            #if len(id_peaks_sess_all_trl[kk][0])>0:
            #    ax5x.plot3D(np.array(x_loc_sess_all_trl[kk])[f2(id_peaks_sess_all_trl[kk][0])],np.array(y_loc_sess_all_trl[kk])[f2(id_peaks_sess_all_trl[kk][0])],np.ones(len(id_peaks_sess_all_trl[kk][0]))*np.array(speed_sess_all_trl[kk])[f2(id_peaks_sess_all_trl[kk][0])],"X",color='k',markersize=5)
            
            
            if (len(Behav_all_sess['idpeaks'][kk][0])>0) & ('inward' in (Behav_all_sess['idpeaks'][kk][1])):
                ax8x.plot(np.array(Behav_all_sess['xloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],np.array(Behav_all_sess['yloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],"o",color='red',markersize=5)
                ax8x.plot(Behav_all_sess['xloc'][kk],Behav_all_sess['yloc'][kk], '.',color='grey',alpha=.1)
            if (len(Behav_all_sess['idpeaks'][kk][0])>0) & ('outward' in (Behav_all_sess['idpeaks'][kk][1])):
                ax9x.plot(np.array(Behav_all_sess['xloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],np.array(Behav_all_sess['yloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],"o",color='red',markersize=5)
                ax9x.plot(Behav_all_sess['xloc'][kk],Behav_all_sess['yloc'][kk], '.',color='silver',alpha=.1)

            # ax3.hist(all_evnt_speed,bins=15)
            # ax3.set_xlabel('speed')
            # ax3.set_ylabel('histogram')
            if (len(Behav_all_sess['idpeaks'][kk][0])>0) & ('inward' in (Behav_all_sess['idpeaks'][kk][1])):
                ax6x.plot(Behav_all_sess['xloc'][kk],Behav_all_sess['yloc'][kk], '.',color='grey',alpha=.1)
                #ax6x.plot3D(Behav_all_sess['xloc'][kk],Behav_all_sess['yloc'][kk],Behav_all_sess['speed'][kk],'--',linewidth=1,alpha=.4,color='green')
                ax6x.plot3D(np.array(Behav_all_sess['xloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],np.array(Behav_all_sess['yloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],np.ones(len(Behav_all_sess['idpeaks'][kk][0]))*np.array(Behav_all_sess['speed'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],"X",color='k',markersize=5)

            if (len(Behav_all_sess['idpeaks'][kk][0])>0) & ('outward' in (Behav_all_sess['idpeaks'][kk][1])):
                ax7x.plot(Behav_all_sess['xloc'][kk],Behav_all_sess['yloc'][kk], '.',color='grey',alpha=.1)
                #ax7x.plot3D(Behav_all_sess['xloc'][kk],Behav_all_sess['yloc'][kk],Behav_all_sess['speed'][kk],'--',linewidth=1,alpha=.4,color='green')
                ax7x.plot3D(np.array(Behav_all_sess['xloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],np.array(Behav_all_sess['yloc'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],np.ones(len(Behav_all_sess['idpeaks'][kk][0]))*np.array(Behav_all_sess['speed'][kk])[f2(Behav_all_sess['idpeaks'][kk][0])],"X",color='k',markersize=5)


        Speed_sess_flat.extend(Behav_all_sess['speed'][kk]+lspd)
        Id_sess_flat.extend(Behav_all_sess['idpeaks'][kk][0]+lidpeak)
        lidpeak=len(Behav_all_sess['speed'][kk])+lidpeak
        lspd=np.max(Behav_all_sess['speed'][kk])+lspd
        lidpeak_all.append(lspd)
    ax6x.view_init(elev=5, azim=-10) 
    ax7x.view_init(elev=5, azim=-10) 
    ax8x.set_title('inward')
    ax9x.set_title('outward')
    ax6x.set_title('inward')
    ax7x.set_title('outward')
    ax8x.set_title('inward '+' All '+titleax)

    plt.savefig(savefolder+'All speed'+titleax+'.png')





# def extract_data2(spk,data,thr_burts,fs,condition,remove_failed_trials,): # extract the specified index from selected trials and returns it
    
#     class sessionsdata():
#         def __init__(self, name, data,loc,trlidx,trial_data,id_peaks):
#             self.name = name
#             self.data = data
#             self.loc = loc
#             self.trlidx = trlidx
#             self.trial_data = trial_data
#             self.id_peaks = id_peaks


#     # fs=fs
#     # fsthr_burts=fsthr_burts
#     winlen=fs*10# 10 second
#     poprate,id_peaks,bursts,seqs,spike_times,rasters = binned_burst(spk,winlen,thr_burts,fs,)


#     selected_trials = np.array(data['EvtT'][condition])      
#     correct_trials = np.array(selected_trials.reshape(int(len(selected_trials)/2), 2))
    
#     if remove_failed_trials=='True':
        
#         failed = np.array([x in (data['EvtT']['failed_trials'][0::2]) for x in data['EvtT'][condition][0::2]])
#         correct_trials= correct_trials[~failed]
        
#     trial_data=[]
#     extract=[]
#     extract_xloc=[]
#     extract_yloc=[]
#     for x in correct_trials:
#         trial_data.append(np.array(spk[:,x[0]:x[1]]))# valid trials
#         extract.extend(np.transpose(np.array(spk[:,x[0]:x[1]])))# valid trials
#         extract_xloc.extend(data['EvtT']['x'][x[0]:x[1]])
#         extract_yloc.extend(data['EvtT']['y'][x[0]:x[1]])

#     extract=np.array(np.transpose(extract)) 

#     #learn_data[condition] = extract # put data from selected trials

#     xloc= np.expand_dims(np.array(data['EvtT']['x']), axis=0)
#     yloc= np.expand_dims(np.array(data['EvtT']['y']), axis=0)
#     EvtT2 = np.concatenate((np.expand_dims(np.array(extract_xloc), axis=0),np.expand_dims(np.array(extract_yloc), axis=0)))
    


#     learn_data =   sessionsdata(condition,extract,EvtT2,correct_trials,trial_data,id_peaks)


#     return learn_data




def extract_data2(spk,data,class_trails,cond_name,winlen,thr_burts,fs,speeds,min_ratio,clstr_param,nrm,filename,session_mask,condition,remove_failed_trials): # extract the specified index from selected trials and returns it
    
    t = np.arange(spk.shape[1])/fs
    #winlen=fs*10# 10 second
    poprate,id_peaks,bursts,seqs,spike_times,rasters = binned_burst(spk,winlen,thr_burts,fs,timewins=[])

    Spike_times = np.empty(np.shape(spk)[0], dtype=float).tolist()# spike time of individual cells for place field analysis
    for clid in range(np.shape(spk)[0]):
        dt=spk[clid,:]
        dt2 = np.expand_dims(dt, axis=0)
        timewins=f2(np.ceil(np.linspace(0,len(dt),int(np.ceil(len(dt)/winlen)))))
        poprate_ss,id_peaks_ss,bursts_ss,seqs_ss,pike_times_ss,rasters_ss = binned_burst(dt2,winlen,thr_burts,fs,timewins=timewins)
        Spike_times[clid]=pike_times_ss[0]
        #poprate_cluster,id_peaks_cluster,bursts_cluster,seqs_cluster,spike_times_cluster,rasters_cluster =  mot.binned_burst(dt2,winlen,thr_burts,fs,timewins=timewins)




    repid,nsig,pval,bmat,zmat,corrmat = allmot(seqs,nrm);
    ids_clust = cluster(bmat,zmat,clstr_param)
    #temp_info = templates(bursts,seqs,nrm,ids_clust,min_ratio = min_ratio)
    ncells=spk.shape[0]
    # tbursts= mot.t_from_mat(list(bursts),fs)
    # dummy,templateseq = mot.average_sequence(bursts)
    # sort=np.argsort(templateseq/fs)
    
    # linsp=np.arange(1,ncells+1)
    # plt.plot(templateseq[sort],linsp, '.k')
    # plt.show
    # plot_sequence_time=False
    # if 0:
    #     figs=plt.figure(figsize=(100,5))
    #     Seqs=seqs[0:10]
    #     # for ix,seq in enumerate(seqs):
    #     #     a=np.arange(len(seq))/len(seq)+id_peaks[ix]
    #     #     plt.scatter(a,seq,marker='|')
    #     plt.plot((poprate/np.max(poprate)*spk.shape[0]),alpha=.5)
    #     plt.vlines(id_peaks,0,spk.shape[0],alpha=.8)

    # if 0:
    #     seqmat=np.zeros((ncells,spk.shape[1]))
    #     for ix in range(len(seqs)):
    #         tx=seqs[ix]
    #         ty=id_peaks[ix]+np.arange(0,len(seqs[ix]))
    #         seqmat[tx,ty] = 1
    #     plt.figure(figsize=(100,1))
    #     plt.imshow(seqmat,vmin =0, vmax=.01, cmap='binary',aspect='auto')
    #     #plt.plot(poprate)

    # if 0:
    #     #temp_info['duration'] = np.shape(lentrials)[0]
    #     #temp_info['idpeaks'] = id_peaks_all
    #     seqAll = copy.copy(seqs)
    #     seqerate=[]
    #     sequenccount = []
    #     checks=[]
    #     for nc in range(len(temp_info['radius'])):    
    #         seqAll.append(temp_info['template'][nc])
    #         chck = mot.check_template(seqs,temp_info['template'][nc],nrm)
    #         checks.append(chck[1])

    #         #seqerate.append(fs*np.sum(chck[1])/temp_info['duration'])
    #         sequenccount.append(np.sum(chck[1]))
    #     temp_info['sequencerate'] = np.sum(checks,axis=0)>0
    #     temp_info['sequenccount'] = np.mean(sequenccount)

    #     if plot_sequence_time ==True:
    #         templates=[]
    #         tsamples=[]
    #         dummies=[]
    #         for nt in range(len(temp_info['clist'])):

    #             if len(np.where(np.array(temp_info['exclude'])==nt)[0])>0:
    #                 continue

                
    #             clist=temp_info['clist'][nt]
    #             print(nt,len(clist))
    #             samples = np.array(bursts)[clist,:,:]
    #             dummy,template = mot.average_sequence(samples)
    #             templates.append(template/fs)
    #             dummies.append(np.mean(samples,axis=0))

    #             samples=np.array(bursts)[clist,:,:]
    #             tsamples.append(mot.t_from_mat(list(samples),fs))
                
    #         fig=plt.figure(figsize=(10,10))
    #         dy=0.15        
    #         for nt in range(len(templates)):
    #             clist=temp_info['clist'][nt]

    #             ax=fig.add_axes([0.1,0.85-nt*dy,0.8,0.1])
    #             if nt==0:
    #                 ax.set_title(filename[:-4])
    #             isort=np.argsort(templates[nt])
    #             ncells=len(templates[0])
    #             linsp=np.arange(1,ncells+1)
    #             mot.plot_samples(templates,isort,tsamples[nt][0:10],ax)




    if 0:
        plt.figure(figsize=(100,5))
        plt.vlines(id_peaks,np.min(poprate),np.max(poprate),color='k')
        plt.plot(poprate)
        plt.title(cond_name)

        plt.figure(figsize=(7,3))
        plt.subplot(122)
        plt.hist(speeds[id_peaks], histtype="step",alpha=.9,density=True)
        plt.hist(speeds,alpha=.9, histtype="step",density=True)
        plt.legend(['all speeds','event speeds'])
        plt.xlabel('speed cm/sec')





    tftrials=np.ones(spk.shape[1],dtype=bool)# by default sleep and arena are true trials
    task_nbr=np.ones(spk.shape[1])# assing on number to each task,.e.g. arean=0
    if 'arena' in cond_name:
        task_nbr=task_nbr*0
    elif 'learning' in cond_name:
        task_nbr=task_nbr*1
    elif 'learned' in cond_name:
        task_nbr=task_nbr*2




    selected_trials = np.array(data['EvtT'][condition])      
    correct_trials = np.array(selected_trials.reshape(int(len(selected_trials)/2), 2))



    if ('task_learning' in cond_name) | ('task_learned' in cond_name):
        cond_trials = np.array(selected_trials.reshape(int(len(selected_trials)/2), 2))

        selected_corrects = np.array(data['EvtT']['correct_trials']) 
        correct_trials = np.array(selected_corrects.reshape(int(len(selected_corrects)/2), 2))

        selected_false = np.array(data['EvtT']['failed_trials']) 
        false_trials = np.array(selected_false.reshape(int(len(selected_false)/2), 2))
        

                                    
        correxindex= find_correct_index(cond_trials,correct_trials)
        fasleindex= find_correct_index(cond_trials,false_trials)


        if remove_failed_trials=='Correct':

            correct_trials = cond_trials[correxindex]
            print(cond_name+' Correct trials number=', len(correct_trials))

        elif remove_failed_trials=='Failed':
            correct_trials = cond_trials[fasleindex]

            print(cond_name+' False trials number=', len(correct_trials))
        elif remove_failed_trials=='All':
            correct_trials=cond_trials


        if np.shape(false_trials)[0]>0:
            for x in cond_trials[fasleindex]:
                if x[1]>x[0]:# to remove the observed inconsistency of 'task_learned_outward_L_center_20220320'
                    tftrials[x[0]:x[1]]=False# set index of false trials



    # selected_trials = np.array(data['EvtT'][condition])      
    # correct_trials = np.array(selected_trials.reshape(int(len(selected_trials)/2), 2))

    # if ('task_learning' in cond_name) | ('task_learned' in cond_name):
    #     failed = np.array([x in (data['EvtT']['failed_trials'][0::2]) for x in data['EvtT'][condition][0::2]])

    #     if remove_failed_trials==True:
            
    #         correct_trials= correct_trials[~failed]# use only correct trials
    #         print(cond_name+' Correct trials number=', len(correct_trials))

    #     elif remove_failed_trials==False:
    #         print(cond_name+' False trials number=', sum(failed))
    #         correct_trials= correct_trials[failed]# use only false trials
    # # if 'arena' in cond_name:
    # #     correct_trials = np.array(selected_trials.reshape(int(len(selected_trials)/2), 2))


    seq_mask_sessins=[]
    true_false_trial=[]
    t_trials=[]  
    t_trial=[]  
    trial_data=[]
    extract=[]
    extract_xloc=[]
    extract_yloc=[]
    speed=[]
    ids_clust_trials=[]
    poprate_trials=[]
    seqs_trials=[]
    id_peaks_trials=[]
    id_peaks_cell=[]
    i0=0
    id_peaks=np.array(id_peaks)
    bursts=np.asarray(bursts)
    bursts_trials=[]
    Spike_times=np.array(Spike_times)
    seqs=np.array(seqs)
    ids_clust=np.array(ids_clust)
    id_peaks_cell=[[] for kk in range(ncells) ]
    all_tasks_nbr=[]
    Spike_times_mask_all_cells=[]
    fr=[]
    fr_singlecell=[]
    for x in correct_trials:
        if x[1]>x[0]:# to remove the observed inconsistency of 'task_learned_outward_L_center_20220320'
            x[1]=x[1]+1# not sure if this is correct!!!
            all_tasks_nbr.extend(task_nbr[x[0]:x[1]])# assing a number to each task
            trial_data.append(np.array(spk[:,x[0]:x[1]]))# valid trials
            extract.extend(np.transpose(np.array(spk[:,x[0]:x[1]])))# valid trials
            true_false_trial.extend(tftrials[x[0]:x[1]])
            extract_xloc.extend(data['EvtT']['x'][x[0]:x[1]])
            extract_yloc.extend(data['EvtT']['y'][x[0]:x[1]])
            speed.extend(speeds[x[0]:x[1]])
            t_trials.extend(t[x[0]:x[1]])
            #t_trial.extend((np.arrange(x[1]-x[0]))/fs)
            poprate_trials.extend(poprate[x[0]:x[1]])
            mask=(id_peaks>=x[0]) & (id_peaks<x[1])# keep those sequences that are inside the trials
            #if len(id_peaks[mask]>0):
            idtril = id_peaks[mask]-x[0]+i0
            id_peaks_trials.extend(idtril)
            seqs_trials.extend(seqs[mask])
            #ids_clust_trials.extend(ids_clust[mask])
            bursts_trials.extend(bursts[mask])
            fr.append(sum(mask)/(x[1]-x[0]))
            seq_mask_sessins.extend(session_mask[filename]* np.ones(len(seqs[mask])))# assign a number to each sequence that shows to which session it belongs


            trial_spikes_allcells=[np.asarray(xx)[(xx>=x[0]) & (xx<x[1])]/fs for xx in Spike_times ]
            Spike_times_mask_all_cells.append(trial_spikes_allcells)
            trl_len=(x[1]-x[0])
            fr_singlecell.append([len(x)/trl_len for x in trial_spikes_allcells])
           # id_peaks_cell = [[pair,  ext] for pair, ext in zip(id_peaks_cell, np.asarray(Spike_times_mask_all_cells))]
            
            i0 = x[1]-x[0]+i0# add length of current trial to the end of next trial
        #print('x',x, id_peaks[mask],'  correctedids',idtril,'  speed',len(speed))

    id_peaks_trials_cells= merge_spike_times(Spike_times_mask_all_cells,ncells)


    extract=np.array(np.transpose(extract)) 

    EvtT2 = np.concatenate((np.expand_dims(np.array(extract_xloc), axis=0),np.expand_dims(np.array(extract_yloc), axis=0)))
    learn_data =   class_trails(cond_name,extract,EvtT2,correct_trials,trial_data,np.array(id_peaks_trials),speed,np.array(seqs_trials),np.array(t_trials),poprate_trials,ids_clust_trials,true_false_trial,all_tasks_nbr,t,bursts_trials,id_peaks_trials_cells,fr,fr_singlecell,seq_mask_sessins)

    correct_trials_session = cond_trials[correxindex]
    return learn_data,correct_trials_session,Spike_times




















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


def calculate_shannon_entropy(data):
    total_count = data['seq_len']
    entropy = 0.0
    
    for count in data['clsuster_counts']:
        if count > 0:
            proportion = count / total_count
            entropy -= proportion * np.log(proportion)
    
    return entropy



def calculate_diversity_q(data,q):
    total_count = data['seq_len']
    index = 0.0
    
    for count in data['clsuster_counts']:
        proportion = count / total_count
        if count != 0:
            index += (proportion)** q 
    
    div_index=(index)**(1/(1-q))
    
    return div_index

def plot_diversity_index(seq_rate_all,cond,xq):
    #max_va=np.max([np.max(x) for x in np.max(seq_rate_all.values())])
    max_va=11
    if 'rate' in cond:
        fs=20
    plt.figure()    
    for mode in ['learning','learned']:
       
        epochs=['inward','outward','sampling','reward']
        #epochs=['sampling','reward']
        #epochs=['sampling','inward']

        rates={}
        for tsk in epochs:
            task= mode+'_'+tsk
            seq_rate_mean=np.mean([seq_rate_all[x] for x in seq_rate_all.keys()  if task in x],0)
            seq_rate_std=np.std([seq_rate_all[x] for x in seq_rate_all.keys()  if task in x],0)


            rates[tsk]={'mean':seq_rate_mean ,'std':seq_rate_std}
            color='b'
            linestyle='solid'
            if mode == 'learning':
                
                linestyle='dashed'

            if 'inward' in task:
                color='r'
            if 'outward' in task:
                color='g'
            if 'sampling' in task:
                color='b'  
            if 'reward' in task:
                color='k'  
            #plt.plot(rates.keys() ,[rates[x] for x in rates.keys()],color=color)
        
        #plt.errorbar(rates.keys(), [rates[x]['mean'] for x in rates.keys()],yerr=[rates[x]['std'] for x in rates.keys()],label=mode, color=color, marker='s', linestyle='dashed',
        #linewidth=3, markersize=14,alpha=.8)
                
            
            plt.plot(xq, (rates[tsk]['mean']), linewidth=2, label=task,color=color,linestyle=linestyle)
            plt.fill_between(xq, (rates[tsk]['mean']) - (rates[tsk]['std']), (rates[tsk]['mean']) + (rates[tsk]['std']), alpha=0.1)
        plt.legend()
        plt.title(cond)
        plt.xlabel('q')
        plt.ylim([0,max_va])
        
        #if 'index' in cond:
        plt.ylabel('DI')
        #plt.plot(rates.keys(), [rates[x] for x in rates.keys()],label=mode, color=color, marker='s', linestyle='dashed',
        #linewidth=3, markersize=14)
        plt.grid()
        #plt.xticks(rotation=90)



        
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



def LDA_seqrate(seqrates,labels,cod):
    X_train, X_test, y_train, y_test = train_test_split(seqrates, labels, test_size=0.2, random_state=42)

    # Create an instance of the LDA model
    lda = LinearDiscriminantAnalysis()

    # Fit the model to the training data
    lda.fit(X_train, y_train)

    # Predict the class labels for the test set
    y_pred = lda.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = balanced_accuracy_score(y_test, y_pred)
    print("Accuracy: "+cod, accuracy)
    return accuracy



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
        if len(y_train)<3:
            
            continue
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



# def plot_seqs(ti,vec,fs):
#     # thsi is for plotting
#     fig=plt.figure(figsize=(14,8))
#     templates={}
#     tsamples=[]
#     for nt in range(len(ti['clist'])):

#         # if len(np.where(np.array(ti['exclude'])==nt)[0])>0:
#         #     continue

#         print(nt)
#         clist=ti['clist'][nt]
#         samples=np.array(vec)[clist,:,:]
#         dummy,template = average_sequence(samples)
#         templates[nt]=template/fs

#         samples=np.array(vec)[clist,:,:]
#         tsamples.append(t_from_mat(list(samples),fs))

#     dy=0.2
#     ctr=0
#     for nt in templates.keys():
#         ax=fig.add_axes([0.1,0.85-ctr*dy,0.8,0.1])
#         ctr += 1
#         isort=np.argsort(templates[nt])
#         print(nt,np.array(templates[nt]).shape)
#         #mot.plot_samples(templates,isort,tsamples[nt][0:10],ax,1,nc=int(nt))
#         plot_samples2(templates,isort,tsamples[nt][0:10],ax,1,nc=int(nt))

#     plt.show(block=0)




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



def equal_trials(Masks,trial_type):
    # Get the number of equal number of trials for learning and learned
    #trial_type=0
    random.seed(42)
    if trial_type==1:
        nameoftype='correct'
    elif trial_type==0:
        nameoftype='failed'
        
    mask_poss=['','_fr','_burst']# for differnt mask types
    itr=0
    for txt in mask_poss:
        txt2='phases'
        if '_fr' in txt:
            txt2='fr_phase'
        elif '_burst' in txt:
            txt2='bursts_phase'

        phase=0
        learning_trials_mask=(np.asarray(Masks[txt2])==phase) & (np.asarray(Masks[nameoftype+'_trial_idx_mask'+txt])!=0)


        phase=1
        learned_trials_mask=(np.asarray(Masks[txt2])==phase) & (np.asarray(Masks[nameoftype+'_trial_idx_mask'+txt])!=0)


        learning_trial_number = len(np.unique(np.asarray(Masks[nameoftype+'_trial_idx_mask'+txt])[learning_trials_mask]))
        learned_trial_number = len(np.unique(np.asarray(Masks[nameoftype+'_trial_idx_mask'+txt])[learned_trials_mask]))


        min_cl_nbr=np.min((learning_trial_number,learned_trial_number))
        max_cl_nbr=np.max((learning_trial_number,learned_trial_number))

        if itr==0:# get random trials with equait number of trials
            sampled_trial_idx = random.sample(range(1, max_cl_nbr + 1), min_cl_nbr)# get n  trials of learned data. n is the number of learnig trials.

        itr +=1
        print(sampled_trial_idx)
        Masks[nameoftype+'equal_trial'+txt]=np.zeros_like(Masks[nameoftype+'_trial_idx_mask'+txt])
        for ii in sampled_trial_idx:
        
            Masks[nameoftype+'equal_trial'+txt]+=(np.asarray(Masks[nameoftype+'_trial_idx_mask'+txt])==ii)

    return Masks
    


def diff_bar_plot(data,cond):
    if 'motif' in cond:
        txt='#'
    elif 'rate' in cond:
        txt='Hz'
    categories = list(data.keys())
    values = list(data.values())
    colors = ['red' if val < 0 else 'green' for val in values]
    max_val=np.max(np.abs(values))
    y = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(5,4))
    bars = ax.barh(y, values, color=colors,edgecolor='k')

    ax.set_xlabel('Difference '+ txt)
    #ax.set_ylabel('')
    ax.set_title(cond)
    #ax.set_xlim([-abs(max_val)-abs(max_val)/10,abs(max_val)+abs(max_val)/10])
    if 'rate' in cond:
        ax.set_xlim([-.45,.45])
    else:
        ax.set_xlim([-7,7])
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    plt.tight_layout()
    plt.vlines(0,ymin=-1,ymax=5,linestyles='dashed',colors='k')
    plt.ylim([-1,4])





def LDA_seqrate(seqrates,labels,cod):
    X_train, X_test, y_train, y_test = train_test_split(seqrates, labels, test_size=0.2, random_state=42)

    # Create an instance of the LDA model
    lda = LinearDiscriminantAnalysis()

    # Fit the model to the training data
    lda.fit(X_train, y_train)

    # Predict the class labels for the test set
    y_pred = lda.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = balanced_accuracy_score(y_test, y_pred)
    print("Accuracy: "+cod, accuracy)
    return accuracy



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

def plot_seq_rate(seq_rate_all,cond,fs):# get the sequence or motif rates and plot the average over epochs
    plt.figure()
    if 'rate' in cond:
        fs=20
    for mode in ['learning','learned']:
        
        epochs=['inward','outward','sampling','reward']

        rates={}
        for tsk in epochs:
            task= mode+'_'+tsk
            seq_rate_mean=np.mean([seq_rate_all[x] for x in seq_rate_all.keys()  if task in x])
            seq_rate_std=np.std([seq_rate_all[x] for x in seq_rate_all.keys()  if task in x])


            rates[tsk]={'mean':seq_rate_mean ,'std':seq_rate_std}
            color='r'
            if mode == 'learning':
                color='b'
                
            #plt.plot(rates.keys() ,[rates[x] for x in rates.keys()],color=color)
        
        plt.errorbar(rates.keys(), [rates[x]['mean'] for x in rates.keys()],yerr=[rates[x]['std'] for x in rates.keys()],label=mode, color=color, marker='s', linestyle='dashed',
        linewidth=3, markersize=14,alpha=.8)
                


        #plt.plot(rates.keys(), [rates[x] for x in rates.keys()],label=mode, color=color, marker='s', linestyle='dashed',
        #linewidth=3, markersize=14)

        plt.xticks(rotation=90)
    plt.title(cond)
    
    plt.ylabel('DI')
    if 'number' in cond:
        plt.ylabel('#')

    if 'index' in cond:
        plt.ylabel('')

    plt.legend()
    #ax = plt.gca()




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



def apply_masks_neworder(sess_info,Masks,cond_numbers,cond_name,sessin_numbers,odd_even,sess_name,trial_type,phase):

    run_data={'idpeaks_cells':[[] for _ in range(len(sess_info['Spike_times_cells']))]}

    if odd_even != None: 
        mask_odd= np.asarray(Masks['odd_even'])==odd_even# mask for odd or even trials(0/1)
        mask_odd_seqs= np.asarray(Masks['odd_even_seqs'])==odd_even# mask for sequences of odd or even trials(0/1)
        mask_odd_fr= np.asarray(Masks['odd_even_fr'])==odd_even

    else:
        mask_odd=np.ones(len(Masks['phases'])).astype(bool)
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
        #sess_name_=list(sess_name.keys())[sess_nbr]
        #sess_names= sess_names+ ' and '  +sess_name_[:-4]

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
        # cond_name_=list(cond_name.keys())[cond_nbr]
        # cond_names= cond_names+ ' and '  +cond_name_



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




    #run_data['trial_idx_mask']=np.asarray(sess_info['trial_idx_mask'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]

    run_data['poprate']=np.asarray(sess_info['pop_rate'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    cell_trace_sess1=np.asarray(sess_info['extract'][celid])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']] 

    run_data['trace_cells']=np.asarray([x[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']] for x in sess_info['extract']])# raw trace of all cells

    run_data['lin_pos']=np.asarray(sess_info['lin_pos'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]

    run_data['conditions']=np.asarray(Masks['conditions'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]


    run_data['x_loc']=np.asarray(sess_info['xloc'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['y_loc']=np.asarray(sess_info['yloc'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['speed']=np.asarray(sess_info['speed'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    spk_temp=np.asarray(sess_info['Spike_binary'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['spike_idx']=np.where(spk_temp)[0]


    spk_cell_tmp=sess_info['Spike_binary_cells'][:,mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]
    run_data['spike_idx_cells']=[np.where(x)[0] for x in spk_cell_tmp]




    run_data['fr']=np.asarray(sess_info['fr'])[mask_sess_fr & mask_cond_fr &  mask_correct_fr & mask_phase_fr]

    run_data['t']=np.asarray(sess_info['t'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase & Masks['speed']]

    #t_all=np.asarray(sess_info['t'])[mask_sess_t & mask_cond_t & mask_odd & mask_correct & mask_phase ]

    #spk_times=np.where(np.isin(t_all,run_data['idpeaks']))# time of spike for population rate in the determined condtions
    
    #spk_times_cell= [np.where(np.isin(t_all,x)) for x in run_data['idpeaks_cells']]

    if 0:
        plt.figure()
        plt.plot(run_data['lin_pos'])
        if len (run_data['poprate'])>0:
            plt.plot(run_data['poprate']/np.max(run_data['poprate']))

        plt.eventplot(spk_times,lineoffsets=1,color='r')
        plt.eventplot(spk_times_cell[celid],lineoffsets=2,color='k')
        plt.plot(2+cell_trace_sess1/10)


    #run_data['spike_idx']=spk_times
    #run_data['spike_idx_cells']=spk_times_cell
    run_data['sess_name']=sess_names
    run_data['phase_name']=phase_name
    run_data['cond_name']=cond_names


    return(run_data)
















def linearize_2d_track_from_skeleton(track,left1,right1,left2,right2,corr,skel):
    '''

    Parameters
    ----------
    track : 2d array
        Behavioural trackin with coordinate on first dimension (x,y) and time on the second.
    left1, right1, left2, right2 : list
        start and end points used. The first list is used to extract start points, the second one to obtain end points.
    skel : dict
        Skeleton of this mosue.
    Returns
    -------
    Linear position left and right.

    '''


    # Only consider correct trials.
    left1=separate_in(left1,corr)
    right1=separate_in(right1,corr)
    
    left2=separate_in(left2,corr)
    right2=separate_in(right2,corr)
    
    left,right=[],[]
    left.extend(left1[0::2])
    left.extend(left2[1::2])
    left=np.sort(left)
    
    right.extend(right1[0::2])
    right.extend(right2[1::2])
    right=np.sort(right)

    track_left=separate_in_2d_array(track,left)
    track_right=separate_in_2d_array(track,right)
    plt.plot(track_left[0],track_left[1], '-.',color='gray')
    plt.plot(track_right[0],track_right[1], '-.',color='blue')

    # Case left
    x_real=track_left[0]
    y_real=track_left[1]
    c=skel['skeleton left']
    total_length=skel['length left']
    lin_pos=[]
    for n in range(len(x_real)):
        first_ind=[x_real[n],y_real[n]]
        distance,index = spatial.KDTree(c).query(first_ind)
        lin_pos.append(index/total_length)
    
    lin_pos_left=lin_pos
	
    # Case right
    x_real=track_right[0]
    y_real=track_right[1]
    c=skel['skeleton right']
    total_length=skel['length right']
    lin_pos=[]
    for n in range(len(x_real)):
        first_ind=[x_real[n],y_real[n]]
        distance,index = spatial.KDTree(c).query(first_ind)
        lin_pos.append(index/total_length)
    
    lin_pos_right=lin_pos
    
    
    return lin_pos_left,lin_pos_right,left,right



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

def significance_chekc(pc_ratio,ax,first_four_colors):
        data_vetors=list(pc_ratio.values())
        statistic, p_value = stats.kruskal(*data_vetors)
        labels_names = list(pc_ratio.keys())

        labels=np.concatenate([ix*np.ones(len(pc_ratio[x]))  for ix, x in enumerate(pc_ratio.keys())])
        labels_long=np.concatenate([[x]*(len(pc_ratio[x]))  for ix, x in enumerate(pc_ratio.keys())])


        data_vetors_flat=np.concatenate(data_vetors)


        num_groups = len(pc_ratio.keys())
        for i in range(num_groups):
            offset=.51
            for j in range(i + 1, num_groups):
                if stats.ttest_ind(np.asarray(data_vetors_flat)[labels == i], np.asarray(data_vetors_flat)[labels == j], equal_var=False).pvalue < 0.05:
                    # Add significance marker
                    #print('boz',list(pc_ratio.keys())[i],list(pc_ratio.keys())[j])
                    offset=offset+.1
                    #ax.text(i, max(np.asarray(data_vetors_flat)[labels == j]) + offset, "*", ha='center', va='bottom', color=first_four_colors[j],fontsize=22)
                    #ax.text(j, max(np.asarray(data_vetors_flat)[labels == i]) + offset, "*", ha='center', va='bottom', color=first_four_colors[i],fontsize=22)


                    ax.text(i,  offset, "*", ha='center', va='bottom', color=first_four_colors[j],fontsize=22)
                    ax.text(j,  offset, "*", ha='center', va='bottom', color=first_four_colors[i],fontsize=22)



def running_average(X, nbin = 100):
    Y = np.cumsum(X, axis=0)
    Y = Y[nbin:, :] - Y[:-nbin, :]
    return Y






def plot_cluster_number(seq_rate_len,ax):
# plot the number of trials for learning and learned
    for key, value in seq_rate_len.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, list) and not sub_value:
                seq_rate_len[key][sub_key] = 0

                
    sessions = list(seq_rate_len.keys())
    correct_trials = [seq_rate_len[session]['correct'] for session in sessions]
    failed_trials = [seq_rate_len[session]['failed'] for session in sessions]

    #fig, ax = plt.subplots()
    bar_width = 0.35
    index = range(len(sessions))

    bar1 = ax.bar(index, correct_trials, bar_width, label='Correct Trials')
    bar2 = ax.bar([i + bar_width for i in index], failed_trials, bar_width, label='Failed Trials')

    ax.set_xlabel('Sessions')
    ax.set_ylabel('Number of Trials')
    #ax.set_title('Length of Correct and Failed Trials for Different Sessions')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    #plt.show()



import numpy as np
from scipy.ndimage import gaussian_filter1d 

# %%
def transient_mask(trace, sigma=None, nsigma=3, mindur=0.2, fps=20):
    '''
    Parameters
    ----------
    trace : np.ndarray (1D)
        Fluorescence trace.
    sigma : float
        Standard deviation of the baseline fluctuation (noise sigma).
        Provide this value using calcium.normalize().
        1 if the trace is z-scored.
    nsigma : int
        Number times the noise sigma above which signal will be considered as
        candidate transients.
    mindur : float
        Minimum transient duration in sec.
    fps : float
        Frames per second.
        
    Returns
    -------
    transient : np.ndarray (1D)
        Boolean array of transient (same length as trace).
    '''
    if sigma is None:
        sigma = np.std(trace)
        trace = trace - np.mean(trace)
        
    transient = (trace > nsigma*sigma)
    
    ## Check for minimum transient width
    T = len(trace)
    minwidth = int(mindur*fps)  # Minimum width in number of data points
    j = 0  # Index pointer
    while (j < T):
        if transient[j]:
            # Candidate transient starts, find its end point
            k = next((idx for idx, val in enumerate(transient[j:]) if not val), None)
            if k is not None:
                # Found transient[j+k] the next first False
                if (k < minwidth):
                    transient[j:j+k] = False
                j = j+k+1
            else:
                # k is None, i.e. transient is True until the end
                if (T-j) < minwidth:
                    transient[j:] = False
                j = T
        else:
            # Not transient, skip to the next data point
            j = j+1
    
    return transient

#%% 
def significant_tranients(fn,mask):
    ''' Helper function that extracts significant transients from df/f data.
        Significant transients are obtained by taking the raw value only in periods with significant transients.
        
        Input:
        
        fn: Output from analyze_miniscope_transients (appended in hdf5)
        mask: Output from analyze_miniscope_transients (appended in hdf5)
    '''
   
    sign_output=np.zeros(len(fn))
     
    for n in range(len(fn)):
        if mask[n]==True:
            sign_output[n]=fn[n]
        
    return sign_output







def average_similar_keys(data):
    """
    Averages over similar keys in a nested dictionary. Used fot plotting entropy results
    """
    averages = {}
    count = {}

    # Iterate over each key in the dictionary
    for key in data:
        for subkey in data[key]:
            if subkey not in averages:
                averages[subkey] = {}
                count[subkey] = {}

            for inner_key in data[key][subkey]:
                if inner_key not in averages[subkey]:
                    averages[subkey][inner_key] = 0
                    count[subkey][inner_key] = 0

                # Sum up the values
                averages[subkey][inner_key] += data[key][subkey][inner_key]
                count[subkey][inner_key] += 1

    # Calculate the average
    for subkey in averages:
        for inner_key in averages[subkey]:
            averages[subkey][inner_key] /= count[subkey][inner_key]

    return averages




def plot_entropy(entropy,titlename):
    nmeoftrial=['correct','failed']
    fig,ax=plt.subplots(2,1,)

    for ix, namtrl in enumerate(nmeoftrial):

        data1=entropy[namtrl]

        learning_values = [data1[key] for key in data1 if 'learning' in key]
        learned_values = [data1[key] for key in data1 if 'learned' in key]

        # Create labels for the conditions
        conditions = [key.split('_')[1] for key in data1 if 'learning' in key]

        # Plotting the bar graph
        width = 0.35
        bar1 = ax[ix].bar(conditions, learning_values, width, label='Learning')
        bar2 = ax[ix].bar([i + width for i in range(len(conditions))], learned_values, width, label='Learned')

        # Adding labels, title, and legend
        ax[ix].set_ylabel('Entropy',fontsize=16)
        ax[ix].set_title(' '+namtrl+' '+' '+titlename)
        if ix==1:
            ax[ix].legend()

        # Show the plot
    plt.tight_layout()
    plt.show()






    

def plot_sequences2(mat,seq,vec,idpeaks,sig,fs,speed_trl,poprate,xlim):
    #poprate=np.sum(mat,axis=0)
    thresh=np.mean(poprate)+sig*np.std(poprate)
    if 1:
        sorted=True
        Plot_entrire_recording=True
        sequencnmbr=11
        
        if Plot_entrire_recording:
           dta=mat
           colors='k' 
           linewidth=1 
           fsize=18    
        # elif sorted==True:
        #     dta = vec[sequencnmbr][seq[sequencnmbr]]
        #     colors='r'
        #     ttitle='sorted'
        #     linewidth=3  
        #     fsize=20  
        # else:
        #     dta= vec[sequencnmbr]
        #     colors='b'
        #     ttitle='unsorted'
        #     linewidth=3   
        #     fsize=20 
        


        #khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) if np.nansum(x)!=0]
        khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) ]
        khers=np.nan_to_num(khers, copy=True, nan=0.0)

        #clidx=[i for i, x in enumerate(dta) if np.nansum(x)!=0]
        clidx=[i for i, x in enumerate(dta) ]

        ticks = np.linspace(0,np.shape(khers)[1]-1,5)
       
        ticklabels = ["{:0.0f}".format(i) for i in ticks/fs]

        palette = sns.color_palette("hls", np.shape(khers)[-1])
        with plt.rc_context({'font.size': fsize}):
            fig=plt.figure(figsize=(30,10))
            if Plot_entrire_recording:
                ax0=fig.add_axes([0.1,0.1,0.4,.7])

                ax=fig.add_axes([0.1,.8,0.4,.1])
                ax.vlines(idpeaks,ymin=-1,ymax=max(poprate+10),linewidth=1,color='k',alpha=.8)
                ax.vlines(idpeaks,ymin=0,ymax=max(poprate+10),linewidth=10,color='g',alpha=.3)
                ax.plot(poprate,linewidth=5)
                ax.hlines(thresh,xmin=0,xmax=len(poprate),color='r',linestyles= 'dashed')
                
                ax.set_ylabel('ΔF/F0')
                #ax.set_title('Detected bursts of active cells')
                ax.set_title('Detected bursts')

                ax.set_xticks([])
                ax.set_yticks([])


                ax0.set_ylim(0,len(clidx))

                # ax3=fig.add_axes([0.1,.9,0.4,.1])
                # ax3.plot(speed_trl,linewidth=4,color='green')
                # plt.xlim([0,len(poprate)])
                # ax3.set_ylabel('Speed Cm/sec')

                ax0.vlines(idpeaks,ymin=0,ymax=len(clidx),linewidth=1,color='k',alpha=.8)
                ax0.vlines(idpeaks,ymin=0,ymax=len(clidx),linewidth=10,color='g',alpha=.3)
                i=0
                for kk,ii in enumerate(khers):
                    ax0.plot(ii+i,color='k', linewidth=linewidth)

                    i=i+1



            # if sorted == True:
            #     plt.yticks(np.arange(len(seq[sequencnmbr])),seq[sequencnmbr])
            # else:
            #     plt.yticks(np.arange(len(clidx)),clidx)
            ax0.set_xticks(ticks,ticklabels)
            ax0.set_xlim(xlim)
            ax.set_xlim(xlim)
            ax0.set_xlabel('time s')
            ax0.set_ylabel('Cell #')
            #ax2.set_title(ttitle) 


            



            with plt.rc_context({'font.size': fsize}):
                fig2=plt.figure(figsize=(5,10))
                ax1=fig2.add_axes([0.1,0.1,0.4,.7])
                ax2=fig2.add_axes([0.65,0.1,0.4,.7])

                dta = vec[sequencnmbr][seq[sequencnmbr]]
                colors='r'
                ttitle='sorted'
                linewidth=3  
                fsize=20  
                khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) if np.nansum(x)!=0]
                clidx=[i for i, x in enumerate(dta) if np.nansum(x)!=0]

                i=0
                for kk,ii in enumerate(khers):
                        ax2.plot(ii+i,color=colors, linewidth=linewidth)
                        i=i+1

                ax2.set_title(ttitle)
                ax2.set_yticks(np.arange(len(seq[sequencnmbr])),seq[sequencnmbr])



                dta= vec[sequencnmbr]
                colors='b'
                ttitle='unsorted'
                khers=[(x-np.min(x))/(np.max(x)-np.min(x)) for i, x in enumerate(dta) if np.nansum(x)!=0]
                clidx=[i for i, x in enumerate(dta) if np.nansum(x)!=0]
                i=0
                for kk,ii in enumerate(khers):
                        ax1.plot(ii+i,color=colors, linewidth=linewidth)
                        i=i+1

                ax1.set_title(ttitle)
                ax1.set_yticks(np.arange(len(seq[sequencnmbr])),np.sort(seq[sequencnmbr]))
                
                ticks = np.linspace(0,np.shape(khers)[1]-1,3)
                ticklabels = ["{:0.2f}".format(i) for i in ticks/fs]
                ax1.set_xticks(ticks,ticklabels,fontsize=14)
                ax1.set_xlabel('time s')
                ax1.set_ylabel('Cell index')

                ax2.set_xticks(ticks,ticklabels,fontsize=14)
                ax2.set_xlabel('time s')

    return ax, ax0, ax1, ax2

    # winlen=np.shape(STMx2)[1]-1
    # poprate,id_peaks,bursts,seqs= mot.binned_burst(STMx2,winlen,thr_burts,fs,timewins=[])

    # xlim=[9330,11500]
    # speed_trl=[]
    # plot_sequences(STMx2,seqs,bursts,id_peaks,thr_burts,fs,speed_trl,np.asarray(poprate),xlim)
    #plt.xlim([10,111])

def diversiyt_index_subsampling(temp_info_subsampling):

        
    cond_numbers={}
    cond_numbers['outward']=[2,3] # conditon name (outwards)
    cond_numbers['inward']=[6,7] # inwards
    cond_numbers['sampling']=[0,1] # sampling
    cond_numbers['reward'] =[4,5] # reward
    subsampling_restults={'cluster_dist':{}}

    Entropy_subsamplings={}
    SS_info={}
    for ss in range(len(temp_info_subsampling)):# subsampling for entropy

        entropy={'correct':{},'failed':{}}
        subsampling_info={'correct':{},'failed':{}}
        for trial_type in range(2):
            fr_trial={}
            
            #sessin_numbers=np.arange(len(session_mask))


            if trial_type==1:
                typename='correct'
            elif trial_type==0:
                typename='failed'


            for phase_name in range(2):
                if phase_name==1:
                    phname='learned'
                elif phase_name==0:
                    phname='learning'



    # temp_info_subsampling[1]['new_cond_mask']=new_cond_mask
    # temp_info_subsampling[1]['new_phase_mask']=new_phase_mask
    # temp_info_subsampling[1]['new_correct_failed_mask']=new_correct_failed_mask

                for cond_number_name in cond_numbers.keys():
                    ampling_mask_cond=np.zeros_like(temp_info_subsampling[ss]['new_cond_mask'])
                    for ii in cond_numbers[cond_number_name]:
                        ampling_mask_cond+=temp_info_subsampling[ss]['new_cond_mask']==ii# task epochs: samplng, reward, ...
                    l_mask=temp_info_subsampling[ss]['new_phase_mask']==phase_name# learning or learned sequences
                    c_mask=temp_info_subsampling[ss]['new_correct_failed_mask']==trial_type# correct or fialed sequences

                    final_mask=((c_mask) & (l_mask) & (ampling_mask_cond.astype(bool)) )
                    labels, counts = np.unique(temp_info_subsampling[ss]['ids_clust'][0][final_mask], return_counts=True)
                    


                    subsampling_restults['cluster_dist'][phname+'_'+cond_number_name] = {'seq_len':len(temp_info_subsampling[ss]['ids_clust'][0][final_mask]),
                                                                                'cluster_number': labels, 'clsuster_counts':counts}
            subsampling_info[typename]=subsampling_restults
            # compute diversity index for different qs
            Div_idxq={}

            for epochs in subsampling_restults['cluster_dist'].keys():
                clstr_data=subsampling_restults['cluster_dist'][epochs]
                diq=[]
                xq=np.arange(0.011,5.51,.1)
                xq[0]=0
                for q in (xq):
                    
                    diq.append( calculate_diversity_q(clstr_data,q))
                
                #Div_idxq[epochs]=diq
                Div_idxq[epochs]=calculate_shannon_entropy(clstr_data)
                entropy[typename][epochs]=Div_idxq[epochs]
        Entropy_subsamplings[ss]=entropy# entropy fo all saubsamplings
        SS_info[ss]=subsampling_info
    return Entropy_subsamplings,SS_info

def calculate_average_dynamic(data):
    average = {}
    num_entries = len(data)
    
    # Initialize average dictionary
    for category in ['correct', 'failed']:
        average[category] = {key: 0 for key in data[0][category]}
    
    # Sum up all values
    for i in range(num_entries):
        for category in ['correct', 'failed']:
            for key in data[i][category]:
                average[category][key] += data[i][category][key]
    
    # Calculate average
    for category in ['correct', 'failed']:
        for key in average[category]:
            average[category][key] /= num_entries
            
    return average


# Function to merge all values of the provided data for any number of repetitions
def merge_all_values(data):
    merged = {}
    
    # Initialize merged dictionary
    for category in ['correct', 'failed']:
        merged[category] = {key: [] for key in data[0][category]}
    
    # Merge values
    for i in data.values():
        for category in ['correct', 'failed']:
            for key, value in i[category].items():
                merged[category][key].append(value)
            
    return merged


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




def plot_sequence(temp_info,bmat_temp,zmat_temp):
    bursts=np.squeeze(temp_info['bursts'])
    plot_sequence_time=True
    if plot_sequence_time ==True:
        templates=[]
        tsamples=[]
        dummies=[]
        for nt in range(len(temp_info['ids_clust'])):
            clist=(np.where(temp_info['ids_clust']==nt)[0])
            # if len(np.where(np.array(temp_info['exclude'])==nt)[0])>0:# bad clusers are already removed in template function
            #     continue

            
            #clist = temp_info['ids_clust'][nt]
            #print(nt,len(clist))
            samples = np.array(bursts)[clist,:,:]
            dummy,template = average_sequence(samples)
            templates.append(template/fs)
            dummies.append(np.mean(samples,axis=0))

            samples=np.array(bursts)[clist,:,:]
            tsamples.append(t_from_mat(list(samples),fs))
            
        fig=plt.figure(figsize=(10,10))
        dy=0.15        
        for nt3 in range(len(templates)):
            #clist = temp_info['ids_clust'][nt3]
            clist=(np.where(temp_info['ids_clust']==nt3)[0])
            print(nt3)
            ax=fig.add_axes([0.1,0.85-nt3*dy,0.8,0.1])
            #if nt==0:
                #ax.set_title(filename[:-4])
            isort=np.argsort(templates[nt3])
            ncells=len(templates[0])
            linsp=np.arange(1,ncells+1)


            nshift=0
            for nt2 in range(len(templates)):
                if bmat_temp[nt3,nt2]:
                    ax.plot(nshift+templates[nt2][isort], linsp, 'r.')
                else:
                    ax.plot(nshift+templates[nt2][isort], linsp, 'b.') 
                ax.text(nshift,ncells+10,"%.1f" %zmat_temp[nt3,nt2])
                #ax.text(nshift,ncells+5,"%.f" %bmat_temp[nt3,nt2])

                nshift += 1








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
