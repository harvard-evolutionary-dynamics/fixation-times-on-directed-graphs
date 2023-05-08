# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23, 2018
@author: jtkadlec
"""
import itertools
import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import time


# Given an undirected graph G represented as a list of the form
# [(#nodes, #edges), (edge_1),(edge_2),..(edge_k)],
# and a relative fitness r > 1 of the mutant,
# function "solvegraph(G,r)" computes the fixation probability fp,
# (unconditional) absorption time AT, and (conditional) fixation time FT
# under uniform initialization, rounded to 6 decimal digits
#
# Example:
# Consider a complete graph on 4 vertices has 4 nodes and 6 edges.
# It is represented as
# [(4,6),(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
# If r=2 then fp=(1-1/2)/(1-1/2^4)=8/15=0.5333...
# and AT=6, FT=8.3
# The function outputs [0.533333, 6, 8.3]
def solvegraph(G,r):
    n,e=G[0] # stores the number of nodes and edges of G
    degs=g2degs(G) # a list counting how many neighbors each node has
    mat=g2mat(G,degs,r) # a transition matrix of the underlying Markov chain

    fps=fprob(mat) # a list of fp's from each single initial node
    ats=atime(mat) # a list of AT's from each single initial node
    cfts=cftime(mat) # a list of FT's from each single initial node

    results=[fps,ats,cfts]
    return [round(sum(x)/float(n),6) for x in results]
    

# auxilliary function:
# reads a vector of 0s and 1s as a number in binary
# and outputs that number
def vec2num(vec):
    if len(vec)==1:
        return vec[0]
    else:
        return vec[len(vec)-1]+ 2*vec2num(vec[:-1])


# auxilliary function:
# given an undirected unweighted graph G, computes the degrees of the respective nodes
def g2degs(G):
    (v,e)=G[0]
    degs=[0]*v
    for ed in G[1:]:
        degs[ed[0]]+=1
        degs[ed[1]]+=1
        if ed[0]==ed[1]: # accounting for self-loops
            degs[ed[0]]-=1
    return degs


# A key function that computes the transition matrix of the underlying Markov chain
# This is used to compute fp, AT, and FT from any initial node
def g2mat(G,degs,r):
    (v,e)=G[0]
    states=2**v
    Mprob=np.zeros(shape=(states,states))
    # vecst is a vector of 0s and 1s corresponding to a state
    powers=[]
    for k in range(v):
        powers.append(2**(v-1-k))

    for vecst in itertools.product((0,1),repeat=v):
        mut=[] # stores the multiplicative constants for mutants/residents
        for k in range(v):
            if vecst[k]==1:
                mut.append(r)
            else:
                mut.append(1)
        numst=vec2num(vecst)
        totfit=sum(mut)
        
        for ed in G[1:]:
            k=ed[0]
            l=ed[1]
            #print ed
            if vecst[k] != vecst[l]:
                if vecst[k]==0:   # node k spreads to node l
                    stnew=numst-powers[l]
                else:
                    stnew=numst+powers[l]
                Mprob[numst,stnew]+=mut[k]/float(totfit*degs[k])
                Mprob[numst,numst]-=mut[k]/float(totfit*degs[k])
                
                if vecst[l]==0:   # node l spreads to node k
                    stnew=numst-powers[k]
                else:
                    stnew=numst+powers[k]
                Mprob[numst,stnew]+=mut[l]/float(totfit*degs[l])
                Mprob[numst,numst]-=mut[l]/float(totfit*degs[l])
    Mprob[0,0]=1
    Mprob[states-1,states-1]=1
    return Mprob


# given a transition matrix of the Markov chain,
# computes the fixation probabilities from all possible initial states
# and returns the fprobs from each single initial node
def fprob(mat):
    size=mat.shape[1]
    rhs=[0]*(size)
    rhs[size-1]=1
    fprobsol= spsl.spsolve(sps.csr_matrix(mat),rhs)
    
    retprobs=[]
    ind=1
    while ind<size:
        retprobs.append(fprobsol[ind])
        ind=2*ind
    return retprobs[::-1] # reversing the order
    
# given a transition matrix of the Markov chain,
# computes the absorption times from all possible initial states
# and returns the atimes from each single initial node
def atime(mat):
    size=mat.shape[1]
    rhstime=[-1]*(size)
    rhstime[0]=0    
    rhstime[size-1]=0    
    ftimesol= spsl.spsolve(sps.csr_matrix(mat),rhstime)
    
    rettimes=[]
    ind=1
    while ind<size:
        rettimes.append(ftimesol[ind])
        ind=2*ind
    return rettimes[::-1] # reversing the order
    
# given a transition matrix of the Markov chain,
# computes the (conditional) fixation times from all possible initial states
# and returns the cftimes from each single initial node
def cftime(mat):
    size=mat.shape[1]
    
    rhs=[0]*(size)
    rhs[size-1]=1
    fprobsol= spsl.spsolve(sps.csr_matrix(mat),rhs)
    
    cmat=np.copy(mat)
    for i in range(size):
        for j in range(size):
            if fprobsol[i] != 0:
                cmat[i,j]=cmat[i,j]*fprobsol[j]/float(fprobsol[i])
    return atime(cmat)

# t0=time.clock()

# Example: Computes fp,AT,FT for a complete graph on 4 nodes when r=2.
# G=[(4,6),(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
# r=2
# print solvegraph(G,r)
# print 'time taken: ', time.clock()-t0
# '''
# '''