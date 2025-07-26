#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from importlib import reload
import torch
import math
import ncon_t as nt
import numpy as np
import copy
import time
from torch.utils.checkpoint import checkpoint


from svd import SVD
svd = SVD.apply

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

def bdir2pvecL(bdir):
    if bdir==0:
        pvecL = [0,1,2,3,4]
    elif bdir==1:
        pvecL = [1,2,3,0,4]
    elif bdir==2:
        pvecL = [2,3,0,1,4]
    elif bdir==3:
        pvecL = [3,0,1,2,4]
    return pvecL

def CTMRG_torch(T,chi,initsteps,tol,ini='rand',use_check_point=0,filename_txt=None,dtch_iso=0,T_prime=None,C=None,E=None):
    D=T.shape[0]
    ini_chi=chi
    dtype=T.dtype
    if T_prime is None:
        T_prime=T.conj()
    if C is None:
        C = [0 for x in range(4)]
        for bdir in range(4):
            if ini=='rand':
                #breakpoint()
                C[bdir]=torch.rand(chi,chi,dtype=dtype)
            elif ini=='symmetric':
                pvecL=bdir2pvecL(bdir)
                C[bdir] = (nt.ncon_torch([T_prime.permute(pvecL),T.permute(pvecL)],[[1,2,-3,-1,3],[1,2,-4,-2,3]])).reshape(D**2,D**2)
            elif ini=='ssb':
                C[bdir] =(torch.ones(1,dtype=dtype)).reshape(1,1)
    
    if E is None:
        E = [0 for x in range(4)]
        for bdir in range(4):
            if ini=='rand':
               E[bdir]=torch.rand(chi,D,D,chi,dtype=dtype)
            elif ini=='symmetric':
               pvecL=bdir2pvecL(bdir)
               E[bdir] = (nt.ncon_torch([T_prime.permute(pvecL),T.permute(pvecL)],[[-1,2,-5,-3,3],[-2,2,-6,-4,3]])).reshape(D**2,D,D,D**2)
            elif ini=='ssb':
               E[bdir]=(torch.tensor([[1,0],[0,0]],dtype=dtype)).reshape(1,D,D,1)
          #  print('bdir=',k,'ini shape E',E[k].shape)      
   # breakpoint()    
    temp=np.ones((1,3))
    stol = 1e-30
    error=1

    for k in range(initsteps):
        for bdir in range(4):
            #breakpoint()            
            tensors=T,C,E,bdir,chi,stol,dtch_iso,T_prime

            if use_check_point==1:
                C,E,norms=checkpoint(doBoundCont_torch_new, *tensors, use_reentrant=False)
            else: 
                C,E,norms=doBoundCont_torch_new(*tensors)
            #C,E,norms = doBoundCont_torch_new(T,C,E,bdir,chi,stol,T_prime)
            if bdir==0: 
                error=np.linalg.norm(norms[0,2]-temp[0,2])
                temp=norms
            #breakpoint()
            if k % 10 == 0 and bdir==0:     
#        if 1==1:
                if filename_txt is not None:
                   f = open(filename_txt, "a")  # append mode 
                   f.write('CTMRG_step='+str(k)+',-log10_CTMRG_error='+str(-np.log10(error))+'\n')
                   f.close()
                print('CTMRG_step='+str(k)+',-log10_CTMRG_error='+str(-np.log10(error))+'\n')
        if error<tol: 
            break
    return C,E,chi,error
            
   
def doBoundCont_torch_new(*tensors):

    T,C,E,bdir,chi,stol,dtch_iso,T_prime=tensors
    chiD = T.shape[0]
    pvecL=bdir2pvecL(bdir)
    
    # optimise for A-B truncation
    TT=nt.ncon_torch([T_prime.permute(pvecL),T.permute(pvecL)],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]])
    #breakpoint() 
    tensors=[C[pvecL[0]], E[pvecL[0]], E[pvecL[0]], C[pvecL[1]],                    
             E[pvecL[3]], TT,          TT,          E[pvecL[1]],                   
             E[pvecL[3]], TT,          TT,          E[pvecL[1]],                   
             C[pvecL[3]], E[pvecL[2]], E[pvecL[2]], C[pvecL[2]]] 
    legs=[[4,1],        [1,5,6,-1],               [-4,7,8,3],               [3,9],    
          [16,11,10,4], [11,10,5,6,-2,-3,18,17],  [-5,-6,7,8,14,15,20,19],  [9,14,15,21],    
          [28,23,22,16],[23,22,18,17,24,25,30,29],[24,25,20,19,26,27,32,31],[21,26,27,33],    
          [34,28],      [35,30,29,34],            [36,32,31,35],            [33,36]]

    contract_order=[4,1,5,6,10,11,9,3,7,8,14,15,34,28,22,23,29,30,36,33,26,27,31,32,24,25,35,19,20,21,16,17,18]
    #time_1=time.time()
    #print('bdir=',bdir,'C shape old=',(C[pvecL[0]]).shape, 'E shape old=',(E[pvecL[3]]).shape) 
    #breakpoint()
    cut= (nt.ncon_torch(tensors,legs,contract_order)).reshape(E[pvecL[0]].shape[3]*chiD**2,E[pvecL[0]].shape[0]*chiD**2)
    
    #torch.save(AB_cut,'AB_cut.pt')
    #time_2=time.time()
    uM, sM, vhM = svd(cut)
    if dtch_iso==1:
        uM=uM.detach()
    #time_3=time.time()
    #uu,ss,vv=torch.linalg.svd(AB_cut)
    #time_4=time.time()
    #print(device,'size=',AB_cut.shape[0],'time_svd=',time_3-time_2,'time_ncon=',time_2-time_1,'time_linalg_svd=',time_4-time_3)
    #print(device,'size=',AB_cut.shape[0],'time_svd=',time_3-time_2,'time_ncon=',time_2-time_1)
    chiNew = min(sum(sM > stol),chi)  
    L=(uM[:,:chiNew]).reshape(E[pvecL[0]].shape[3],chiD,chiD,chiNew)
    L=L.conj()
    R=L.conj()
  
    
    # generate new boundary tensors and normalize
    C1temp=nt.ncon_torch([C[pvecL[0]],E[pvecL[3]],L],[[1,2],[-1,3,4,1],[2,3,4,-2]])
    C2temp=nt.ncon_torch([C[pvecL[1]],E[pvecL[1]],R],[[2,1],[1,3,4,-2],[2,3,4,-1]])
    #print('size_R=',R.shape,'size_E=',(E[pvecL[0]]).shape,'size_T=',(T.permute(pvecL)).shape,'size_L=',L.shape)
    Etemp=nt.ncon_torch([R,E[pvecL[0]],T_prime.permute(pvecL),T.permute(pvecL),L],
        [[1,2,4,-1],[1,3,5,7],[2,3,8,-2,6],[4,5,9,-3,6],[7,8,9,-4]])
 


    C[pvecL[0]]=C1temp /torch.max(torch.abs(C1temp))
    C[pvecL[1]]=C2temp /torch.max(torch.abs(C2temp))
    E[pvecL[0]]=Etemp /torch.max(torch.abs(Etemp))
    #print('E_shape_new=',(E[pvecL[0]]).shape)
    
    def temp_fun(x):

        if device=='cpu':
            y=(torch.max(torch.abs(x))).detach().numpy()
        elif device=='cuda':
            y=((torch.max(torch.abs(x)))).to('cpu').detach().numpy()
        y=y.reshape(1)
        return y
    
    norms=np.zeros((1,6))
    norms[0,0]=temp_fun(C1temp) 
    norms[0,1]=temp_fun(C2temp)
    norms[0,2]=temp_fun(Etemp)  
              
    return C, E, norms


    






   

    

    
    

