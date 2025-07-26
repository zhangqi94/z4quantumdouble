#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import math
from ncon_t import ncon_torch as ncon 
import numpy as np
import copy
import time
from torch2np import torch2np
from torch.utils.checkpoint import checkpoint
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
from ncon import ncon as ncon_np

#dtype = torch.double
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)



def CTMRG_torch(T, 
                chi, 
                initsteps, 
                tol, 
                eig_safe,
                C_0=None,
                E_0=None,
                use_check_point=0,
                filename_txt=None,
                dtch_iso=0,
                T_prime=None
                ):
    D=T.shape[0]
    dtype=T.dtype
    ini_chi=chi
    
    if C_0 is None:
        C_0_temp=ncon([T.conj(),T],[[2,3,-1,-2,1],[2,3,-3,-4,1]])
        C_0_temp=C_0_temp.reshape((D**2,D**2))
        C_0_temp=(C_0_temp+C_0_temp.conj().t())/2
        C_0=torch.zeros((chi,chi),dtype=dtype)
        C_0[0:D**2,0:D**2]=C_0_temp
        C_0=C_0 + 1e-5 * torch.randn(C_0.shape, dtype=C_0.dtype, device=C_0.device)
        
        # C_0=torch.zeros((chi,chi),dtype=dtype)
        # C_0=torch.randn(C_0.shape, dtype=C_0.dtype, device=C_0.device)

    if E_0 is None:   
        E_0_temp=ncon([T.conj(),T],[[2,-1,-3,-5,1],[2,-2,-4,-6,1]])
        E_0_temp=E_0_temp.reshape(D**2,D,D,D**2)
        E_0_temp=(E_0_temp+E_0_temp.conj().permute(3,2,1,0))/2 
        E_0_temp=E_0_temp.reshape((D**2,D**2,D**2))
        E_0=torch.zeros((chi,D**2,chi),dtype=dtype)
        E_0[0:D**2,0:D**2,0:D**2]=E_0_temp
        E_0=E_0 + 1e-5 * torch.randn(E_0.shape, dtype=E_0.dtype, device=E_0.device)
        
        # E_0=torch.zeros((chi,D**2,chi),dtype=dtype)
        # E_0=torch.randn(E_0.shape, dtype=E_0.dtype, device=E_0.device)

    temp=np.ones((1,chi))
    stol = 1e-20
    error=1
    for k in range(initsteps):
        if T_prime==None:
            tensors=T,C_0,E_0,chi,stol,eig_safe,dtch_iso,T
        else:
            tensors=T,C_0,E_0,chi,stol,eig_safe,dtch_iso,T_prime
        if use_check_point==1:
            C_0,E_0,s,trc_err=checkpoint(doBoundCont_torch_new, *tensors, use_reentrant=False)
        else: 
            C_0,E_0,s,trc_err=doBoundCont_torch_new(*tensors)
        
        trc_err=torch2np(trc_err,device)
        s=torch2np(s,device)  
#        breakpoint()
        try:
            error=np.linalg.norm(np.abs(s)-temp[0:len(s)])
        except: 
            temp=np.ones((1,len(s))) 
            error=np.linalg.norm(np.abs(s)-temp[0:len(s)])
        temp=s     
        # if k % 1 == 0:     
        #     # if filename_txt is not None:
        #     #    f = open(filename_txt, "a")  # append mode 
        #     #    f.write('CTMRG_step='+str(k)+',-log10_CTMRG_error='+str(-np.log10(error))+',trc_err='+str(trc_err)+'\n')
        #     #    f.close()
        #     # print('CTMRG_step='+str(k)+',-log10_CTMRG_error='+str(-np.log10(error))+',trc_err='+str(trc_err)+'\n')
        #     if filename_txt is not None:
        #        f = open(filename_txt, "a")  # append mode 
        #        f.write(f"ctmrg {k}    err {error}    trc_err {trc_err}\n")
        #        f.close()
        #     print(f"ctmrg {k}    err {error}    trc_err {trc_err}")
        if error<tol: 
            break
    chi_ctm=C_0.shape[0]  
    C=[C_0,C_0,C_0,C_0]
    #E=[E_0,E_0_temp,E_0_temp,E_0]
    E_0_temp=E_0.reshape(E_0.shape[0],D,D,E_0.shape[-1])
    E=[E_0_temp,E_0_temp,E_0_temp,E_0_temp]
    return C,E,C_0,E_0,chi,error
            
   
def doBoundCont_torch_new(*tensors):
    T,C_0,E_0,chi,stol,eig_safe,dtch_iso,T_prime=tensors

    class EigenSolver(torch.autograd.Function):
          @staticmethod
          def forward(self, A):
              w, v = torch.linalg.eigh(A)
              self.save_for_backward(w, v)
              return w, v

          @staticmethod
          def backward(self, dw, dv):
              w, v = self.saved_tensors
              dtype, device = w.dtype, w.device
              N = v.shape[0]

              F = w - w[:,None]
              F.diagonal().fill_(np.inf)
        # safe inverse
              msk = (torch.abs(F) < eig_safe)
              F[msk] += eig_safe
              F = 1./F  

              vt = v.t().conj()
              vdv = vt@dv

              return v@(torch.diag(dw) + F*(vdv-vdv.t().conj())/2) @vt


    eigh=EigenSolver.apply
    D = T.shape[0]
    chi_ctm=C_0.shape[0]
    # optimise for A-B truncation
    TT=ncon([T_prime.conj(),T],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]])
    TT=TT.reshape(D**2,D**2,D**2,D**2)
  
    #if TT.is_complex():
    #    TT=TT.real

  #if TT.is_complex():
    #    breakpoint()
    #test sym
    #error_1=torch.norm(TT-TT.permute(1,0,3,2))
    #error_2=torch.norm(TT-TT.permute(3,2,1,0))
#    print('square_sym_err_of_TT=',error_1)
    tensors=[C_0,  E_0,                    
             E_0,  TT]                   
                  
    legs=[[2,1],     [1,3,-1],    
          [2,4,-3], [4,3,-2,-4]]  
 
    contract_order=[1,2,4,3]

    #for idx, tensor in enumerate(tensors):
    #    if tensor.is_complex():
    #        print(f"Complex tensor found at index {idx}: {tensor}")
    #        breakpoint() 
    cut=(ncon(tensors,legs,contract_order)).reshape(chi_ctm*D**2,chi_ctm*D**2) 
   
    error=torch.norm(cut-cut.t())
 #   print('square_sym_err_of_cut=',error)
    s,u = eigh(cut)
    if dtch_iso==1:
        u=u.detach()
    s,order=torch.sort(torch.abs(s), descending=True)
    u=u[:,order]
    chi_new=min(sum(s>stol),chi)  
        # generate new boundary tensors and normalize
    u=u[:,0:chi_new]
    u=u.reshape(chi_ctm,D**2,chi_new)
    trc_err=torch.sum(s[chi_new:-1])/torch.sum(s)
    C_temp=ncon([C_0,E_0,E_0,TT,u.conj(),u],[[2,1],[1,3,5],[2,4,7],[4,3,6,8],[5,6,-2],[7,8,-1]]) 
    E_temp=ncon([u.conj(),E_0,TT,u],[[1,2,-1],[1,3,4],[3,2,-2,5],[4,5,-3]])
    C_temp=(C_temp+C_temp.conj().t())/2
    E_temp=(E_temp+E_temp.conj().permute(2,1,0))/2
    C_0=C_temp /torch.max(torch.max(torch.abs(C_temp)))
    E_0=E_temp /torch.max(torch.max(torch.abs(E_temp)))
     
    return C_0,E_0,s,trc_err







   

    

    
    

