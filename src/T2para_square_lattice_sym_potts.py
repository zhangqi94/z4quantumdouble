import torch
import numpy as np
from ncon import ncon

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fname(q,D):
    return 'src/basis/Square_lattice_sym_basis_q='+str(q)+'_D='+str(D)+'.pt'
 

def T2para_numpy(T):
    D=T.shape[0]
    q=T.shape[-1]
    data=torch.load(fname(q,D), weights_only=False)
    basis=data.get('basis')
    T=T.reshape((D**4*q))
    coeff=np.zeros((len(basis)))
#    breakpoint()
    for n0 in range(len(basis)):
        coeff[n0]=T.reshape((1,D**4*q))@(basis[n0]).reshape((D**4*q,1))
    return coeff


# def T2para_torch(T,grad=True):
#     D=T.shape[0]
#     q=T.shape[-1]
#     data=torch.load(fname(q,D), weights_only=False) 
#     basis=data.get('basis')
#     T=T.reshape((D**4*q))
#     params=[]
# #    breakpoint()
#     for n0 in range(len(basis)):
#         params.append(torch.tensor(T.reshape((1,D**4*q))@(basis[n0]).reshape((D**4*q,1)),requires_grad=grad))
#     return params

# def para2T_torch(coeff,q,D):
#     dtype=torch.double
#     T=torch.zeros(D**4*q,1,dtype=dtype)
#     data=torch.load(fname(q,D), weights_only=False)
#     basis=data.get('basis')
#     for nn0 in range(len(basis)):
#         T=T+coeff[nn0]*(torch.tensor(basis[nn0],dtype=dtype)).reshape(D**4*q,1)
#     return T.reshape(D,D,D,D,q)

def make_para2Ttrans_torch(q, D):
    data = torch.load(fname(q, D), weights_only=False)
    basis = data.get('basis')
    basis = np.array(basis)
    basis_tensor = torch.from_numpy(basis).to(dtype).to(device)
    basis_reshaped = basis_tensor.reshape(len(basis), D**4 * q)
    
    def T2para_torch(T):
        T = torch.tensor(T, dtype=dtype)
        T_flat = T.reshape(-1)
        params = torch.mv(basis_tensor, T_flat)
        params = params.clone().detach().requires_grad_()
        return params
    
    def para2T_torch(coeff):
        T = torch.einsum('n,nm->m', coeff, basis_reshaped)
        return T.reshape(D, D, D, D, q)

    return T2para_torch, para2T_torch


def para2T_numpy(coeff,q,D):
    T=np.zeros((1,D**4*q))
    data=torch.load(fname(q,D), weights_only=False)
    basis=data.get('basis')
    for nn0 in range(len(basis)):
#         breakpoint()
         T=T+coeff[nn0]*basis[nn0]
    return T.reshape([D,D,D,D,q])
#do a test

#D=2
#data=torch.load('/home/t30/pol/ge74suj/post_doc_project/TC_PEPS_pytorch/generate_basis/Square_lattice_sym_basis_D='+str(D)+'.pt') 
#basis=data.get('basis')
#T=np.random.rand(2,2,2,2,2)
#T=(T+T.transpose([1,0,3,2,4]))/2
#T=(T+T.transpose([3,2,1,0,4]))/2
#coeff=T2para_numpy(T)
#T_temp=para2T_numpy(coeff,D)
#print('difference=',np.linalg.norm(T-T_temp))







 







