import torch
import math
from CTMRG_t_sym_1_ucell import CTMRG_torch as CTMRG
#import CTMRG_t_v2
# import ncon_t as nt
from ncon_t import ncon_torch
import numpy as np
import sys
# from T2para_square_lattice_sym_potts import para2T_torch
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from ncon import ncon
from torch2np import torch2np
# from CTMRG_t_1_site_ucell import CTMRG_torch as non_sym_CTMRG
dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
####################################################################################################

legs_11=[[1,3],[3,4,5,11],[11,12],
             [2,6,7,1],[6,7,4,5,13,14,8,9],[12,13,14,15],
             [10,2],[16,8,9,10],[15,16]]

legs_12=[[1,3],[3,4,5,11],[11,12,13,19],[19,20],
             [2,6,7,1],[6,7,4,5,14,15,8,9],[14,15,12,13,21,22,16,17],[20,21,22,23],
             [10,2],[18,8,9,10],[24,16,17,18],[23,24]]


legs_12_prime=[[1,3],[3,4,5,11],[11,12,13,19],[19,20],
               [2,6,7,1],[6,7,4,5,12,13,21,22,16,17,8,9],[20,21,22,23],
               [10,2],[18,8,9,10],[24,16,17,18],[23,24]]

order_11=[16,10,15,8,9,13,14,2,6,7,12,4,5,11,1,3]
order_12=[10,20,24,16,17,21,22,12,13,19,18,8,9,14,15,2,6,7,4,5,11,1,3]
order_12_prime=[20,10,24,18,23,8,9,16,17,21,22,2,6,7,12,13,19,4,5,11,1,3]

x=torch.tensor([[0,1],[1,0]],dtype=dtype, requires_grad=False)
z=torch.tensor([[1,0],[0,-1]],dtype=dtype, requires_grad=False)
Id = torch.eye(2, dtype=dtype, requires_grad=False, device=device)

X_Id =  ncon_torch([x, Id], [[-1, -3], [-2, -4]])
Z_Id =  ncon_torch([z, Id], [[-1, -3], [-2, -4]])
Id_X =  ncon_torch([Id, x], [[-1, -3], [-2, -4]])
Id_Z =  ncon_torch([Id, z], [[-1, -3], [-2, -4]])
X_Z =  ncon_torch([x, z], [[-1, -3], [-2, -4]])
Z_X =  ncon_torch([z, x], [[-1, -3], [-2, -4]])
X_Id = X_Id.reshape(4, 4)
Z_Id = Z_Id.reshape(4, 4)
Id_X = Id_X.reshape(4, 4)
Id_Z = Id_Z.reshape(4, 4)
X_Z = X_Z.reshape(4, 4)
Z_X = Z_X.reshape(4, 4)

####################################################################################################
def loss_fn(params,h_x,h_z,h_w,
            D,chi,initsteps,CTMRG_tol,C_0,E_0,
            eig_safe, use_check_point,
            para2T_torch,
            **kwargs):
    filename_txt=kwargs.get('filename_txt')
    dtch_iso=kwargs.get('dtch_iso',0)
    q=4
    # T=para2T_torch(params,q,D)
    T=para2T_torch(params)
    T=T/torch.norm(T)
    D=T.shape[0]
       
    C,E,C_0,E_0,chi,error=CTMRG(T, chi, initsteps, CTMRG_tol,
                                eig_safe, C_0=C_0, E_0=E_0,
                                use_check_point=use_check_point,
                                filename_txt=filename_txt,
                                dtch_iso=dtch_iso
                                )
    


    TT_X_Id =  ncon_torch([T.conj(),X_Id,T],[[-1,-3,-5,-7,1],[1,2],[-2,-4,-6,-8,2]])
    TT_Z_Id =  ncon_torch([T.conj(),Z_Id,T],[[-1,-3,-5,-7,1],[1,2],[-2,-4,-6,-8,2]])
    TT_Id_X =  ncon_torch([T.conj(),Id_X,T],[[-1,-3,-5,-7,1],[1,2],[-2,-4,-6,-8,2]])
    TT_Id_Z =  ncon_torch([T.conj(),Id_Z,T],[[-1,-3,-5,-7,1],[1,2],[-2,-4,-6,-8,2]])
    TT_X_Z  =  ncon_torch([T.conj(),X_Z,T],[[-1,-3,-5,-7,1],[1,2],[-2,-4,-6,-8,2]])
    TT_Z_X  =  ncon_torch([T.conj(),Z_X,T],[[-1,-3,-5,-7,1],[1,2],[-2,-4,-6,-8,2]])

    TT =  ncon_torch([T.conj(),T],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]])
    tensors_norm_11 = [C[0],E[0],C[1],E[3],TT,E[1],C[3], E[2],C[2]] 
    
    tensors_norm_12 = [C[0],E[0],E[0],C[1],E[3],TT,TT,E[1],C[3],E[2],E[2],C[2]]   
    
    tensors_X_Id      = [C[0], E[0], C[1], E[3], TT_X_Id, E[1], C[3], E[2], C[2]]
    tensors_Id_Z      = [C[0], E[0], C[1], E[3], TT_Id_Z, E[1], C[3], E[2], C[2]]
    tensors_X_Z       = [C[0], E[0], C[1], E[3], TT_X_Z,  E[1], C[3], E[2], C[2]]
    tensors_Id_X_Id_X = [C[0], E[0], E[0], C[1], E[3], TT_Id_X, TT_Id_X, E[1], C[3], E[2], E[2], C[2]]
    tensors_Z_X_Z_X   = [C[0], E[0], E[0], C[1], E[3], TT_Z_X, TT_Z_X,   E[1], C[3], E[2], E[2], C[2]]
    tensors_Z_Id_Z_Id = [C[0], E[0], E[0], C[1], E[3], TT_Z_Id, TT_Z_Id, E[1], C[3], E[2], E[2], C[2]]


    norm_11 =  ncon_torch(tensors_norm_11,legs_11,order_11)
    norm_12 =  ncon_torch(tensors_norm_12,legs_12,order_12)
    
    E_X_Id      =  ncon_torch(tensors_X_Id,legs_11,order_11)/norm_11
    E_Id_Z      =  ncon_torch(tensors_Id_Z,legs_11,order_11)/norm_11
    E_X_Z       =  ncon_torch(tensors_X_Z,legs_11,order_11)/norm_11
    E_Id_X_Id_X =  ncon_torch(tensors_Id_X_Id_X,legs_12,order_12)/norm_12
    E_Z_X_Z_X   =  ncon_torch(tensors_Z_X_Z_X,legs_12,order_12)/norm_12
    E_Z_Id_Z_Id =  ncon_torch(tensors_Z_Id_Z_Id,legs_12,order_12)/norm_12

    Energy = - E_X_Id - E_Id_Z - E_X_Z - h_x*E_Id_X_Id_X - h_w*E_Z_X_Z_X - h_z*E_Z_Id_Z_Id
    return Energy, C_0, E_0, error,

####################################################################################################
def expec_local_op(T, chi, initsteps, CTMRG_tol, 
                   eig_safe, C_0, E_0,
                   h_x, h_z, h_w
                   ):
    q = 4
    T = T / torch.norm(T)
    D = T.shape[0]
    # eig_safe=0
    C, E, C_0, E_0, chi, error = CTMRG(T, chi, initsteps, CTMRG_tol, 
                                       eig_safe, C_0=C_0, E_0=E_0,
                                       )

    TT_X_Id =  ncon_torch([T.conj(), X_Id, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])
    TT_Z_Id =  ncon_torch([T.conj(), Z_Id, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])
    TT_Id_X =  ncon_torch([T.conj(), Id_X, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])
    TT_Id_Z =  ncon_torch([T.conj(), Id_Z, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])
    TT_X_Z =  ncon_torch([T.conj(), X_Z, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])
    TT_Z_X =  ncon_torch([T.conj(), Z_X, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])

    TT =  ncon_torch([T.conj(), T], [[-1, -3, -5, -7, 1], [-2, -4, -6, -8, 1]])
    tensors_norm_11 = [C[0], E[0], C[1], E[3], TT, E[1], C[3], E[2], C[2]]
    tensors_norm_12 = [C[0], E[0], E[0], C[1], E[3], TT, TT, E[1], C[3], E[2], E[2], C[2]]

    tensors_X_Id      = [C[0], E[0], C[1], E[3], TT_X_Id, E[1], C[3], E[2], C[2]]
    tensors_Id_Z      = [C[0], E[0], C[1], E[3], TT_Id_Z, E[1], C[3], E[2], C[2]]
    tensors_X_Z       = [C[0], E[0], C[1], E[3], TT_X_Z, E[1], C[3], E[2], C[2]]
    tensors_Id_X_Id_X = [C[0], E[0], E[0], C[1], E[3], TT_Id_X, TT_Id_X, E[1], C[3], E[2], E[2], C[2]]
    tensors_Z_X_Z_X   = [C[0], E[0], E[0], C[1], E[3], TT_Z_X, TT_Z_X, E[1], C[3], E[2], E[2], C[2]]
    tensors_Z_Id_Z_Id = [C[0], E[0], E[0], C[1], E[3], TT_Z_Id, TT_Z_Id, E[1], C[3], E[2], E[2], C[2]]

    norm_11 =  ncon_torch(tensors_norm_11, legs_11, order_11)
    norm_12 =  ncon_torch(tensors_norm_12, legs_12, order_12)

    E_X_Id      =  ncon_torch(tensors_X_Id, legs_11, order_11) / norm_11
    E_Id_Z      =  ncon_torch(tensors_Id_Z, legs_11, order_11) / norm_11
    E_X_Z       =  ncon_torch(tensors_X_Z,  legs_11, order_11) / norm_11
    E_Id_X_Id_X =  ncon_torch(tensors_Id_X_Id_X, legs_12, order_12) / norm_12
    E_Z_X_Z_X   =  ncon_torch(tensors_Z_X_Z_X,   legs_12, order_12) / norm_12
    E_Z_Id_Z_Id =  ncon_torch(tensors_Z_Id_Z_Id, legs_12, order_12) / norm_12

    Energy = - E_X_Id - E_Id_Z - E_X_Z - h_x * E_Id_X_Id_X - h_z * E_Z_Id_Z_Id - h_w * E_Z_X_Z_X

    tensors_Z_Id = [C[0], E[0], C[1], E[3], TT_Z_Id, E[1], C[3], E[2], C[2]]
    tensors_Id_X = [C[0], E[0], C[1], E[3], TT_Id_X, E[1], C[3], E[2], C[2]]
    tensors_Z_X  = [C[0], E[0], C[1], E[3], TT_Z_X,  E[1], C[3], E[2], C[2]]

    O_Z_Id =  ncon_torch(tensors_Z_Id, legs_11, order_11) / norm_11
    O_Id_X =  ncon_torch(tensors_Id_X, legs_11, order_11) / norm_11
    O_Z_X =  ncon_torch(tensors_Z_X, legs_11, order_11) / norm_11

    return Energy, C_0, E_0, error, \
            E_X_Id, E_Id_Z, E_X_Z, \
            E_Id_X_Id_X, E_Z_X_Z_X, E_Z_Id_Z_Id, \
            O_Z_Id, O_Id_X, O_Z_X

####################################################################################################
def Expec_disorder_para(T, chi, initsteps, CTMRG_tol, 
                        eig_safe=0, C_0=None, E_0=None,
                        O='X_Id'
                        ):
    D = T.shape[0]
    C_0 = C_0 + 1e-6 * torch.randn_like(C_0)
    E_0 = E_0 + 1e-6 * torch.randn_like(E_0)
    
    C, E, C_0, E_0, chi, error = CTMRG(T, chi, initsteps, CTMRG_tol, 
                                       eig_safe, C_0=C_0, E_0=E_0
                                       )

    if O == 'X_Id':
        X =  ncon_torch([x, Id], [[-1, -3], [-2, -4]])
    elif O == 'Id_Z':
        X =  ncon_torch([Id, z], [[-1, -3], [-2, -4]])
    elif O == 'X_Z':
        X =  ncon_torch([x, z], [[-1, -3], [-2, -4]])

    X = X.reshape(4, 4)
    TT_X =  ncon_torch([T.conj(), X, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])
    T_X =  ncon_torch([T, X], [[-1, -2, -3, -4, 5], [-5, 5]])
    
    #####################
    C_0 = C_0 + 1e-6 * torch.randn_like(C_0)
    E_0 = E_0 + 1e-6 * torch.randn_like(E_0)
    
    C_X, E_XX, C_0_X, E_0_X, chi, error_X = CTMRG(T, chi, initsteps, CTMRG_tol, 
                                                  eig_safe, C_0=C_0, E_0=E_0,
                                                  T_prime=T_X
                                                  )

    TT =  ncon_torch([T.conj(), T], [[-1, -3, -5, -7, 1], [-2, -4, -6, -8, 1]])

    E_0 = E[0]
    E_2 = E[2]
    E_0_X = E_XX[0]
    E_2_X = E_XX[2]
    # TT = torch2np(TT, device)
    # TT_X = torch2np(TT_X, device)

    # E_0 = torch2np(E_0, device)
    # E_2 = torch2np(E_2, device)
    # E_0_X = torch2np(E_0_X, device)
    # E_2_X = torch2np(E_2_X, device)

    def map_left_numpy_3(v):
        v = v.reshape((chi, D, D, chi))
        v = torch.from_numpy(v).to(device, dtype=dtype)  # numpy → torch
        y =  ncon_torch([v, E_0, TT, E_2], 
                          [[1, 2, 3, 4], [1, 5, 6, -1], [3, 2, 5, 6, -3, -2, 8, 7], [-4, 8, 7, 4]], 
                          [4, 2, 3, 7, 8, 1, 5, 6]
                          )
        y = y.contiguous().view(-1).cpu().numpy()  # torch → numpy
        y = y.reshape((chi ** 2 * D ** 2, 1))
        return y

    def map_left_numpy_2(v2):
        v2 = v2.reshape((chi, chi))
        v2 = torch.from_numpy(v2).to(device, dtype=dtype)  # numpy → torch
        y2 =  ncon_torch([v2, E_0, E_2], 
                           [[1, 2], [1, 5, 6, -1], [-2, 6, 5, 2]], 
                           [1, 2, 5, 6]
                           )
        y2 = y2.contiguous().view(-1).cpu().numpy()  # torch → numpy
        y2 = y2.reshape((chi ** 2, 1))
        return y2

    def map_left_numpy_X_2(v_X_2):
        v_X_2 = v_X_2.reshape((chi, chi))
        v_X_2 = torch.from_numpy(v_X_2).to(device, dtype=dtype)  # numpy → torch
        y_X_2 =  ncon_torch([v_X_2, E_0_X, E_2_X], 
                              [[1, 2], [1, 5, 6, -1], [-2, 6, 5, 2]],
                              [2, 1, 5, 6]
                              )
        y_X_2 = y_X_2.contiguous().view(-1).cpu().numpy()  # torch → numpy
        y_X_2 = y_X_2.reshape((chi ** 2, 1))
        return y_X_2

    def map_left_numpy_X_3(v_X):
        v_X = v_X.reshape((chi, D, D, chi))
        v_X = torch.from_numpy(v_X).to(device, dtype=dtype)  # numpy → torch
        y_X =  ncon_torch([v_X, E_0_X, TT_X, E_2_X], 
                            [[1, 2, 3, 4], [1, 5, 6, -1], [3, 2, 5, 6, -3, -2, 8, 7], [-4, 8, 7, 4]],
                            [4, 2, 3, 7, 8, 1, 5, 6]
                            )
        y_X = y_X.contiguous().view(-1).cpu().numpy()  # torch → numpy
        y_X = y_X.reshape((chi ** 2 * D ** 2, 1))
        return y_X

    def map_left_numpy_ovlp(v_ovlp):
        v_ovlp = v_ovlp.reshape((chi, chi))
        v_ovlp = torch.from_numpy(v_ovlp).to(device, dtype=dtype)  # numpy → torch
        y_ovlp =  ncon_torch([v_ovlp, E_0_X, E_2], 
                             [[1, 2], [1, 5, 6, -1], [-2, 6, 5, 2]],
                             [1, 2, 5, 6]
                             )
        y_ovlp = y_ovlp.contiguous().view(-1).cpu().numpy()  # torch → numpy
        y_ovlp = y_ovlp.reshape((chi ** 2, 1))
        return y_ovlp

    dim3 = chi ** 2 * D ** 2
    dim2 = chi ** 2

    maps_3    = LinearOperator((dim3, dim3), matvec=map_left_numpy_3)
    maps_X_3  = LinearOperator((dim3, dim3), matvec=map_left_numpy_X_3)
    maps_2    = LinearOperator((dim2, dim2), matvec=map_left_numpy_2)
    maps_X_2  = LinearOperator((dim2, dim2), matvec=map_left_numpy_X_2)
    maps_ovlp = LinearOperator((dim2, dim2), matvec=map_left_numpy_ovlp)


    t_3 = eigsh(maps_3, which='LM', maxiter=500, return_eigenvectors=False)
    order = np.argsort(np.abs(t_3))
    order = order[::-1]
    t_3 = t_3[order]

    t_X_3 = eigsh(maps_X_3, which='LM', maxiter=500, return_eigenvectors=False)
    order = np.argsort(np.abs(t_X_3))
    order = order[::-1]
    t_X_3 = t_X_3[order]

    t_2 = eigsh(maps_2, which='LM', maxiter=500, return_eigenvectors=False)
    order = np.argsort(np.abs(t_2))
    order = order[::-1]
    t_2 = t_2[order]

    t_X_2 = eigsh(maps_X_2, which='LM', maxiter=500, return_eigenvectors=False)
    order = np.argsort(np.abs(t_X_2))
    order = order[::-1]
    t_X_2 = t_X_2[order]

    t_ovlp = eigsh(maps_ovlp, which='LM', maxiter=500, return_eigenvectors=False)
    order = np.argsort(np.abs(t_ovlp))
    order = order[::-1]
    t_ovlp = t_ovlp[order]
    
    xi = -1 / np.log(t_2[1] / t_2[0])
    area = (t_X_3[0] / t_X_2[0]) / (t_3[0] / t_2[0])
    perimeter = t_ovlp[0] / (np.sqrt(np.abs(t_2[0] * t_X_2[0])))

    return xi, area, perimeter




# def Expec_disorder_para(T, chi, initsteps, CTMRG_tol, eig_safe=0, O='X_Id'):
#     D = T.shape[0]
#     C, E, C_0, E_0, chi, error = CTMRG(T, chi, initsteps, CTMRG_tol, eig_safe)

#     if O == 'X_Id':
#         X =  ncon_torch([x, Id], [[-1, -3], [-2, -4]])
#     elif O == 'Id_Z':
#         X =  ncon_torch([Id, z], [[-1, -3], [-2, -4]])
#     elif O == 'X_Z':
#         X =  ncon_torch([x, z], [[-1, -3], [-2, -4]])

#     X = X.reshape(4, 4)
#     TT_X =  ncon_torch([T.conj(), X, T], [[-1, -3, -5, -7, 1], [1, 2], [-2, -4, -6, -8, 2]])
#     T_X =  ncon_torch([T, X], [[-1, -2, -3, -4, 5], [-5, 5]])

#     C_X, E_XX, C_0_X, E_0_X, chi, error_X = CTMRG(T, chi, initsteps, CTMRG_tol, eig_safe, T_prime=T_X)

#     TT =  ncon_torch([T.conj(), T], [[-1, -3, -5, -7, 1], [-2, -4, -6, -8, 1]])

#     E_0 = E[0]
#     E_2 = E[2]
#     E_0_X = E_XX[0]
#     E_2_X = E_XX[2]
#     TT = torch2np(TT, device)
#     TT_X = torch2np(TT_X, device)

#     E_0 = torch2np(E_0, device)
#     E_2 = torch2np(E_2, device)
#     E_0_X = torch2np(E_0_X, device)
#     E_2_X = torch2np(E_2_X, device)

#     def map_left_numpy_3(v):
#         v = v.reshape((chi, D, D, chi))
#         y = ncon([v, E_0, TT, E_2], [[1, 2, 3, 4], [1, 5, 6, -1], [3, 2, 5, 6, -3, -2, 8, 7], [-4, 8, 7, 4]])
#         y = y.reshape((chi ** 2 * D ** 2, 1))
#         return y

#     def map_left_numpy_2(v2):
#         v2 = v2.reshape((chi, chi))
#         y2 = ncon([v2, E_0, E_2], [[1, 2], [1, 5, 6, -1], [-2, 6, 5, 2]])
#         y2 = y2.reshape((chi ** 2, 1))
#         return y2

#     def map_left_numpy_X_2(v_X_2):
#         v_X_2 = v_X_2.reshape((chi, chi))
#         y_X_2 = ncon([v_X_2, E_0_X, E_2_X], [[1, 2], [1, 5, 6, -1], [-2, 6, 5, 2]])
#         y_X_2 = y_X_2.reshape((chi ** 2, 1))
#         return y_X_2

#     def map_left_numpy_X_3(v_X):
#         v_X = v_X.reshape((chi, D, D, chi))
#         y_X = ncon([v_X, E_0_X, TT_X, E_2_X], [[1, 2, 3, 4], [1, 5, 6, -1], [3, 2, 5, 6, -3, -2, 8, 7], [-4, 8, 7, 4]])
#         y_X = y_X.reshape((chi ** 2 * D ** 2, 1))
#         return y_X

#     def map_left_numpy_ovlp(v_ovlp):
#         v_ovlp = v_ovlp.reshape((chi, chi))
#         y_ovlp = ncon([v_ovlp, E_0_X, E_2], [[1, 2], [1, 5, 6, -1], [-2, 6, 5, 2]])
#         y_ovlp = y_ovlp.reshape((chi ** 2, 1))
#         return y_ovlp

#     dim3 = chi ** 2 * D ** 2
#     dim2 = chi ** 2

#     maps_3 = LinearOperator((dim3, dim3), matvec=map_left_numpy_3)
#     maps_X_3 = LinearOperator((dim3, dim3), matvec=map_left_numpy_X_3)
#     maps_2 = LinearOperator((dim2, dim2), matvec=map_left_numpy_2)
#     maps_X_2 = LinearOperator((dim2, dim2), matvec=map_left_numpy_X_2)
#     maps_ovlp = LinearOperator((dim2, dim2), matvec=map_left_numpy_ovlp)


#     t_3 = eigsh(maps_3, which='LM', maxiter=500, return_eigenvectors=False)
#     order = np.argsort(np.abs(t_3))
#     order = order[::-1]
#     t_3 = t_3[order]

#     t_X_3 = eigsh(maps_X_3, which='LM', maxiter=500, return_eigenvectors=False)
#     order = np.argsort(np.abs(t_X_3))
#     order = order[::-1]
#     t_X_3 = t_X_3[order]

#     t_2 = eigsh(maps_2, which='LM', maxiter=500, return_eigenvectors=False)
#     order = np.argsort(np.abs(t_2))
#     order = order[::-1]
#     t_2 = t_2[order]

#     t_X_2 = eigsh(maps_X_2, which='LM', maxiter=500, return_eigenvectors=False)
#     order = np.argsort(np.abs(t_X_2))
#     order = order[::-1]
#     t_X_2 = t_X_2[order]

#     t_ovlp = eigsh(maps_ovlp, which='LM', maxiter=500, return_eigenvectors=False)
#     order = np.argsort(np.abs(t_ovlp))
#     order = order[::-1]
#     t_ovlp = t_ovlp[order]
    
#     xi = -1 / np.log(t_2[1] / t_2[0])
#     area = (t_X_3[0] / t_X_2[0]) / (t_3[0] / t_2[0])
#     perimeter = t_ovlp[0] / (np.sqrt(np.abs(t_2[0] * t_X_2[0])))

#     return xi, area, perimeter
