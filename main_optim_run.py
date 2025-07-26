
####################################################################################################

import os
import sys
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

import time
import torch
import numpy as np
from ncon import ncon
import argparse

####################################################################################################

import src.loss_fn_torch_sym_AT as LFT
from src.INI_TEN_AT import initialize_tensor
from src.torch2np import torch2np

time_start = time.time()
dtype = torch.float64
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

####################################################################################################
print("\n========== Initialize configs ==========", flush=True)

parser = argparse.ArgumentParser()

parser.add_argument("--D", type=int, default=4)
parser.add_argument("--chi", type=int, default=60)
parser.add_argument('--use_check_point', type=int, default=1)
parser.add_argument('--output_path', type=str, default='data')
parser.add_argument('--instate', type=str, default='random')

parser.add_argument('--h_x', type=float, default=0.0)
parser.add_argument('--h_z', type=float, default=0.0)
parser.add_argument('--h_w', type=float, default=0.0)

parser.add_argument('--is_reverse', type=int, default=0)

parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--tol_grad', type=float, default=1e-10)

parser.add_argument('--max_CTMRG_steps', type=int, default=100)
parser.add_argument('--CTMRG_tol', type=float, default=1e-13)

args = parser.parse_args()

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
print("\n".join(f"{k} = {v}" for k, v in vars(args).items()))

D = args.D
chi = args.chi
use_check_point = args.use_check_point
output_path = args.output_path

is_reverse = args.is_reverse
instate = args.instate

h_x = args.h_x
h_z = args.h_z
h_w = args.h_w

max_CTMRG_steps = args.max_CTMRG_steps
CTMRG_tol = args.CTMRG_tol

max_iter = args.max_iter
tol_grad = args.tol_grad
# tol_energy = args.tol_energy

eig_safe = 1e-12
phys_dim = 4

####################################################################################################
print("\n========== Initialize ipeps tensor ==========", flush=True)

filename     = output_path + "_state.pt"
filename_txt = output_path + "_log.txt"
print("output file:", filename)
print("output file txt:", filename_txt)

f = open(filename_txt, 'w')
f.write(
    f"Filename: {filename}\n"
    f"Device: {device}\n"
)

if instate == 'random':
    T = initialize_tensor(D, 1e-2, 'FM' if is_reverse == 1 else 'PM')
    print("Initialize T")
    print("T.shape:", T.shape)

else:
    data = torch.load(instate, weights_only=False)
    T = data['T']
    del data
    print("Load T from file:", instate)    
    print("T.shape:", T.shape)


####################################################################################################
print("\n========== Optimize ipeps ==========", flush=True)

from src.T2para_square_lattice_sym_potts import make_para2Ttrans_torch
T2para_torch, para2T_torch = make_para2Ttrans_torch(phys_dim, D)

params = T2para_torch(T)
print("num of params:", len(params))
# for i, p in enumerate(params):
#     print(i, p.requires_grad, p.grad)

optimizer = torch.optim.LBFGS([params], 
                            max_iter=max_iter,
                            tolerance_grad=tol_grad, 
                            tolerance_change=1e-12,
                            history_size=30,
                            line_search_fn='strong_wolfe'
                            )

#### initialize C_0 and E_0
C_0 = ncon([T.conj(), T], [[2, 3, -1, -2, 1], [2, 3, -3, -4, 1]])
C_0 = C_0.reshape((D ** 2, D ** 2))
C_0 = (C_0 + C_0.conj().transpose()) / 2
E_0 = ncon([T.conj(), T], [[2, -1, -3, -5, 1], [2, -2, -4, -6, 1]])
E_0 = E_0.reshape(D ** 2, D, D, D ** 2)
E_0 = (E_0 + E_0.conj().transpose(3, 2, 1, 0)) / 2
E_0 = E_0.reshape((D ** 2, D ** 2, D ** 2))

C_0 = torch.tensor(C_0, dtype=dtype, device=device, requires_grad=False)
E_0 = torch.tensor(E_0, dtype=dtype, device=device, requires_grad=False)


closure_state = {"loss": 0, 
                 "C": C_0, 
                 "E": E_0,
                 "dtch_iso": 0,
                 "iter": 0,
                 }
# dtch_iso = 0
def closure():
    # global dtch_iso
    t_CTMRG_0 = time.time()
    optimizer.zero_grad()

    C_0 = closure_state['C'].clone().detach().requires_grad_(False)
    E_0 = closure_state['E'].clone().detach().requires_grad_(False)
    C_0 = C_0 + 1e-8 * torch.randn_like(C_0)
    E_0 = E_0 + 1e-8 * torch.randn_like(E_0)
    dtch_iso = closure_state['dtch_iso']
    loss_old = closure_state['loss']

    loss, C, E, CTMRG_error= LFT.loss_fn(params, h_x, h_z, h_w, 
                                        D, chi, max_CTMRG_steps, CTMRG_tol, C_0, E_0,
                                        eig_safe, use_check_point, 
                                        para2T_torch,
                                        dtch_iso=dtch_iso,
                                        )

    t_CTMRG_1 = time.time()
    
    t_grad_0 = time.time()
    loss.backward()
    t_grad_1 = time.time()
    
    # Grad_norm = torch.norm(torch.cat([p.grad.view(1) for p in params], dim=0).unsqueeze(0))
    Grad_norm = torch.norm(params.grad)
    
    if Grad_norm == torch.tensor(float('nan')) or Grad_norm > 1000:
        dtch_iso = 1

        f.write("Grad norm is unstable\n")
        print("Grad norm is unstable")
        
        optimizer.zero_grad()

        loss, C, E, CTMRG_error= LFT.loss_fn(
            params, h_x, h_z, h_w, 
            D, chi, max_CTMRG_steps, CTMRG_tol, C_0, E_0,
            eig_safe, use_check_point, 
            para2T_torch,
            dtch_iso=dtch_iso,
            )
        t_CTMRG_1 = time.time()
        
        t_grad_0 = time.time()
        loss.backward()
        t_grad_1 = time.time()
        
        # Grad_norm = torch.norm(torch.cat([p.grad.view(1) for p in params], dim=0).unsqueeze(0))
        Grad_norm = torch.norm(params.grad)

    closure_state['loss'] = loss
    closure_state['C'] = C
    closure_state['E'] = E
    closure_state['dtch_iso'] = dtch_iso
    closure_state['iter'] = closure_state['iter'] + 1

    log_str = (
        f"iter = {closure_state['iter']:4d}, "
        f"t_grad = {t_grad_1 - t_grad_0:.2f}, "
        f"t_CTMRG = {t_CTMRG_1 - t_CTMRG_0:.2f}, "
        f"CTMRG_error = {CTMRG_error:.2e}, "
        f"e = {loss.item():.16f}, "
        f"e_err = {torch.abs(loss-loss_old).item():.2e}, "
        f"norm_grad = {Grad_norm:.2e}, "
        f"D = {D}, "
        f"chi = {(C[0]).shape[0]}, "
        f"dtch_iso = {dtch_iso}"
    )
    
    f.write(log_str + '\n')
    print(log_str, flush=True)
    
    return loss


epoch = 0
Grad_norm = 1

f.write(f'hx = {h_x:.6f}, hz = {h_z:.6f}, hw = {h_w:.6f}, epoch={epoch}\n')
print(f"hx = {h_x:.6f}, hz = {h_z:.6f}, hw = {h_w:.6f}, epoch={epoch}")
optimizer.step(closure)


####################################################################################################
print("\n========== Calcualte expectation values ==========", flush=True)

###
t_obs0 = time.time() 

T = para2T_torch(params)
T = T / torch.norm(T)
T = torch2np(T, device)
# Energy = torch2np(Energy, device)
# Grad_norm = torch.norm(torch.cat([p.grad.view(1) for p in params], dim=0).unsqueeze(0))
Grad_norm = torch.norm(params.grad)
Grad_norm = torch2np(Grad_norm, device)


C_0 = closure_state['C'].clone().detach().requires_grad_(False)
E_0 = closure_state['E'].clone().detach().requires_grad_(False)
C_0 = C_0 + 1e-6 * torch.randn_like(C_0)
E_0 = E_0 + 1e-6 * torch.randn_like(E_0)
    
Energy, C_0, E_0, error, \
        E_X_Id, E_Id_Z, E_X_Z, \
        E_Id_X_Id_X, E_Z_X_Z_X, E_Z_Id_Z_Id, \
        O_Z_Id, O_Id_X, O_Z_X \
    = LFT.expec_local_op(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                         eig_safe=eig_safe, C_0=C_0, E_0=E_0,
                         h_x=h_x, h_z=h_z, h_w=h_w
                         )

Energy = torch2np(Energy, device)
E_X_Id = torch2np(E_X_Id, device)
E_Id_Z = torch2np(E_Id_Z, device)
E_X_Z =  torch2np(E_X_Z, device)
E_Id_X_Id_X = torch2np(E_Id_X_Id_X, device)
E_Z_X_Z_X =   torch2np(E_Z_X_Z_X, device)
E_Z_Id_Z_Id = torch2np(E_Z_Id_Z_Id, device)
O_Z_Id = torch2np(O_Z_Id, device)
O_Id_X = torch2np(O_Id_X, device)
O_Z_X =  torch2np(O_Z_X, device)

###
C_0 = C_0 + 1e-6 * torch.randn_like(C_0)
E_0 = E_0 + 1e-6 * torch.randn_like(E_0)

xi, area_X_Id, perimeter_X_Id \
    = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                              eig_safe=eig_safe, C_0=C_0, E_0=E_0, O='X_Id'
                              )
    
xi, area_Id_Z, perimeter_Id_Z \
    = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                              eig_safe=eig_safe, C_0=C_0, E_0=E_0, O='Id_Z'
                              )
    
xi, area_X_Z,  perimeter_X_Z  \
    = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                              eig_safe=eig_safe, C_0=C_0, E_0=E_0, O='X_Z')


log_str = f"""Energy = {Energy:.16f},
E_X_Id = {E_X_Id:.12f}, E_Id_X_Id_X = {E_Id_X_Id_X:.12f}, O_Z_Id = {O_Z_Id:.12f},
E_Id_Z = {E_Id_Z:.12f}, E_Z_X_Z_X   = {E_Z_X_Z_X:.12f}, O_Id_X = {O_Id_X:.12f},
E_X_Z  = {E_X_Z:.12f}, E_Z_Id_Z_Id = {E_Z_Id_Z_Id:.12f}, O_Z_X  = {O_Z_X:.12f},
xi = {xi:.12f},
area_X_Id = {area_X_Id:.12f}, perimeter_X_Id = {perimeter_X_Id:.12f},
area_Id_Z = {area_Id_Z:.12f}, perimeter_Id_Z = {perimeter_Id_Z:.12f},
area_X_Z  = {area_X_Z:.12f}, perimeter_X_Z  = {perimeter_X_Z:.12f}
"""

f.write(log_str + '\n')
print(log_str, flush=True)

t_obs1 = time.time() 
f.write(f"t_obs = {t_obs1 - t_obs0:.2f} s\n")
print(f"t_obs = {t_obs1 - t_obs0:.2f} s", flush=True)

output_dict = {"T": T,
               "Energy": Energy,
               "Grad_norm": Grad_norm,
               "E_X_Id": E_X_Id,
               "E_Id_Z": E_Id_Z,
               "E_X_Z":  E_X_Z,
               "E_Id_X_Id_X": E_Id_X_Id_X,
               "E_Z_X_Z_X":   E_Z_X_Z_X,
               "E_Z_Id_Z_Id": E_Z_Id_Z_Id,
               "O_Z_Id": O_Z_Id,
               "O_Id_X": O_Id_X,
               "O_Z_X":  O_Z_X,
               "xi": xi,
               "area_X_Id":      area_X_Id,
               "perimeter_X_Id": perimeter_X_Id,
               "area_Id_Z":      area_Id_Z,
               "perimeter_Id_Z": perimeter_Id_Z,
               "area_X_Z":       area_X_Z,
               "perimeter_X_Z":  perimeter_X_Z,
               "h_x": h_x,
               "h_z": h_z,
               "h_w": h_w,
               "args": args,               
               }

torch.save(output_dict, filename)

time_end = time.time()
f.write(f'total_time = {time_end - time_start:.2f} s')
f.close()
print(f'total_time = {time_end - time_start:.2f} s')


####################################################################################################
