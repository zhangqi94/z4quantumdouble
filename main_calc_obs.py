
####################################################################################################

import os
import sys
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

import time
import torch
import numpy as np
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

# parser.add_argument("--D", type=int, default=4)
parser.add_argument("--chi", type=int, default=60)
parser.add_argument('--use_check_point', type=int, default=1)
parser.add_argument('--output_path', type=str, default='data')
parser.add_argument('--instate', type=str, default='random')

parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--tol_grad', type=float, default=1e-10)

parser.add_argument('--max_CTMRG_steps', type=int, default=200)
parser.add_argument('--CTMRG_tol', type=float, default=1e-13)

args = parser.parse_args()

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
print("\n".join(f"{k} = {v}" for k, v in vars(args).items()))

# D = args.D
chi = args.chi
use_check_point = args.use_check_point
output_path = args.output_path
instate = args.instate
max_CTMRG_steps = args.max_CTMRG_steps
CTMRG_tol = args.CTMRG_tol

max_iter = args.max_iter
tol_grad = args.tol_grad

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

data = torch.load(instate, weights_only=False)
T = data['T']
T = T / np.linalg.norm(T)
h_x, h_z, h_w = data['h_x'], data['h_z'], data['h_w']
Grad_norm = data['Grad_norm']
D = data['args'].D

print("Load T from file:", instate)    
print("T.shape:", T.shape)


f.write(f'hx = {h_x:.6f}, hz = {h_z:.6f}, hw = {h_w:.6f}\n')
print(f"hx = {h_x:.6f}, hz = {h_z:.6f}, hw = {h_w:.6f}")

####################################################################################################
print("\n========== Calcualte expectation values ==========", flush=True)

# from src.T2para_square_lattice_sym_potts import make_para2Ttrans_torch
# T2para_torch, para2T_torch = make_para2Ttrans_torch(phys_dim, D)

#### initialize C_0 and E_0
import ncon.ncon as ncon
C_0 = ncon([T.conj(), T], [[2, 3, -1, -2, 1], [2, 3, -3, -4, 1]])
C_0 = C_0.reshape((D ** 2, D ** 2))
C_0 = (C_0 + C_0.conj().transpose()) / 2
E_0 = ncon([T.conj(), T], [[2, -1, -3, -5, 1], [2, -2, -4, -6, 1]])
E_0 = E_0.reshape(D ** 2, D, D, D ** 2)
E_0 = (E_0 + E_0.conj().transpose(3, 2, 1, 0)) / 2
E_0 = E_0.reshape((D ** 2, D ** 2, D ** 2))

C_0 = torch.tensor(C_0, dtype=dtype, device=device, requires_grad=False)
E_0 = torch.tensor(E_0, dtype=dtype, device=device, requires_grad=False)
C_0 = C_0 + 1e-6 * torch.randn_like(C_0)
E_0 = E_0 + 1e-6 * torch.randn_like(E_0)

Energy, C_0, E_0, error, \
        E_X_Id, E_Id_Z, E_X_Z, \
        E_Id_X_Id_X, E_Z_X_Z_X, E_Z_Id_Z_Id, \
        O_Z_Id, O_Id_X, O_Z_X \
    = LFT.expec_local_op(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                         C_0, E_0,
                         h_x=h_x, h_z=h_z, h_w=h_w
                         )

Energy = torch2np(Energy, device)
E_X_Id = torch2np(E_X_Id, device)
E_Id_Z = torch2np(E_Id_Z, device)
E_X_Z =  torch2np(E_X_Z, device)
E_Id_X_Id_X = torch2np(E_Id_X_Id_X, device)
E_Z_X_Z_X   = torch2np(E_Z_X_Z_X, device)
E_Z_Id_Z_Id = torch2np(E_Z_Id_Z_Id, device)
O_Z_Id = torch2np(O_Z_Id, device)
O_Id_X = torch2np(O_Id_X, device)
O_Z_X =  torch2np(O_Z_X, device)

###
C_0 = C_0 + 1e-6 * torch.randn_like(C_0)
E_0 = E_0 + 1e-6 * torch.randn_like(E_0)

xi, area_X_Id, perimeter_X_Id \
    = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                              eig_safe=0, C_0=C_0, E_0=E_0, O='X_Id'
                              )
    
xi, area_Id_Z, perimeter_Id_Z \
    = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                              eig_safe=0, C_0=C_0, E_0=E_0, O='Id_Z'
                              )
    
xi, area_X_Z,  perimeter_X_Z  \
    = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, 
                              eig_safe=0, C_0=C_0, E_0=E_0, O='X_Z')

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
