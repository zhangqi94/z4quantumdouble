import time
import torch
import loss_fn_torch_sym_AT as LFT
import numpy as np
from INI_TEN_AT import initialize_tensor
from ncon import ncon
import sys
from T2para_square_lattice_sym_potts import T2para_torch
from T2para_square_lattice_sym_potts import para2T_torch
import os
import argparse
from torch2np import torch2np

start = time.time()
dtype = torch.double
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

parser = argparse.ArgumentParser()

parser.add_argument("-D", type=int, default=4)
parser.add_argument("-chi", type=int, default=100)
parser.add_argument('-use_check_point', type=int, default=1)
parser.add_argument('-abs_path', type=str, default='')
parser.add_argument('-h_x', type=float, default=0)
parser.add_argument('-is_reverse', type=int, default=0)


args = parser.parse_args()
use_check_point = args.use_check_point
abs_path = args.abs_path
is_reverse = args.is_reverse
h_x= args.h_x
max_CTMRG_steps = 300
CTMRG_tol = 1E-13
max_iter = 15
tol_grad = 10 ** (-6)
eig_safe = 10 ** (-12)


D = args.D
chi = args.chi


# List_h = np.arange(0.1, 2.1, 0.1)
List_h = np.arange(0.7, 1.1, 0.01)
if is_reverse == 1:
    List_h = List_h[::-1]


filename = abs_path + 'AT_iPEPS_h_x='+str(h_x)+'_h=' + str(List_h[0]) + '_to_' + str(List_h[-1]) \
    + '_D=' + str(D) + '_chi=' + str(chi) + '.pt'
filename_txt = filename[0:-3] + '.txt'

List_T = [0 for x in range(len(List_h))]
List_Energy = [0 for x in range(len(List_h))]
List_Grad_norm = [0 for x in range(len(List_h))]
List_E_X_Id = [0 for x in range(len(List_h))]
List_E_Id_Z = [0 for x in range(len(List_h))]
List_E_X_Z = [0 for x in range(len(List_h))]
List_E_Id_X_Id_X= [0 for x in range(len(List_h))]
List_E_Z_X_Z_X= [0 for x in range(len(List_h))]
List_E_Z_Id_Z_Id= [0 for x in range(len(List_h))]
List_O_Z_Id= [0 for x in range(len(List_h))]
List_O_Id_X= [0 for x in range(len(List_h))]
List_O_Z_X= [0 for x in range(len(List_h))]
List_xi = [0 for x in range(len(List_h))]
List_area_X_Id = [0 for x in range(len(List_h))]
List_perimeter_X_Id = [0 for x in range(len(List_h))]
List_area_Id_Z = [0 for x in range(len(List_h))]
List_perimeter_Id_Z = [0 for x in range(len(List_h))]
List_area_X_Z = [0 for x in range(len(List_h))]
List_perimeter_X_Z = [0 for x in range(len(List_h))]

if os.path.exists(filename_txt):
    os.remove(filename_txt)
else:
    with open(filename_txt, 'w') as f:
        sys.stdout = f
        print("The file does not exist")
        print(filename)
        print('device=', device)
        sys.stdout = sys.__stdout__
if is_reverse == 1:
    phase = 'FM'
else:
    phase = 'PM'

delta = 10 ** (-2)
T = initialize_tensor(D, delta, phase)



for n0 in range(len(List_h)):
    h = List_h[n0]
    h_z=h
    h_w=h
    params = T2para_torch(T)
    optimizer = torch.optim.LBFGS(params, 
                                  tolerance_grad=1e-05, 
                                  line_search_fn='strong_wolfe'
                                  )
    dtch_iso = 0
    def closure():
        global dtch_iso
        # print(dtch_iso)
        t_CTMRG_0 = time.time()
        ini_chi = chi

        C_0 = ncon([T.conj(), T], [[2, 3, -1, -2, 1], [2, 3, -3, -4, 1]])
        C_0 = C_0.reshape((D ** 2, D ** 2))
        C_0 = (C_0 + C_0.conj().transpose()) / 2
        E_0 = ncon([T.conj(), T], [[2, -1, -3, -5, 1], [2, -2, -4, -6, 1]])
        E_0 = E_0.reshape(D ** 2, D, D, D ** 2)
        E_0 = (E_0 + E_0.conj().transpose(3, 2, 1, 0)) / 2
        E_0 = E_0.reshape((D ** 2, D ** 2, D ** 2))

        C_0 = torch.tensor(C_0, requires_grad=False)
        E_0 = torch.tensor(E_0, requires_grad=False)
        optimizer.zero_grad()

        loss, C, E, CTMRG_error= LFT.loss_fn(params, h_x, h_z, h_w, 
                                             D, chi, max_CTMRG_steps, CTMRG_tol, C_0, E_0,
                                             eig_safe, use_check_point, dtch_iso=dtch_iso,
                                             filename_txt=filename_txt
                                             )
        # print('loss ', loss)
        
        t_CTMRG_1 = time.time()
        t_grad_0 = time.time()
        loss.backward()
        t_grad_1 = time.time()
        temp = torch.zeros((1, len(params)))
        for i in range(len(params)):
            temp[0, i] = (params[i]).grad

        Grad_norm = torch.norm(temp)
        if Grad_norm == torch.tensor(float('nan')) or Grad_norm > 1000:
            dtch_iso = 1

            with open(filename_txt, 'a') as f:
                sys.stdout = f
                print("Grad norm is unstable")
                sys.stdout = sys.__stdout__
            # eig_safe_2=10**(-8.5)
            optimizer.zero_grad()

            loss, C, E, CTMRG_error= LFT.loss_fn(
                params, h_x, h_z, h_w, D, chi, max_CTMRG_steps, CTMRG_tol, C_0, E_0,
                eig_safe, use_check_point, dtch_iso=dtch_iso,
                filename_txt=filename_txt)
            t_CTMRG_1 = time.time()
            t_grad_0 = time.time()
            loss.backward()
            t_grad_1 = time.time()
            temp = torch.zeros((1, len(params)))
            for i in range(len(params)):
                temp[0, i] = (params[i]).grad
            Grad_norm = torch.norm(temp)
            Grad_norm = torch2np(Grad_norm, device)

        with open(filename_txt, 'a') as f:
            sys.stdout = f
            print('h_x=', h_x,
                  'h_z=', h_z,
                  'h_w=', h_w, 
                  'time_grad=', t_grad_1 - t_grad_0, 
                  'time_CTMRG=', t_CTMRG_1 - t_CTMRG_0, 
                  'CTMRG_error=', CTMRG_error, 
                  'Energy=', loss.item(), 
                  'norm_grad=', Grad_norm, 
                  '\n'
                  'chi=', (C[0]).shape[0], 
                  ', D=', D, 
                  ', dtch_iso=', dtch_iso
                  )
            sys.stdout = sys.__stdout__
        
        print('h_x=', h_x,
                'h_z=', h_z,
                'h_w=', h_w, 
                'time_grad=', t_grad_1 - t_grad_0, 
                'time_CTMRG=', t_CTMRG_1 - t_CTMRG_0, 
                'CTMRG_error=', CTMRG_error, 
                'Energy=', loss.item(), 
                'norm_grad=', Grad_norm, 
                '\n'
                'chi=', (C[0]).shape[0], 
                ', D=', D, 
                ', dtch_iso=', dtch_iso
                )
        return loss


    epoch = 0
    Grad_norm = 1
    while epoch < max_iter and Grad_norm > tol_grad:

        with open(filename_txt, 'a') as f:
            sys.stdout = f
            print('h_x,h_z,h_w=', h_x,h_z,h_w, 'epoch=', epoch)
            sys.stdout = sys.__stdout__
        optimizer.step(closure)

        Energy = closure()
        temp = torch.zeros((1, len(params)))
        for i in range(len(params)):
            temp[0, i] = (params[i]).grad

        Grad_norm = torch.norm(temp)
        #    if Grad_norm<tol_grad:
        #        break
        epoch = epoch + 1

    T = para2T_torch(params, q, D)
    T = T / torch.norm(T)
    T = torch2np(T, device)
    Energy = torch2np(Energy, device)
    Grad_norm = torch2np(Grad_norm, device)
    Energy, C_0, E_0, error, E_X_Id, E_Id_Z, E_X_Z, E_Id_X_Id_X, E_Z_X_Z_X, E_Z_Id_Z_Id, O_Z_Id, O_Id_X, O_Z_X \
        = LFT.expec_local_op(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol)

    xi,area_X_Id,perimeter_X_Id=LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, eig_safe=0,O='X_Id')
    xi, area_Id_Z, perimeter_Id_Z = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, eig_safe=0, O='Id_Z')
    xi, area_X_Z, perimeter_X_Z = LFT.Expec_disorder_para(torch.tensor(T), chi, max_CTMRG_steps, CTMRG_tol, eig_safe=0, O='X_Z')
    List_Energy[n0] = Energy
    List_Grad_norm[n0] = Grad_norm
    List_T[n0] = T
    List_E_X_Id[n0] = E_X_Id
    List_E_Id_Z[n0] = E_Id_Z
    List_E_X_Z[n0] = E_X_Z
    List_E_Id_X_Id_X[n0] = E_Id_X_Id_X
    List_E_Z_X_Z_X[n0] = E_Z_X_Z_X
    List_E_Z_Id_Z_Id[n0] = E_Z_Id_Z_Id
    List_O_Z_Id[n0] = O_Z_Id
    List_O_Id_X[n0] = O_Id_X
    List_O_Z_X[n0] = O_Z_X
    List_xi[n0] = xi
    List_area_X_Id[n0] = area_X_Id
    List_perimeter_X_Id[n0] = perimeter_X_Id
    List_area_Id_Z[n0] = area_Id_Z
    List_perimeter_Id_Z[n0] = perimeter_Id_Z
    List_area_X_Z[n0] = area_X_Z
    List_perimeter_X_Z[n0] = perimeter_X_Z

    # print('Energy=',Energy,'O_Z=',np.sqrt(O_Z_real**2+O_Z_imag**2),'area=',area,'perimeter=',perimeter)
    dict_PEPS = {"List_T": List_T,
                 "List_Energy": List_Energy,
                 "List_Grad_norm": List_Grad_norm,
                 "D": D,
                 "chi": chi,
                 "List_h": List_h,
                 "List_E_X_Id": List_E_X_Id,
                 "List_E_Id_Z": List_E_Id_Z,
                 "List_E_X_Z": List_E_X_Z,
                 "List_E_Id_X_Id_X": List_E_Id_X_Id_X,
                 "List_E_Z_X_Z_X": List_E_Z_X_Z_X,
                 "List_E_Z_Id_Z_Id": List_E_Z_Id_Z_Id,
                 "List_O_Z_Id": List_O_Z_Id,
                 "List_O_Id_X": List_O_Id_X,
                 "List_O_Z_X": List_O_Z_X,
                 "List_xi": List_xi,
                 "List_area_X_Id": List_area_X_Id,
                 "List_perimeter_X_Id": List_perimeter_X_Id,
                 "List_area_Id_Z": List_area_Id_Z,
                 "List_perimeter_Id_Z": List_perimeter_Id_Z,
                 "List_area_X_Z": List_area_X_Z,
                 "List_perimeter_X_Z": List_perimeter_X_Z,
                 "h_x": h_x}
    torch.save(dict_PEPS, filename)

end = time.time()
with open(filename_txt, 'a') as f:
    sys.stdout = f
    print('total_time=' + str(end - start) + 's')
    sys.stdout = sys.__stdout__
