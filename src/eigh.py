import numpy as np
import torch

#device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_default_device(device)
 
def safe_inverse(x, eig_safe=10**(-12)):

      device=x.device.type
      torch.set_default_device(device) 
      return x/(x**2 + torch.rand(1,dtype=torch.double)*eig_safe)



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
           F = safe_inverse(F)
           F.diagonal().fill_(0)     
           vt = v.t()
           vdv = vt@dv
           return v@(torch.diag(dw) + F*(vdv-vdv.t())/2) @vt

