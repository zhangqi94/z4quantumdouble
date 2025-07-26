import torch
import numpy as np

def safe_inverse(x, svd_safe=10**(-12)):
    return x/(x**2 + svd_safe)

class SVD(torch.autograd.Function):
      @staticmethod
      def forward(self, A):
          U, S, V = torch.linalg.svd(A)
          V= V.transpose(-2,-1).conj()
          self.save_for_backward(U, S, V)
          return U, S, V
 
      @staticmethod
      def backward(self, dU, dS, dV):
          U, S, V= self.saved_tensors
          Vt = V.t().conj()
          Ut = U.t().conj()
          M = U.size(0)
          N = V.size(0)
          NS = len(S)
 
          F = (S - S[:, None])
          F = safe_inverse(F)
          F.diagonal().fill_(0)
 
          G = (S + S[:, None])
          G.diagonal().fill_(np.inf)
          G = 1/G

          UdU = Ut @ dU
          VdV = Vt @ dV

          Su = (F+G)*(UdU-UdU.t().conj())/2
          Sv = (F-G)*(VdV-VdV.t().conj())/2

          dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
          if (M>NS):
              dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt
          if (N>NS):
              dA = dA + (U/S) @ dV.t().conj() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
          return dA

