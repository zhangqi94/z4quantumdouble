import numpy as np
import torch
from ncon import ncon
import argparse
import scipy

# parser = argparse.ArgumentParser()
# parser.add_argument("-D", type=int, default=4)

# args = parser.parse_args() 
# D=args.D

D=8
q=4
Id=np.eye(q)
Id_D=np.eye(D)

P1=(ncon([Id_D,Id_D,Id_D,Id_D,Id],[[-1,-6],[-2,-7],[-3,-8],[-4,-9],[-5,-10]])+\
    ncon([Id_D,Id_D,Id_D,Id_D,Id],[[-2,-6],[-3,-7],[-4,-8],[-1,-9],[-5,-10]])+\
    ncon([Id_D,Id_D,Id_D,Id_D,Id],[[-3,-6],[-4,-7],[-1,-8],[-2,-9],[-5,-10]])+\
    ncon([Id_D,Id_D,Id_D,Id_D,Id],[[-4,-6],[-1,-7],[-2,-8],[-3,-9],[-5,-10]]))/4\

P2=(ncon([Id_D,Id_D,Id_D,Id_D,Id],[[-1,-6],[-2,-7],[-3,-8],[-4,-9],[-5,-10]])+\
    ncon([Id_D,Id_D,Id_D,Id_D,Id],[[-3,-6],[-2,-7],[-1,-8],[-4,-9],[-5,-10]]))/2

P1=P1.reshape((q*D**4,q*D**4))
P2=P2.reshape((q*D**4,q*D**4))
print('if P1 and P2 commute=',np.linalg.norm(P1@P2-P2@P1))
P=P1@P2
print('if P is hermitian=',np.linalg.norm(P-P.T))
[e,v]=scipy.linalg.eigh(P)
basis=[]

for n0 in range(len(e)):
     if np.abs(e[n0])>0.5:
         if np.max(np.abs((v[:,n0]).imag))>10**-10:
             print('basis is complex')
         else:
             basis.append(np.real(v[:,n0]))
print('D=',D,'len_basis=',len(basis))
M=np.zeros((D**4*q,len(basis)))
for n0 in range(len(basis)):
    M[:,n0]=basis[n0]

[Q,R]=np.linalg.qr(M)
for n0 in range(len(basis)):
    basis[n0]=Q[:,n0]

overlap=np.zeros((len(basis),len(basis)))
for i in range(len(basis)):
     for j in range(len(basis)):
         overlap[i,j]=np.dot(basis[i],basis[j])
print('diff_overlap_Id=',np.linalg.norm(overlap-np.eye(len(basis))))

     
data={'basis':basis,'len_basis':len(basis)}
torch.save(data,'Square_lattice_sym_basis_q='+str(q)+'_D='+str(D)+'.pt')
# test that T is correct
para=np.random.rand(1,len(basis))
basis=data.get('basis')
T=np.zeros((1,q*D**4))
for n0 in range(len(basis)):
    T=T+para[0,n0]*basis[n0]

T=T.reshape((D,D,D,D,q))
temp1=np.linalg.norm(T-T.transpose([3,0,1,2,4]))
temp2=np.linalg.norm(T-T.transpose([0,3,2,1,4]))
print('test if the tensor has square lattice symmetry',temp1+temp2)

 

