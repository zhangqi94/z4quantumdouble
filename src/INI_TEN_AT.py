import numpy as np

def initialize_tensor(D,delta,phase):
    T_temp=FPT(phase)
    T=np.zeros((D,D,D,D,4))
    T[0,0,0,0,:]=T_temp
    T=T+delta*np.random.rand(int(D),int(D),int(D),int(D),4)
    return T

def FPT(phase):
    if phase=='FM':
        T=np.zeros((1,4))
        T[0,0]=1
        T=T.reshape(1,1,1,1,4)
    elif phase=='PM':
        T=(np.ones((1,4))).reshape(1,1,1,1,4)
    return T