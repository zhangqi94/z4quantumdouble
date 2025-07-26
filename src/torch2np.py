import torch

def torch2np(A,device):
    if device=='cpu':
        B=A.detach().numpy()
    elif device=='cuda':
        B=A.to('cpu').detach().numpy()
    return B
