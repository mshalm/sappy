import torch 
from torch.autograd import Function, Variable
import torch.nn as nn
import numpy as np
torch.set_default_dtype(torch.double)


from sappy_solver import Solver

import time
import timeit
import math
import pdb

class SAPSolver(Function):
    
    @staticmethod
    def forward(ctx, J, q, eps):
        solver = Solver(q.shape[-1]//3,J.shape[-1])
        #print("test")
        if q.dim() == 2:
            l_numpy = np.zeros_like(q)
            v_numpy = np.zeros_like(J[...,0,:])
            
            for i, (Ji, qi) in enumerate(zip(J,q)):
                l_numpy[i, :], v_numpy[i, :] = solver.solve(Ji.numpy(),qi.numpy(), eps)
        else:
            l_numpy, v_numpy = solver.solve(J.numpy(),q.numpy(), eps)
        l = torch.from_numpy(l_numpy)
        v = torch.from_numpy(v_numpy)
        ctx.save_for_backward(J, q, l, v, torch.tensor(eps))
        return l
    
    @staticmethod    
    def backward(ctx, grad_l):
        J, q, l, v, eps = ctx.saved_tensors
        solver = Solver(l.shape[-1]//3,v.shape[-1])
        if l.dim() == 2:
            grad_J = np.zeros_like(J)
            grad_q = np.zeros_like(q)
            for i, (Ji, qi, li, vi, grad_li) in enumerate(zip(J, q, l, v, grad_l)):
                grad_J[i,...], grad_q[i,...] = solver.gradient(Ji.numpy(),
                                                 qi.numpy(),
                                                 li.numpy(),
                                                 vi.numpy(),
                                                 grad_li.numpy(),
                                                 eps.item())
        else:
            grad_J, grad_q = solver.gradient(J.numpy(),
                                             q.numpy(),
                                             l.numpy(),
                                             v.numpy(),
                                             grad_l.numpy(),
                                             eps.item())
        grad_J = torch.from_numpy(grad_J)
        grad_q = torch.from_numpy(grad_q)
        return grad_J, grad_q, None