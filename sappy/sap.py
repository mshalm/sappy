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
            for Ji, qi in zip(J,q):
                l_numpy, v_numpy = solver.solve(Ji.detach().numpy(),qi.detach().numpy(), eps)
        else:
            l_numpy, v_numpy = solver.solve(J.detach().numpy(),q.detach().numpy(), eps)
        l = torch.from_numpy(l_numpy)
        v = torch.from_numpy(v_numpy)
        ctx.save_for_backward(l, v, eps)
        return l
    
    @staticmethod    
    def backward(ctx, grad_l):

        P,q,l = ctx.saved_tensors
        return torch.zeros_like(P), torch.zeros_like(q)