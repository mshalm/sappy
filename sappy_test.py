from sappy import SAPSolver
import torch

from time import time
import timeit
import pdb

NC = 7
N = 3 * NC
#NV = N
NV = 3 * 3 + 6
eps = 1e-4
ss = SAPSolver()
T = 1000
J_all = torch.rand((T,N,NV), requires_grad = True)
q_all = torch.rand((T,N), requires_grad = True)

TEST_LQCP = True

reorder_mat = torch.zeros((N,N))
for i in range(NC):
	reorder_mat[i][3*i + 2] = 1
	reorder_mat[NC + 2*i][3*i] = 1
	reorder_mat[NC + 2*i + 1][3*i + 1] = 1

def sappy_run(J_run, q_run):
	l = ss.apply(J_run, q_run, eps)
	return l
	#for (J,q) in zip(J_all, q_all):
	#	l = ss.apply(J, q, eps)

def sappy_run_diff(J_run, q_run):
	l = ss.apply(J_run, q_run, eps)
	return l.mean().backward()
	#for (J,q) in zip(J_all, q_all):
	#	l = ss.apply(J, q, eps)

t_sap = timeit.timeit(lambda: sappy_run(J_all, q_all), number = 1)/T
t_sap_diff = timeit.timeit(lambda: sappy_run_diff(J_all, q_all), number = 1)/T
print('')
print(f'Speed:')
print(f'time per forward eval: {t_sap}')
print(f'time per forward+backward eval: {t_sap_diff}')
print(f'time per backward eval: {t_sap_diff - t_sap}')
print(f'backward eval as fraction of forward: {(t_sap_diff - t_sap)/t_sap}')
print('')

q_all.grad = torch.zeros_like(q_all.grad)
J_all.grad = torch.zeros_like(J_all.grad)

q_all_orig = q_all.clone().detach()
J_all_orig = J_all.clone().detach()
Q = J_all[0].mm(J_all[0].t())
LR = 1e-4
ss = SAPSolver()
l = sappy_run(J_all[0].unsqueeze(0), q_all[0].unsqueeze(0)).squeeze()
loss = l.mean()
lsave = loss.clone()
loss.backward()
sap_grad_q = q_all.grad[0].clone()
sap_grad_J = J_all.grad[0].clone()
with torch.no_grad():
	q_all[0] += LR * sap_grad_q
	J_all[0] += LR * sap_grad_J
q_all.grad = torch.zeros_like(q_all.grad)
J_all.grad = torch.zeros_like(J_all.grad)
loss_new = sappy_run(J_all[0].unsqueeze(0), q_all[0].unsqueeze(0)).mean().detach()
dloss = loss_new -loss
linearization_drop = LR * (sap_grad_J.norm()**2 + sap_grad_q.norm() ** 2)
err = (dloss - linearization_drop).abs()/(0.5 * (dloss.abs() + linearization_drop.abs()))

print('')
print(f'Gradient accuracy:')
print(f'change in loss after step: {dloss}')
print(f'gradient inner product with change in parameters: {linearization_drop}')
print(f'relative error: {err}')
print('')

