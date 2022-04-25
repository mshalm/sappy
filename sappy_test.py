from sappy import SAPSolver
from diffqcqp import LCQPFn2
import torch

from time import time
import timeit

NC = 7
N = 3 * NC
#NV = N
NV = 3 * 3 + 6
eps = 1e-4
ss = SAPSolver()
lc = LCQPFn2()
T = 1000
J_all = torch.rand((T,N,NV))
q_all = torch.rand((T,N))
Q_l_all = torch.bmm(J_all,J_all.transpose(-1,-2))
Q_l_all += eps * torch.eye(N).unsqueeze(0)

TEST_LQCP = False



def sappy_run():
	l = ss.apply(J_all, q_all, eps)
	#for (J,q) in zip(J_all, q_all):
	#	l = ss.apply(J, q, eps)

def lcqp_run():
	for (Q_l,q) in zip(Q_l_all, q_all):
		q_l = q.unsqueeze(0).squeeze()

		Q_l = Q_l.unsqueeze(0)
		q_l = q_l.unsqueeze(-1).unsqueeze(0)

		l_l = lc.apply(Q_l,q_l,torch.rand(q_l.shape),1e-7,100000)

for i in range(0):
	sappy_run()
	lcqp_run()

t_sap = timeit.timeit(lambda: sappy_run(), number = 1)/T
t_lcqp = 0.
if TEST_LQCP:
	t_lcqp = timeit.timeit(lambda: lcqp_run(), number = 1)/T
print(t_sap,t_lcqp)


reorder_mat = torch.zeros((N,N))
for i in range(NC):
	reorder_mat[i][3*i + 2] = 1
	reorder_mat[NC + 2*i][3*i] = 1
	reorder_mat[NC + 2*i + 1][3*i + 1] = 1

#print(reorder_mat.t())
#print(reorder_mat.t().mm(reorder_mat))

J = J_all[0]
q = q_all[0]

Q = J.mm(J.t())

l_nominal = torch.linalg.solve(Q + eps * torch.eye(N),-q)
print("nominal: ", l_nominal)

ss = SAPSolver()
l = ss.apply(J,q, eps)
print(l)
print(0.5 * l.unsqueeze(0).mm((Q + eps * torch.eye(N)).mm(l.unsqueeze(1)) + 2 * q.unsqueeze(1)))

lc = LCQPFn2()
Q_l = reorder_mat.mm(Q).mm(reorder_mat.t())
Q_l += eps * torch.eye(N)

q_l = q.unsqueeze(0).mm(reorder_mat.t()).squeeze()

Q_l = Q_l.unsqueeze(0)
q_l = q_l.unsqueeze(-1).unsqueeze(0)

l_l = lc.apply(Q_l,q_l,torch.zeros_like(q_l),1e-7,100000)
l_l = reorder_mat.t().mm(l_l.squeeze(0)).squeeze()
print(l_l)
print(0.5 * l_l.unsqueeze(0).mm((Q + eps * torch.eye(N)).mm(l_l.unsqueeze(1)) + 2 * q.unsqueeze(1)))

