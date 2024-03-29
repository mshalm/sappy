# PyTorch solver for Lorentz Cone-Constrained Quadratic Programs
## Overview
PyTorch implementation (via Eigen/C++ bindings) of a solver for strictly convex Lorentz cone-constrained Quadratic Programs (LCQP's), equivalent to strongly monotone Second Order Linear Complementarity Problems (SOLCP's), based on Alejandro Castro et al.'s [Semi-Analytic Primal Solver](https://arxiv.org/pdf/2110.10107.pdf), developed by Toyota Research Institute.
## Interface
Problems are sovled with decision variables $l = [l_{x_1},l_{y_1},l_{z_1},\dots, l_{x_k},l_{y_k},l_{z_k}] \in \mathbb R^{3k}$, which must lie in the Lorentz ("ice cream") cone:
```math
\mathcal L = \left\{l : l_{z_i} \geq \left\lVert \begin{bmatrix} l_{x_i} \\ l_{y_i} \end{bmatrix} \right\rVert_2 \right\}\,.
```
Given problem data $J,q$ and tolerance $\varepsilon > 0$, `sappy` solves the following equivalent LCQP/SOLCP pair for $l$:
```math
\arg\min_l \frac{1}{2}l^t\left(JJ^T + \varepsilon I\right)l + q \cdot l = \left\{l : \mathcal L \ni \left(JJ^T + \varepsilon I\right)l + q \perp l \in \mathcal L \right\}\,.
```

The solver interface is as follows:
```
from sappy import SAPSolver
solver = SAPSolver()
l = solver.apply(J, q, eps)
```
The interface supports one dimension of batching, such that `J,q` can be `torch.Tensor`'s of shapes `(n, 3k), (3k,)` or `(n_batch, n, 3k), (n_batch, 3k,)`.

`solver.apply` supports reverse-mode differentiation via sensitivity analysis.
## installation
`pip install git+https://github.com/mshalm/sappy.git`
