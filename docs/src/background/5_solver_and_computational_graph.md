# Solvers, Adjoints & Computational Graph Synergy

Executing an inner optimization problem at every geometry evaluation introduces two critical mathematical requirements:
1. Solving the nested nonlinear system rapidly and robustly.
2. Accurately propagating total derivatives through the solved implicit state to compute analytic adjoint sensitivities for the outer MDO optimizer {cite:p}`fletcher2026implicit`.

---

## 5.1. KKT Optimality System & Exact Newton Solver

The first-order necessary optimality conditions (Karush-Kuhn-Tucker system) for the equality-constrained implicit parameterization sub-problem form the system residual $\mathbf{R}$:

$$\mathbf{R}(\boldsymbol{\alpha}_s, \boldsymbol{\lambda}) = \begin{bmatrix}
\nabla_{\boldsymbol{\alpha}_s} \mathcal{L} \\
\mathbf{c}(\boldsymbol{\alpha}_s)
\end{bmatrix} = \begin{bmatrix}
2 \mathbf{W} \boldsymbol{\alpha}_s + \mathbf{J}_{\mathbf{c}}^T \boldsymbol{\lambda} \\
\mathbf{c}(\boldsymbol{\alpha}_s)
\end{bmatrix} = \mathbf{0}$$

where $\boldsymbol{\lambda}$ is the vector of Lagrange multipliers and $\mathbf{J}_{\mathbf{c}} = \frac{\partial \mathbf{c}}{\partial \boldsymbol{\alpha}_s}$ is the constraint Jacobian.

### Exact Newton Method
`lsdo_geo` solves this system using an **exact Newton-Raphson solver**:

$$\begin{bmatrix}
\boldsymbol{\alpha}_s \\
\boldsymbol{\lambda}
\end{bmatrix}_{k+1} = \begin{bmatrix}
\boldsymbol{\alpha}_s \\
\boldsymbol{\lambda}
\end{bmatrix}_k - \left[ \mathbf{K}_{\text{KKT}} \right]^{-1} \mathbf{R}_k$$

where $\mathbf{K}_{\text{KKT}}$ is the symmetric Karush-Kuhn-Tucker matrix:

$$\mathbf{K}_{\text{KKT}} = \begin{bmatrix}
2 \mathbf{W} + \sum_i \lambda_i \nabla_{\boldsymbol{\alpha}_s}^2 c_i & \mathbf{J}_{\mathbf{c}}^T \\
\mathbf{J}_{\mathbf{c}} & \mathbf{0}
\end{bmatrix}$$

### Fast Convergence Properties
- **Linear Constraints $\implies$ 1-Step Solve:** Because the inner objective $f_g = \boldsymbol{\alpha}_s^T \mathbf{W} \boldsymbol{\alpha}_s$ is purely quadratic, whenever all geometric constraints are linear affine functions of the states, the Hessian is constant and the Newton solver converges in **exactly one linear solve**!
- **Nonlinear Constraints:** For nonlinear geometric constraints (e.g., planform area, aspect ratio), a backtracking line search is employed to guarantee global convergence.

---

## 5.2. Adjoint Sensitivity Analysis Through the Solver

In gradient-based MDO, the outer optimizer requires total derivatives of disciplinary objective and constraint functions $F$ with respect to geometric design variables $\mathbf{x}_g$:

$$\frac{d F}{d \mathbf{x}_g} = \frac{\partial F}{\partial \mathbf{x}_g} + \frac{\partial F}{\partial \boldsymbol{\alpha}} \frac{d \boldsymbol{\alpha}}{d \mathbf{x}_g}$$

Because $\boldsymbol{\alpha} = P_i(\boldsymbol{\alpha}_s^*)$ depends on the implicitly converged state $\boldsymbol{\alpha}_s^*$, computing $\frac{d \boldsymbol{\alpha}_s^*}{d \mathbf{x}_g}$ directly via forward sensitivities would be prohibitively expensive.

### Implicit Function Theorem & Adjoint System
By applying the Implicit Function Theorem to the KKT residual $\mathbf{R}(\boldsymbol{\alpha}_s^*, \boldsymbol{\lambda}^*; \mathbf{x}_g) = \mathbf{0}$:

$$\frac{\partial \mathbf{R}}{\partial \begin{bmatrix} \boldsymbol{\alpha}_s \\ \boldsymbol{\lambda} \end{bmatrix}} \frac{d \begin{bmatrix} \boldsymbol{\alpha}_s^* \\ \boldsymbol{\lambda}^* \end{bmatrix}}{d \mathbf{x}_g} = - \frac{\partial \mathbf{R}}{\partial \mathbf{x}_g}$$

Notice that the left-hand matrix is **identically the KKT matrix $\mathbf{K}_{\text{KKT}}$** already assembled and factored during the Newton solve!

In the adjoint formulation, we solve the linear adjoint system:

$$\mathbf{K}_{\text{KKT}}^T \boldsymbol{\psi} = \begin{bmatrix} \left( \frac{\partial F}{\partial \boldsymbol{\alpha}} \frac{\partial P_i}{\partial \boldsymbol{\alpha}_s} \right)^T \\ \mathbf{0} \end{bmatrix}$$

The total geometric derivative is obtained via a single matrix-vector product:

$$\frac{d F}{d \mathbf{x}_g} = - \boldsymbol{\psi}^T \frac{\partial \mathbf{R}}{\partial \mathbf{x}_g}$$

This makes the computational cost of computing geometric gradients virtually independent of the number of design variables.

---

## 5.3. Synergy with Graph-Based Modeling (CSDL)

Constructing $\mathbf{K}_{\text{KKT}}$ requires computing second-order derivatives $\nabla_{\boldsymbol{\alpha}_s}^2 c_i$ of geometric constraints. Hand-deriving these matrices for complex assemblies is error-prone and rigid.

`lsdo_geo` integrates tightly with **CSDL (Computational System Design Language)** {cite:p}`gandarillas2024graph`:
1. **Automated Analytic Derivatives:** CSDL automatically generates reverse-mode vector-Jacobian products (VJPs) and Jacobian-vector products (JVPs) across the computational graph, yielding exact first and second derivatives without finite differences.
2. **JAX/C++ Compilation:** The entire implicit solver loop compiles into optimized CPU/GPU kernels via JAX or CSDL-Alpha backends.

---

## 5.4. Graph Transformations & Symbolic Constraint Backtracking

To maximize Newton solver speed, `lsdo_geo` implements symbolic **graph transformations** that eliminate unnecessary nonlinearities from the implicit system {cite:p}`fletcher2026implicit`.

Many engineering definitions are expressed as a chain of linear kinematic transformations followed by a few terminal nonlinear operations:

$$v(\boldsymbol{\alpha}_s) = M_1 \circ M_2 \circ \dots \circ M_{n_{\text{op}}}(\boldsymbol{\alpha}_s)$$

where the constraint is $v(\boldsymbol{\alpha}_s) - v^* = 0$.

If the trailing operations $M_1, \dots, M_l$ are algebraically invertible, the constraint can be transformed by **backtracking**:

$$\hat{v}(\boldsymbol{\alpha}_s) = M_{l}^{-1} \circ \dots \circ M_1^{-1}(v(\boldsymbol{\alpha}_s))$$
$$\hat{v}^* = M_{l}^{-1} \circ \dots \circ M_1^{-1}(v^*)$$

Because $v^*$ is a constant design target, evaluating the inverted operations on $v^*$ produces a constant target $\hat{v}^*$. 

**Benefit:** Non-invertible or nonlinear operations at the tail of the constraint chain are shifted out of the implicit solver loop and evaluated once upfront. For aircraft wing parameters (sweep, aspect ratio, taper ratio), this transformation frequently **linearizes the constraint**, reducing Newton iterations from 6–8 down to a single linear solve!

