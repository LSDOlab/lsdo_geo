# 4. Implicit Nonlinear Parameterization Framework

The defining innovation of `lsdo_geo` is formulating the parameterization mapping $P$ itself as an **inner nonlinear optimization problem** {cite:p}`fletcher2026implicit`. Rather than attempting to derive closed-form explicit functions for complex multi-component interactions, an optimization solver dynamically enforces application-tailored engineering variables and geometric feasibility constraints.

---

## 4.1. The Inner Parameterization Map ($P_i$)

Before defining the implicit optimization problem, an **inner parameterization mapping** $P_i$ is constructed to map a set of parameterization states $\boldsymbol{\alpha}_s \in \mathbb{R}^{n_{\alpha_s}}$ to the full geometric representation parameters $\boldsymbol{\alpha} \in \mathbb{R}^{n_\alpha}$:

$$P_i : \mathbb{R}^{n_{\alpha_s}} \to \mathbb{R}^{n_\alpha}$$

To handle complex systems with diverse components, `lsdo_geo` employs a **hierarchical deformational scheme**:

```
 ┌─────────────────────────────────────────────────────────────┐
 │ Level 3: B-Spline Curve Parameterization                   │
 │          States α_s = Control points of deformation curves  │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Evaluates along principal axis
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Level 2: Sectional Parameterization                         │
 │          Sectional translation, rotation, and stretch modes │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Controls lattice planes
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Level 1: Component Free-Form Deformation (FFD)              │
 │          Trivariate B-spline volume embedding geometry      │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Deforms embedded representation
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Central Geometry Representation (Control points α)          │
 └─────────────────────────────────────────────────────────────┘
```

1. **Level 1 (Component FFD):** Each distinct component (e.g., wing, fuselage, rotor boom) is embedded in an independent FFD volume. This completely decouples components and isolates local geometric resolution.
2. **Level 2 (Sectional Parameterization):** For each FFD volume, a principal parametric axis is assigned (e.g., spanwise for a wing, longitudinal for a fuselage). Control points are grouped into planar sections with defined deformation modes:
   - Rigid-body translations ($\Delta x, \Delta y, \Delta z$)
   - Rigid-body rotations (twist, sweep, dihedral angles)
   - In-plane chord and thickness scaling/stretching
3. **Level 3 (B-Spline Curve Regression):** Sectional parameters along the principal axis are parameterized using 1D B-spline curves. The parameterization states $\boldsymbol{\alpha}_s$ are the control points of these B-splines, allowing the user to enforce linearity, quadratic distributions, or smooth higher-order variations.
4. **Mixed States:** For rigid components (such as motor nacelles or propeller hubs), Level 1–3 can be bypassed in favor of direct 6-DOF rigid-body translation and rotation states, drastically reducing computational overhead.

---

## 4.2. Implicit Problem Formulation

Once the inner map $P_i$ is defined, the implicit parameterization problem is formulated. Given target engineering design variables $\mathbf{x}_g \in \mathbb{R}^{n_{xg}}$ specified by the outer optimizer, the geometry parameterization solver solves:

$$\begin{aligned}
\min_{\boldsymbol{\alpha}_s \in \mathbb{R}^{n_{\alpha_s}}} \quad & f_g(\boldsymbol{\alpha}_s) = \boldsymbol{\alpha}_s^T \mathbf{W} \boldsymbol{\alpha}_s \\
\text{s.t.} \quad & \mathbf{c}_{ge}\left(g^h(\cdot; P_i(\boldsymbol{\alpha}_s)), x_{ng}\right) = \mathbf{0} \\
& \mathbf{c}_{gi}\left(g^h(\cdot; P_i(\boldsymbol{\alpha}_s)), x_{ng}\right) \le \mathbf{0} \\
& \mathbf{c}_{gx}(\boldsymbol{\alpha}_s) = \hat{\mathbf{x}}_g(\boldsymbol{\alpha}_s) - \mathbf{x}_g = \mathbf{0}
\end{aligned}$$

where:
- $f_g(\boldsymbol{\alpha}_s)$ is the squared weighted norm of the states, with symmetric positive-definite weighting matrix $\mathbf{W}$. Penalizing specific states acts as a regularizer, prioritizing smooth deformations over severe distortion.
- $\mathbf{c}_{gx}(\boldsymbol{\alpha}_s) = \hat{\mathbf{x}}_g(\boldsymbol{\alpha}_s) - \mathbf{x}_g = \mathbf{0}$ enforces that the computed geometric properties $\hat{\mathbf{x}}_g$ (such as wing planform area, aspect ratio, or rotor radius) match the outer optimizer's prescribed values $\mathbf{x}_g$.
- $\mathbf{c}_{ge}$ and $\mathbf{c}_{gi}$ enforce physical and kinematic constraints across components.

### Generalization of Existing Methods
This implicit formulation generalizes prior constraint-based schemes:
- Setting $\mathbf{W} = \mathbf{I}$ and targeting discrete points recovers classical **direct manipulation**.
- Setting $\mathbf{W} = \mathbf{K}$ (structural stiffness matrix) recovers **PDE-based strain energy minimization**.
- Supporting nonlinear algebraic functions $\hat{\mathbf{x}}_g$ extends parameterization to true engineering design variables.

---

## 4.3. Multi-Component Kinematic Coupling

Crucially, the implicit system is solved as a **single unified system** across all components rather than isolated sub-problems. 

This enables exact enforcement of **inter-component kinematic constraints**:
- Ensuring lifting propeller booms stay attached to the wing upper/lower surfaces as the wing sweeps, tapers, and flexes.
- Enforcing wing-fuselage root continuity and fairing alignment.
- Constraining rotor spacing and tail moment arms across multi-body assemblies.

---

## 4.4. Constraint Enforcement Strategies

### Equality Constraints: Lagrange Multipliers
Geometric equality constraints ($\mathbf{c}_{ge} = \mathbf{0}$ and $\mathbf{c}_{gx} = \mathbf{0}$) are enforced exactly using the **method of Lagrange multipliers**. Exact satisfaction guarantees that analysis solvers never receive broken or disjoint geometric models.

### Inequality Constraints: Multiplier-Based Interior-Point Method
Enforcing geometric inequality constraints ($\mathbf{c}_{gi} \le \mathbf{0}$, such as non-interference or minimum clearance) in an inner solver requires extreme care:
- Standard active-set or projection methods introduce non-differentiable slope discontinuities (kinks) into the outer optimization landscape.
- Simple exterior penalty methods cause severe ill-conditioning and allow geometric violations that crash mesh generators.

`lsdo_geo` utilizes an **interior-point formulation with multiplier updates**. This preserves continuous differentiability for outer gradient evaluation while strictly guaranteeing geometric feasibility whenever the inner solver converges.

### Weak Variable Enforcement with Outer Consistency Constraints
To guarantee that the inner parameterization problem is always feasible (even if the outer optimizer requests conflicting design variables):
1. Conflicted design variables can be designated as **weakly enforced** by converting their equality into a quadratic penalty in $f_g$.
2. The inner solver has the freedom to violate the target variable to strictly preserve geometric feasibility (preventing overlapping rotors or negative thickness).
3. **Consistency constraints** are placed on the outer optimizer:

   $$\left| \hat{\mathbf{x}}_g(\boldsymbol{\alpha}_s^*) - \mathbf{x}_g \right| \le \epsilon_{\text{tol}}$$

   As the outer optimization converges, these constraints drive the optimizer to choose design points where the desired variables and feasibility constraints naturally align.

