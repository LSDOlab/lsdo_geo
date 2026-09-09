# 3. Mathematical Formulation of Shape Optimization

To establish a rigorous theoretical foundation for geometry parameterization, this section formulates the progression from the continuous, infinite-dimensional shape optimization problem to the discretized and parameterized formulations {cite:p}`fletcher2026implicit`.

---

## 3.1. Intrinsic Shape Optimization Problem

In its most general form, shape optimization seeks to find an optimal geometric spatial configuration that minimizes a multidisciplinary cost objective while satisfying physical and operational constraints.

Let:
- $\Omega_i = [0, 1]^{d_r}$ be a reference domain of dimension $d_r$ (typically $d_r = 1$ for curves, $d_r = 2$ for surfaces, $d_r = 3$ for volumes).
- $\Omega = \bigsqcup_{i=1}^{n_f} \Omega_i$ be the composite parametric domain over which a geometry constructed as a collection of $n_f$ functions is defined.
- $\mathcal{G} = \{ g : \Omega \to \mathbb{R}^d \}$ be the continuous function space of all admissible geometries mapping from the parametric domain to $d$-dimensional physical space (typically $d = 3$).

For an MDO system, let $x_{ng} \in \mathbb{R}^{n_{xng}}$ denote the vector of non-geometric design variables (such as operational altitude, flight speed, battery state of charge, or controller gains). The **intrinsic continuous shape optimization problem** is defined as:

$$\begin{aligned}
\min_{g \in \mathcal{G}, \, x_{ng} \in \mathbb{R}^{n_{xng}}} \quad & f(g, x_{ng}) \\
\text{s.t.} \quad & c_{ge}(g, x_{ng}) = \mathbf{0} \\
& c_{gi}(g, x_{ng}) \le \mathbf{0} \\
& c_{ng}(x_{ng}) = \mathbf{0}
\end{aligned}$$

where:
- $f : \mathcal{G} \times \mathbb{R}^{n_{xng}} \to \mathbb{R}$ is the system objective function (e.g., total vehicle mass, fuel burn, energy consumption).
- $c_{ge} : \mathcal{G} \times \mathbb{R}^{n_{xng}} \to \mathbb{R}^{m_{ge}}$ are geometric equality constraints (e.g., surface closure, symmetry, kinematic joint continuity).
- $c_{gi} : \mathcal{G} \times \mathbb{R}^{n_{xng}} \to \mathbb{R}^{m_{gi}}$ are geometric inequality constraints (e.g., internal fuel tank volume, minimum component clearance, spar thickness).
- $c_{ng} : \mathbb{R}^{n_{xng}} \to \mathbb{R}^{m_{ng}}$ are non-geometric disciplinary constraints (e.g., stress limits, stall margins, battery thermal ceilings).

---

## 3.2. Discretized Shape Optimization Problem

The continuous function space $\mathcal{G}$ is infinite-dimensional and cannot be directly solved computationally. To execute numerical analysis and optimization, a finite-dimensional geometry representation must be chosen.

Let:

$$\mathcal{G}^h = \{ g^h(\cdot; \boldsymbol{\alpha}) \in \mathcal{G} \mid \boldsymbol{\alpha} \in \mathbb{R}^{n_\alpha} \}$$

denote the finite-dimensional subspace of admissible geometries parameterized by a finite coefficient vector $\boldsymbol{\alpha} \in \mathbb{R}^{n_\alpha}$ (such as B-spline control points, mesh vertex coordinates, or NURBS coefficients).

Discretizing the geometry yields the **finite-dimensional shape optimization problem**:

$$\begin{aligned}
\min_{\boldsymbol{\alpha} \in \mathbb{R}^{n_\alpha}, \, x_{ng} \in \mathbb{R}^{n_{xng}}} \quad & f(g^h(\cdot; \boldsymbol{\alpha}), x_{ng}) \\
\text{s.t.} \quad & c_{ge}(g^h(\cdot; \boldsymbol{\alpha}), x_{ng}) = \mathbf{0} \\
& c_{gi}(g^h(\cdot; \boldsymbol{\alpha}), x_{ng}) \le \mathbf{0} \\
& c_{ng}(x_{ng}) = \mathbf{0}
\end{aligned}$$

### Why Direct Discretized Optimization Fails
While mathematically tractable, directly optimizing $\boldsymbol{\alpha}$ presents severe practical hurdles:
1. **Curse of Dimensionality:** $n_\alpha$ can range from thousands to millions of control points or mesh nodes. Optimizing over this space requires vast gradient computations and converges slowly.
2. **Loss of Geometric Feasibility:** High-dimensional optimizers easily discover unphysical local minima characterized by high-frequency surface wrinkling, self-intersections, and negative element volumes.
3. **Loss of Physical Interpretability:** Individual spline coefficients $\alpha_k$ do not correspond to meaningful aircraft or robotic parameters.

---

## 3.3. Parameterized Shape Optimization Problem

To overcome the challenges of direct discretized optimization, an upstream **geometry parameterization mapping** $P$ is introduced:

$$P : \mathbb{R}^{n_{\alpha_p}} \to \mathbb{R}^{n_\alpha}$$

where $\boldsymbol{\alpha}_p \in \mathbb{R}^{n_{\alpha_p}}$ is a compact vector of shape parameters, with $n_{\alpha_p} \ll n_\alpha$.

The parameterization mapping restricts the geometry to an admissible manifold $\hat{\mathcal{G}}^h \subset \mathcal{G}^h$:

$$\hat{\mathcal{G}}^h = \{ g^h(\cdot; \boldsymbol{\alpha}) \in \mathcal{G}^h \mid \boldsymbol{\alpha} = P(\boldsymbol{\alpha}_p), \, \boldsymbol{\alpha}_p \in \mathbb{R}^{n_{\alpha_p}} \}$$

The resulting **parameterized shape optimization problem** is:

$$\begin{aligned}
\min_{\boldsymbol{\alpha}_p \in \mathbb{R}^{n_{\alpha_p}}, \, x_{ng} \in \mathbb{R}^{n_{xng}}} \quad & f(g^h(\cdot; P(\boldsymbol{\alpha}_p)), x_{ng}) \\
\text{s.t.} \quad & c_{ge}(g^h(\cdot; P(\boldsymbol{\alpha}_p)), x_{ng}) = \mathbf{0} \\
& c_{gi}(g^h(\cdot; P(\boldsymbol{\alpha}_p)), x_{ng}) \le \mathbf{0} \\
& c_{ng}(x_{ng}) = \mathbf{0}
\end{aligned}$$

The central research question is: **How should the mapping $P(\boldsymbol{\alpha}_p)$ be defined and computed to maximize flexibility, ensure feasibility, and maintain continuous differentiability?**

