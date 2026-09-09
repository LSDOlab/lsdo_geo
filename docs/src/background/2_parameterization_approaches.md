# Survey of Geometry Parameterization Methods

Existing techniques for geometric parameterization in design optimization generally fall into three major paradigms: **constructive**, **deformational**, and **constraint-based** methods {cite:p}`fletcher2026implicit`. Each offers distinct trade-offs between flexibility, interpretability, and differentiability.

---

## Constructive Parameterization Approaches

In constructive approaches, design variables are directly embedded into the procedural build recipe of the geometry representation.

### Primitives & Shape Functions
- **Concept:** Construct geometries by superimposing predefined analytical basis functions, such as Class-Shape Transformation (CST) functions, Joukowsky airfoils, or standard geometric primitives (cylinders, spheres).
- **Strengths:** Fast to evaluate, naturally restricted to smooth aerodynamic shapes, and parameters (e.g., maximum thickness, camber) have clear physical meaning.
- **Limitations:** Primarily restricted to canonical aerodynamic components (isolated wings, nacelles). They struggle with localized shape control and cannot easily parameterize complex, multi-component topologies.

### CAD-Based Parameterization
- **Concept:** Use a commercial or open-source solid-modeling Computer-Aided Design (CAD) kernel (e.g., OpenCSM, ESP) to construct shapes using feature trees, boolean operations, fillets, and lofts.
- **Strengths:** Directly exportable to manufacturing, application-agnostic, and integrates seamlessly with standard industrial CAD/CAM pipelines.
- **Limitations:** 
  - **Lack of Analytic Differentiability:** Most commercial CAD kernels do not provide exact derivatives, forcing expensive and noisy finite differences.
  - **Feature Tree Non-Uniqueness:** Multiple disparate CAD feature sequences can yield identical shapes, leading to non-smooth or discontinuous sensitivity landscapes.
  - **Brittleness:** Automated regenerations frequently fail when aggressive optimizer steps trigger degenerate topology or self-intersecting fillets.

---

## Deformational Parameterization Approaches

Rather than building design parameters into the representation, deformational approaches start with an existing baseline geometry and smoothly deform it, effectively decoupling the parameterization from the underlying geometric complexity.

### Free-Form Deformation (FFD)
Introduced by Sederberg and Parry {cite:p}`sederberg1986free`, Free-Form Deformation surrounds a geometry with a simpler parametric volume (typically a trivariate B-spline or Bernstein lattice):

$$\mathbf{x}(\xi, \eta, \zeta) = \sum_{i=0}^l \sum_{j=0}^m \sum_{k=0}^n B_i^l(\xi) B_j^m(\eta) B_k^n(\zeta) \, \mathbf{P}_{i,j,k}$$

As the control points $\mathbf{P}_{i,j,k}$ of the enclosing lattice are perturbed, the embedded system geometry smoothly deforms accordingly.

- **Strengths:**
  - Fast, exact analytic derivative computation.
  - Complete decoupling between geometry resolution (e.g., a million surface mesh nodes) and parameterization dimensionality (e.g., tens or hundreds of FFD control points).
  - Excellent local shape control.
- **Limitations:**
  - Direct perturbation of FFD lattice control points yields a large number of weakly significant, non-intuitive design variables.
  - Complex geometries (e.g., aircraft with wings, fuselage, booms, and multiple rotors) cannot easily conform to a single FFD block. Setting up multiple adjoining lattices requires tedious manual effort and often fails to preserve continuity across component intersections.

### Axial and Hierarchical FFD
To improve interpretability, axial FFD introduces an intermediate guide curve along a principal axis, grouping control points into cross-sections that undergo rigid-body rotations, translations, or scaling {cite:p}`kenway2010cad`. While this restricts the design space toward more realistic aircraft wings, manual lattice construction remains difficult for irregular multi-body configurations.

---

## Constraint-Based Deformational Approaches

Constraint-based methods invert the deformational pipeline by creating an implicit system: the user specifies desired geometric target locations or constraints (often referred to as **pilot points**), and a numerical solver computes the underlying control points or shape degrees of freedom required to achieve them.

### Direct Manipulation & Strain-Energy Minimization
Because an infinite number of control point perturbations can satisfy a set of pilot point displacements, an optimization sub-problem is posed:

$$\min_{\Delta \mathbf{x}} \quad \mathcal{E}(\Delta \mathbf{x}) \quad \text{s.t.} \quad \mathbf{C}(\Delta \mathbf{x}) = \mathbf{x}_{\text{pilot}} - \mathbf{x}^* = \mathbf{0}$$

- **State Change Minimization:** Minimizing the $L_2$ norm of state change ($\|\Delta \mathbf{x}\|_2^2$) reduces to a linear least-squares problem.
- **Physics-Based / Strain-Energy Minimization:** Modeling the deformation as an elastic continuum (minimizing linear or hyperelastic strain energy) ensures smooth, physically plausible transitions around displaced pilot points.

### Limitations of Existing Constraint-Based Methods
1. **Pilot Points $\neq$ Engineering Design Variables:** Designers and aerodynamicists rarely think in terms of arbitrary 3D pilot coordinates; they design using wing span, aspect ratio, sweep angle, dihedral, and taper ratio.
2. **Multi-Component Scalability:** Directly manipulating multiple components with distinct functional behaviors (e.g., deforming wings attached to rigid fuselages and rotating propellers) remains difficult to configure.
3. **Absence of Feasibility Guarantees:** These approaches do not readily support general inequality constraints (such as non-intersection bounds or component clearance limits) while preserving differentiability.

---

## The Gap Addressed by `lsdo_geo`

Existing literature leaves a pronounced gap: **how to parameterize complex, multi-component systems with custom, domain-specific engineering variables while rigorously guaranteeing geometric feasibility throughout the optimization.**

`lsdo_geo` addresses this gap by formulating the geometry parameterization itself as a **generalized implicit nonlinear optimization problem**, unifying hierarchical FFD, multi-component kinematic links, and interior-point inequality constraint enforcement.

