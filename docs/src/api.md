# API Reference

This section provides the complete reference for the `lsdo_geo` application programming interface (API), automatically extracted from source docstrings and type annotations.

---

## Package Architecture Overview

`lsdo_geo` is modularized into core subpackages covering geometry representation, deformational parameterization, numerical solving, and CSDL graph optimization:

| Subpackage | Key Classes & Functions | Description |
|:---|:---|:---|
| **[`lsdo_geo.core.geometry`](autoapi/lsdo_geo/core/geometry/index)** | [`Geometry`](autoapi/lsdo_geo/core/geometry/geometry/index), [`Mesh`](autoapi/lsdo_geo/core/geometry/mesh/index), `geometry_functions` | Geometry representations, surface and volume meshes, CAD imports, and spatial transformations. |
| **[`lsdo_geo.core.parameterization`](autoapi/lsdo_geo/core/parameterization/index)** | [`ParameterizationSolver`](autoapi/lsdo_geo/core/parameterization/parameterization_solver/index), [`GeometricVariables`](autoapi/lsdo_geo/core/parameterization/parameterization_solver/index), [`FFDBlock`](autoapi/lsdo_geo/core/parameterization/ffd_block/index), [`SectionalParameterization`](autoapi/lsdo_geo/core/parameterization/sectional_parameterization/index) | Implicit nonlinear solver, Free-Form Deformation lattices, and sectional deformation modes. |
| **[`lsdo_geo.csdl`](autoapi/lsdo_geo/csdl/index)** | [`Optimization`](autoapi/lsdo_geo/csdl/optimization/index) | Interface connecting geometric states and constraints into CSDL graph models for analytic adjoint sensitivity analysis. |
| **[`lsdo_geo.splines`](autoapi/lsdo_geo/splines/index)** | `B-Splines` | Underlying B-spline curves, surfaces, and volumes supporting geometric definitions. |
| **[`lsdo_geo.utils`](autoapi/lsdo_geo/utils/index)** | `geometry_functions` | Helper mathematical and geometric utilities. |

---

## Core Geometry

### `lsdo_geo.core.geometry.Geometry`
The primary class representing multi-component geometric systems. It manages underlying function spaces, component declarations, rigid transformations (rotations, translations, quaternion-based orientations), and exports to CAD/mesh formats (`.iges`, `.obj`, `.stl`).

### `lsdo_geo.core.geometry.Mesh`
Handles structured and unstructured meshes. Provides mapping algorithms to project discipline-specific CFD, FEA, or acoustic meshes onto underlying spline surfaces or CAD boundaries.

---

## Parameterization & Implicit Solvers

### `lsdo_geo.core.parameterization.ParameterizationSolver`
The core solver of `lsdo_geo`. It formulates the geometry parameterization problem as an inner optimization problem:

$$\min_{\boldsymbol{\alpha}_s} \boldsymbol{\alpha}_s^T \mathbf{W} \boldsymbol{\alpha}_s \quad \text{s.t.} \quad \mathbf{c}_{ge} = \mathbf{0}, \quad \mathbf{c}_{gi} \le \mathbf{0}, \quad \hat{\mathbf{x}}_g(\boldsymbol{\alpha}_s) - \mathbf{x}_g = \mathbf{0}$$

Solves the KKT first-order optimality system using an exact Newton method and provides analytic adjoint sensitivities through the converged implicit state.

### `lsdo_geo.core.parameterization.GeometricVariables`
A structured container linking computed geometric quantities (evaluated from geometry surfaces) with target values prescribed by outer optimizers, supporting both exact Lagrange multiplier enforcement and weak penalty formulations.

### `lsdo_geo.core.parameterization.FFDBlock`
Constructs trivariate Free-Form Deformation (FFD) lattice volumes that enclose baseline geometries. Embedded points are smoothly transformed as lattice control points are perturbed.

### `lsdo_geo.core.parameterization.SectionalParameterization`
Groups FFD lattice control points into cross-sections along a specified principal axis, applying rigid translation, rotation (twist, sweep, dihedral), and in-plane stretching to preserve realistic aerodynamic shape variations.

---

## Complete Auto-Generated Reference

Browse the full hierarchy of modules, classes, and functions below:

```{toctree}
:maxdepth: 2
:titlesonly:

autoapi/lsdo_geo/index
```