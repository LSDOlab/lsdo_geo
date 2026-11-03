# Release Notes

## lsdo_geo 1.2.0 (November 2026)

`lsdo_geo` 1.2.0 introduces geometric assembly modeling, kinematic joints, semantic markers, and automated multi-component constraint enforcement, upgrading the underlying foundation to `lsdo_function_spaces >= 1.2.0`. This release enables complex aircraft and vehicle assemblies with multi-body kinematic connections, surface-to-surface alignment, and seamless integration into `ParameterizationSolver`.

---

### Highlights

* **Geometric Assemblies & Multi-Body Networks (`Assembly`)**: Assembly container subclassing `FunctionNetwork` for managing multi-component CAD configurations, automated internal seam discovery (`discover_internal_seams`), and topological freedom analysis.
* **Semantic Geometric Markers (`lsdo_geo.Marker`)**: Geometry-aware marker subclassing `lfs.Marker` supporting physical 3D coordinate projection (`Marker.from_locations`), parametric patch definitions (`Marker.from_parametric`), surface normal evaluation (`compute_surface_normal`), and orthonormal tangent null spaces (`compute_null_space`).
* **Kinematic Connections (`Connection`)**: High-level semantic interface connections including `spherical` (coincident points), `revolute` (hinge alignment), `prismatic` (sliding line), `fixed` (full 6-DOF weld), and `offset` constraints with skew-symmetric operator utilities (`S_of`).
* **ParameterizationSolver Integration**: Direct support for adding kinematic `Connection`, `InterfaceCondition`, or complete `Assembly` constraints via `solver.add_connection()` and `solver.add_assembly()`.
* **Unified Disciplinary Meshes (`Mesh`, `CamberSurfaceMesh`, `UnstructuredMesh`)**: `lsdo_geo.Mesh` now subclasses `Marker`, inheriting full discrete and continuous function space support (e.g. for Isogeometric Analysis, IGA). Meshes can be composed of constituent markers and custom subroutines (e.g. VLM mean camber averaging upper and lower surface wireframes). Added `UnstructuredMesh` with cell connectivity (`quad`, `triangle`), differentiable centroid and normal/area computation in CSDL, and `meshio` export.
* **Explicit API Signatures**: Eliminated unstructured `**kwargs` from marker evaluation and `Geometry.evaluate_signed_distance()`, providing explicit typing and comprehensive docstrings.
* **Upgraded Core Dependency**: Declares official dependency requirement on `lsdo_function_spaces >= 1.2.0`.

---

### Core Architecture & Geometry

* **Assembly & Kinematics (`lsdo_geo.core.geometry.assembly`)**:
  * `Connection`: Represents semantic kinematic joints between geometric markers, supporting spherical, revolute, prismatic, cylindrical, and fixed coupling.
  * `S_of(v)`: Computes the $3 \times 3$ skew-symmetric cross-product matrix $[v]_\times$ in both NumPy and CSDL graph modes for rotational alignment residuals $[v_1]_\times v_2 = 0$.
  * `Assembly`: Multi-body geometric network tracking components, kinematic connections, and automated patch seam discovery.
  * `rebind()` and `copy()`: Full support for cloning assemblies and rebinding marker references during FFD or parameterization solves.
* **Geometric Markers (`lsdo_geo.core.geometry.marker`)**:
  * `Marker.from_locations`: Projects physical 3D points onto geometry patches to define evaluation sites.
  * `compute_surface_normal`: Evaluates normalized surface normal vectors $n = (t_u \times t_v) / \|t_u \times t_v\|_2$.
  * `compute_tangents`: Evaluates surface tangent vectors $t_u, t_v$.
  * `compute_null_space`: Generates a $3 \times 2$ orthonormal basis spanning the surface tangent plane.
* **Disciplinary Meshes & Topologies (`lsdo_geo.core.geometry.mesh`)**:
  * `Mesh`: Subclasses `Marker`, supporting both discrete coordinate evaluations and exact continuous function spaces for Isogeometric Analysis (IGA). Supports constituent `markers` and custom `subroutine` post-processing functions.
  * `CamberSurfaceMesh`: Built-in composite mesh averaging upper and lower surface wireframe markers for vortex lattice and aerodynamic panel solvers.
  * `UnstructuredMesh`: Disciplinary mesh with element connectivity (`quad`, `triangle`, `line`). Includes CSDL-differentiable `compute_element_centroids()` and `compute_element_normals_and_areas()`, plus `to_meshio()` export for standard CAE formats.
* **ParameterizationSolver (`lsdo_geo.core.parameterization.parameterization_solver`)**:
  * `add_equality_constraint`: Extended to accept `Connection` or `InterfaceCondition` instances directly, automatically extracting residuals and assigning descriptive constraint names.
  * `add_connection`: Convenience method adding kinematic connection residuals as equality constraints.
  * `add_assembly`: Adds all condition residuals from a multi-body assembly or function network to the optimization problem.

---

## lsdo_geo 1.1.0 (October 2026)

`lsdo_geo` 1.1.0 introduces Rodrigues rotation formulations for differentiable CAD and mesh transformations, improves geometric translation handling, enhances Free-Form Deformation (FFD) block fitting robustness, and broadens mathematical and analytic derivative test verification.

---

### Highlights

* **Rodrigues Rotation Formulation**: Added analytical Rodrigues rotation matrix generation (`compute_rodrigues_rotation_matrix`) and rotation methods (`rotate_using_rodrigues`), providing an efficient, numerically stable alternative to quaternion rotations.
* **Rotation Vector Normalization & Multi-Angle Vectorization**: Guaranteed unit normalization of arbitrary rotation axis vectors and added support for batched multi-angle rotation arrays in `rotate()`.
* **Geometry Translation Robustness**: Fixed single-function component translations in `Geometry.translate()` with automatic shape broadcasting.
* **Dynamic FFD Enclosure Bounds**: Refactored `construct_tight_fit_ffd_block()` to calculate exact enclosure $z$-extents dynamically rather than relying on fixed coordinate indices.
* **Expanded Verification & Derivative Testing**: Added unit tests verifying machine-precision numerical agreement and analytic derivative equivalence between Rodrigues and Quaternion formulations under CSDL automatic differentiation.

---

### Core Architecture & Geometry

* **Rodrigues & Quaternion Rotation Methods (`Geometry`, `geometry_functions`)**:
  * Implemented `compute_rodrigues_rotation_matrix(axis, angle)` evaluating the exact $3 \times 3$ rotation matrix $\mathbf{R} = \mathbf{I} + \sin\theta \mathbf{K} + (1 - \cos\theta)\mathbf{K}^2$.
  * Added `rotate_using_rodrigues()` and `rotate_using_quaternion()` to top-level `lsdo_geo` exports and as methods on `Geometry`.
  * Added `method='rodrigues'|'quaternion'` keyword argument to `rotate()` (defaulting to `'rodrigues'`).
  * Enforced unit normalization of axis vectors prior to rotation evaluation, preventing angle distortion when non-unit vectors are supplied.
  * Added multi-angle rotation vectorization, reshaping output geometries to `(num_angles, ...)` when evaluating arrays of rotation angles.
* **Geometry Translation (`Geometry.translate`)**:
  * Resolved shape compatibility errors when translating a single function (`len(function_indices) == 1`) by expanding 1D translation vectors via `csdl.expand`.
* **Free-Form Deformation (`construct_tight_fit_ffd_block`)**:
  * Evaluates dynamic coordinate minima and maxima (`z_min`, `z_max`) across all enclosure control points, ensuring reliable tight-fitting FFD blocks for diverse CAD geometries.
* **Sectional Parameterization (`SectionalParameterization`)**:
  * Enhanced translation basis evaluation to handle dynamic `csdl.Variable` and `np.ndarray` directional axes consistently.

---

### Verification & Testing

* **Equivalence & Derivative Tests (`tests/test_geometry_functions.py`)**:
  * Verified that Rodrigues and Quaternion rotation formulations match to machine precision ($10^{-12}$).
  * Verified that CSDL automatic differentiation total derivatives with respect to points, axis, and angle match exactly between Rodrigues and Quaternion implementations.
  * Added dedicated test cases for non-unit rotation axes and multi-angle array rotations.
* **Component Translation Tests (`tests/test_geometry.py`)**:
  * Verified selective translation of individual functions within multi-patch geometries.
* **Parameterization Derivative Tests (`tests/test_sectional_parameterization.py`, `tests/test_ffd_block.py`)**:
  * Verified finite, non-zero analytic sensitivity propagation through sectional translation parameters.
  * Verified tight-fitting FFD block generation across arbitrary height distributions.

---

### Documentation & Packaging

* Updated project license metadata to SPDX `LGPL-3.0-or-later`.
* Synchronized version identifiers across `pyproject.toml`, `lsdo_geo/__init__.py`, and documentation configuration.

---

## lsdo_geo 1.0.0 (September 2026)

`lsdo_geo` 1.0.0 is the first major stable release of the LSDO lab's geometry representation, deformational parameterization, and sensitivity analysis framework for Multidisciplinary Design Optimization (MDO). This release modernizes the codebase for full Python 3.9–3.14 and NumPy 2.x compatibility, consolidates the parameterization API, introduces automated CI and PyPI deployment pipelines, and broadens unit test coverage.

---

### Highlights

* **Extended Python Support**: Full native support and automated test coverage across Python 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14.
* **NumPy 2.x Ready**: Resolved scalar slice assignment and shape mutation patterns across all geometric transformations and rotation matrices.
* **Consolidated Sectional Parameterization**: Unified all sectional parameterization under `SectionalParameterization` and `SectionalParameters`, removing legacy redundant modules while preserving backward compatibility.
* **Modern Standards-Compliant Packaging**: Packaged via `pyproject.toml` (PEP 517/518/621/639) with SPDX LGPL-3.0-or-later licensing and automated setuptools discovery restricted to `lsdo_geo*`.
* **Continuous Integration**: GitHub Actions workflow updated to `actions/checkout@v7` and `actions/setup-python@v7` running on Node 24 with a 6-job matrix.

---

### Core Architecture & Features

* **Implicit Parameterization Solver (`ParameterizationSolver`, `NewtonOptimizer`)**:
  * Formulates geometry parameterization as an inner optimization problem with exact KKT first-order optimality systems.
  * Solves equality constraints via exact Lagrange multipliers and inequality constraints via quadratic/linear penalty terms.
  * Evaluates analytic adjoint sensitivities via CSDL Alpha backward graph automatic differentiation.
* **Free-Form Deformation (`FFDBlock`)**:
  * Constructs trivariate B-spline lattice volumes enclosing CAD entities or discrete point clouds (`construct_ffd_block_around_entities`, `construct_ffd_block_from_corners`).
  * Parametric mapping embeds spatial coordinates and applies global continuous deformations.
* **Sectional Parameterization (`SectionalParameterization`, `SectionalParameters`)**:
  * Perceives 3D volumes or 2D surfaces as slices along a specified principal parametric dimension.
  * Supports stretch (scaling), translation (sweep, dihedral), and rotation (twist, pitch) along parametric dimensions, arbitrary 3D spatial vectors (`np.ndarray`), or dynamic `csdl.Variable` axes.
  * Supports custom section-varying pivot points and centers via parametric coordinates.
* **CAD & Mesh Integration (`Geometry`, `Mesh`)**:
  * Direct import and management of B-spline surface patches and STEP files (`import_geometry`).
  * Discrete mapping of CFD, FEA, and acoustic meshes onto underlying spline surfaces.
  * Rigid spatial transformations (translation, Euler rotation, and quaternion rotation).

---

### API Consolidations & Deprecations

* **Removed Legacy Module**:
  * Removed `lsdo_geo.core.parameterization.volume_sectional_parameterization` (-662 lines).
* **Backward Compatibility Aliases**:
  * `VolumeSectionalParameterization` is now a subclass alias of `SectionalParameterization`, emitting a `DeprecationWarning` on instantiation.
  * `VolumeSectionalParameterizationInputs` is now a subclass alias of `SectionalParameters`, emitting a `DeprecationWarning` on instantiation.
  * `add_sectional_stretch`, `add_sectional_translation`, and `add_sectional_rotation` are alias methods on `SectionalParameters`, emitting `DeprecationWarning` and delegating to `add_stretch`, `add_translation`, and `add_rotation`.
  * All canonical and deprecated classes are exported from the top-level `lsdo_geo` package.

---

### Bug Fixes

* **3D Quaternion General Axis Rotation**: Fixed slice indexing in `lsdo_geo/core/geometry/geometry_functions.py` (`csdl.slice[i, :, 1:]`) so arbitrary 3D spatial axis rotations execute without shape mismatch.
* **Section Normal Vector Cross Product**: Resolved `IndexError` in `_compute_section_axis` by evaluating the two non-principal axes in ascending order and computing their cross product with `axis=0`.
* **NumPy Dot Product with CSDL Variables**: Resolved array broadcasting mismatch during stretch and translation basis calculations by extracting evaluated array values for geometric basis computation.
* **Pre-Evaluation Plotting**: Uncommented `self.updated_points = self.parameterized_points` in `SectionalParameterization.__post_init__`, allowing `sp.plot()` calls prior to `sp.evaluate()`.

---

### Documentation & Infrastructure

* Modernized Sphinx documentation theme configuration (`conf.py`) with clean HTML build and zero warnings (`sphinx -b html docs docs/_build/html -q -W`).
* Automated PyPI release workflow (`.github/workflows/publish.yml`) configured with Trusted Publishing (OIDC) and tag-driven deployment.
