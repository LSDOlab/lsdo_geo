# Release Notes

## lsdo_geo 1.0.0 (September 2026)

`lsdo_geo` 1.0.0 is the first major stable release of the LSDO lab's geometry representation, deformational parameterization, and sensitivity analysis framework for Multidisciplinary Design Optimization (MDO). This release modernizes the codebase for full Python 3.9–3.14 and NumPy 2.x compatibility, consolidates the parameterization API, introduces automated CI and PyPI deployment pipelines, and broadens unit test coverage.

---

### Highlights

* **Extended Python Support**: Full native support and automated test coverage across Python 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14.
* **NumPy 2.x Ready**: Resolved scalar slice assignment and shape mutation patterns across all geometric transformations and rotation matrices.
* **Consolidated Sectional Parameterization**: Unified all sectional parameterization under `SectionalParameterization` and `SectionalParameters`, removing legacy redundant modules while preserving backward compatibility.
* **Modern Standards-Compliant Packaging**: Packaged via `pyproject.toml` (PEP 517/518/621) with SPDX Apache-2.0 licensing and automated setuptools discovery restricted to `lsdo_geo*`.
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
