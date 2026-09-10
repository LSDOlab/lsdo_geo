# lsdo_geo

[![Documentation Status](https://readthedocs.org/projects/lsdo-geo/badge/?version=latest)](https://lsdo-geo.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/LSDOlab/lsdo_geo/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOlab/lsdo_geo/actions)
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE.txt)

**lsdo_geo** is a Python library for geometry representation, implicit deformational parameterization, and analytic adjoint sensitivity analysis tailored for gradient-based Multidisciplinary Design Optimization (MDO).

Developed by the [Large-Scale Design Optimization (LSDO) Lab](https://lsdo.eng.ucsd.edu/) at the University of California, San Diego.

---

## Key Features

* **Geometry-Centric MDO**: Serves as a single, consistent geometric "source of truth", generating and updating discipline-specific meshes (CFD, FEA, acoustics) without manual re-meshing or geometric inconsistency.
* **Implicit Parameterization Framework**: Formulates geometry parameterization as an inner constrained optimization problem, solving the first-order Karush-Kuhn-Tucker (KKT) system with an exact Newton solver.
* **Hierarchical Free-Form Deformation (FFD)**: Supports trivariate spline volume lattices with sectional deformation modes—twist, sweep, dihedral, spanwise stretching, and chord taper.
* **Analytic Adjoint Sensitivities**: Provides exact implicit derivatives $\frac{d\mathbf{x}}{d\mathbf{x}_g}$ through the converged KKT state, seamlessly propagating gradients across the CSDL computational graph.
* **CAD & Mesh Interoperability**: Direct import, fitting, and export across OpenVSP (`.vsp3`), IGES (`.iges`), STEP (`.stp`), and surface/volume meshes (`.msh`, `.stl`).

---

## Architecture Overview

| Subpackage | Key Classes | Description |
|:---|:---|:---|
| **`lsdo_geo.core.geometry`** | `Geometry`, `Mesh` | Central multi-component geometric model, CAD imports, rigid transformations, and discipline mesh projections. |
| **`lsdo_geo.core.parameterization`** | `ParameterizationSolver`, `GeometricVariables`, `FFDBlock`, `SectionalParameterization` | Trivariate FFD lattices, sectional mode definitions, and exact Newton KKT solver. |
| **`lsdo_geo.optimization`** | `Optimization`, `NewtonOptimizer` | Connects geometric states and constraints into CSDL graph models for analytic adjoint sensitivity analysis. |

---

## Quickstart

```python
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg

# 1. Initialize CSDL graph recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# 2. Define or import geometry
geometry = lg.import_geometry("path/to/wing.stp")

# 3. Create a Free-Form Deformation (FFD) block around the geometry
ffd_block = lg.construct_ffd_block_around_entities(
    geometry,
    num_control_points=(4, 4, 2),
)

# 4. Apply geometric transformations (rotation, translation)
rotation_origin = np.array([0.0, 0.0, 0.0])
rotation_axis = np.array([0.0, 0.0, 1.0])
rotated_points = lg.rotate(
    geometry.coefficients,
    rotation_origin=rotation_origin,
    axis_vector=rotation_axis,
    angles=15.0,
    units="degrees",
)
```

---

## Installation

### Prerequisites & Installation
`lsdo_geo` relies on `CSDL_alpha` and `lsdo_function_spaces`:

```sh
# 1. Install dependencies
pip install jax networkx
pip install git+https://github.com/LSDOlab/CSDL_alpha.git@dev_andrew
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git

# 2. Install lsdo_geo (User)
pip install lsdo_geo
# Or install development version directly from GitHub:
pip install git+https://github.com/LSDOlab/lsdo_geo.git
```

### For Developers
Clone the repository and install in editable mode with testing and documentation extras:
```sh
git clone https://github.com/LSDOlab/lsdo_geo.git
cd lsdo_geo
pip install jax networkx
pip install git+https://github.com/LSDOlab/CSDL_alpha.git@dev_andrew
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git
pip install -e ".[test,docs]"
```

---

## Testing

Run the test suite using `pytest`:
```sh
pytest
```

To run fast unit tests only:
```sh
pytest -m "not slow"
```

---

## Documentation

Full documentation, theoretical background, tutorials, and API reference are hosted on Read the Docs:
👉 **[lsdo-geo.readthedocs.io](https://lsdo-geo.readthedocs.io/en/latest/)**

To build the documentation locally:
```sh
sphinx-build -b html docs docs/_build/html
```

---

## License

`lsdo_geo` is distributed under the terms of the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.
