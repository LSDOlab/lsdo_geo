# Showcase Examples

The `examples/showcase_examples` directory contains curated, production-ready reference examples demonstrating geometry parameterization, Free-Form Deformation (FFD), and multidisciplinary design optimization:

## Rectangular Wing (`rectangular_wing/`)
Fundamental baseline geometries, morphing parameterizations, and coupled MDO workflows:

* **`ex_rectangular_wing.py`**: Baseline rectangular wing geometry representation and surface mesh generation with FFD blocks.
* **`ex_rectangular_wing_to_bwb.py`**: Continuous morphing and cross-sectional parameterization transforming a rectangular wing into a blended wing-body (BWB) layout.
* **`ex_rectangular_wing_aero_shape_optimization.py`**: Coupled aerodynamic shape optimization using CSDL and vortex lattice methods (VLM).
* **`ex_rectangular_wing_aero_shape_optimization_with_chord_profile.py`**: Aerodynamic shape optimization with spanwise chord profile parameterization.
* **`ex_rectangular_wing_aerostructural_optimization.py`**: Coupled aerostructural optimization of a flexible lifting surface with geometric design variables.
* **`ex_rectangular_wing_aerostructural_shape_optimization_with_chord_profile.py`**: Full aerostructural shape optimization incorporating chord profile parameterization.


## Lift + Cruise eVTOL (`lift_plus_cruise/`)
Complex multi-component Urban Air Mobility (UAM) configuration:

* **`ex_lift_plus_cruise.py`**: Multi-component lift-plus-cruise eVTOL aircraft model parameterization.
* **`ex_lift_plus_cruise_with_inequalities.py`**: Full eVTOL configuration parameterization enforcing geometric inequality constraints (e.g. minimum volume, clearance).
* **`ex_lift_plus_cruise_timings.py`**: Performance benchmarking and Newton solver timing evaluations across input perturbations.
* **`ex_lift_plus_cruise_aerostructural_optimization.py`**: Coupled aerostructural optimization of the lift-plus-cruise vehicle.

