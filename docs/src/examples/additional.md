# Additional Examples

The `examples/additional_examples` directory contains specialized studies, benchmarks, and historical parameter sweeps:

## Hand-Launched UAVs (`hand_launched_uavs/`)
Studies in small unmanned aerial vehicle configurations:

* **`hand_launched_uavs_parameterization.py`**: UAV geometry parameterization and Latin Hypercube Sampling (LHS) design of experiments.
* **`regenerate_lhs_grid.py`**: Post-processing utility to compose LHS sample grids from rendered images without full re-simulation.

## Aerodynamic & Aerostructural Benchmarks
* **`ex_openaerostruct.py`**: Coupled aerostructural analysis integrating OpenAeroStruct with `lsdo_geo`.
* **`ex_nasa_aero_shape_opt_benchmark.py`**: NASA aerodynamic shape optimization benchmark configuration.
* **`ex_rectangular_wing_toy_optimization.py`**: Lightweight optimization problem demonstrating basic geometric sensitivity flows.
* **`ex_rectangular_wing_with_panel_method.py`**: Rectangular wing parameterization evaluated with 3D panel aerodynamic methods.
* **`ex_rectangular_wing_with_shape.py`**: Shape parameterization with custom cross-sectional profiles.

## Vehicle Variants & Parameter Sweeps
* **`incomplete_ex_tbw.py`**: Strut-braced / truss-braced wing (TBW) geometry parameterization study.
* **`run_elevator_sweep.py`**: Parametric sweep analysis of control surface deflections and elevator trim characteristics.
* **`create_perturbation_video.py`**: Script rendering offscreen PyVista animations of geometric perturbation cycles.

