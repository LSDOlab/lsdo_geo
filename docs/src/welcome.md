# Welcome to lsdo_geo (v1.0.0)

![LSDO Lab](/src/images/lsdolab.png "LSDO Lab")

**lsdo_geo** is a Python library for geometry and state modeling/representation, parameterization, and mesh generation tailored for Multidisciplinary Design Optimization (MDO).

It integrates with [CSDL Alpha](https://github.com/LSDOlab/CSDL_alpha) and provides analytic derivative-compatible parameterizations (such as Free-Form Deformation (FFD) and B-spline sectional parameterization) and automated mesh projection.

## Key Capabilities
- **Sectional & Shape Parameterization**: Easily modify cross-sections, twist, sweep, dihedral, and chord distributions.
- **Analytic Derivatives**: Seamless integration with CSDL graph execution and automatic differentiation.
- **Mesh Projection & Mapping**: Project aerodynamic/structural meshes onto underlying CAD geometries and spline definitions.

See the [Getting Started](src/getting_started.md) guide to install and get started with `lsdo_geo`, browse the [Tutorials](src/tutorials.md) for step-by-step walk-throughs, or explore the [Examples](src/examples.md) gallery.

## Cite us

If you use `lsdo_geo` in your research or applications, please cite our paper:

```bibtex
@inproceedings{fletcher2026implicit,
  title     = {Implicit Nonlinear Geometry Parameterization for Multidisciplinary Design Optimization},
  author    = {Fletcher, Andrew H. and Hwang, John T.},
  booktitle = {Proceedings of the ASME 2026 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC/CIE2026)},
  number    = {DETC2026-192508},
  year      = {2026},
  address   = {Houston, TX}
}
```

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/tutorials
src/examples
src/api
```
