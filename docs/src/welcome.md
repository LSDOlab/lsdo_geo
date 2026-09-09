# Welcome to lsdo_geo (v0.1.0)

![LSDO Lab](/src/images/lsdolab.png "LSDO Lab")

**lsdo_geo** is a Python library for geometry and state modeling/representation, parameterization, and mesh generation tailored for Multidisciplinary Design Optimization (MDO).

It integrates with [CSDL](https://github.com/LSDOlab/csdl) / CSDL Alpha and provides analytic derivative-compatible parameterizations (such as Free-Form Deformation (FFD) and B-spline sectional parameterization) and automated mesh projection.

## Key Capabilities
- **Sectional & Shape Parameterization**: Easily modify cross-sections, twist, sweep, dihedral, and chord distributions.
- **Analytic Derivatives**: Seamless integration with CSDL graph execution and automatic differentiation.
- **Mesh Projection & Mapping**: Project aerodynamic/structural meshes onto underlying CAD geometries and spline definitions.

See the [Getting Started](src/getting_started.md) guide to install and get started with `lsdo_geo`, browse the [Tutorials](src/tutorials.md) for step-by-step walk-throughs, or explore the [Examples](src/examples.md) gallery.

## Cite us
```none
@article{lsdo_geo,
    author = {Andrew Fletcher},
    title = {lsdo_geo: Geometry Representation and Parameterization for MDO},
    year = {2024}
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
