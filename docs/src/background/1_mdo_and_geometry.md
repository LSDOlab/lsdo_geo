# 1. Geometry-Centric MDO & Parameterization

Modern engineering design frequently tackles complex, high-dimensional systems involving tightly coupled physical phenomena—such as electric vertical take-off and landing (eVTOL) aircraft, soft swimming robots, hypersonic and laser-powered vehicles, and legged robotic platforms. 

In these systems, performance cannot be accurately captured by isolated single-discipline evaluations. Aerodynamic loading dictates structural deflection; structural compliance alters the aerodynamic flow field; acoustic radiation and motor thermal constraints impose tight spatial trade-offs. Multidisciplinary Design Optimization (MDO) provides a systematic mathematical framework to explore these trades and achieve optimal, non-intuitive system configurations {cite:p}`ruh2026airtaxi`.

```{figure} ../images/lsdolab.png
:align: center
:alt: LSDO Lab
```

---

## The Challenge of Inconsistent Geometry Discretizations

Geometry is the primary shared input across virtually all physics-based analysis models. However, each discipline typically relies on a distinct spatial discretization tailored to its specific governing equations:

- **Aerodynamics (CFD & Panel Methods):** Requires watertight surface meshes with specialized boundary-layer inflation or vortex panel distributions.
- **Structural Mechanics (FEA):** Requires interior wingbox shells, spar and rib beam meshes, and solid elements with structural connectivity.
- **Flight Dynamics & Control:** Requires rigid-body masses, inertia tensors, and center-of-gravity locations.
- **Acoustics & Propulsion:** Requires propeller disk planes, rotor blade planforms, and acoustic observer rays.

When each discipline generates and modifies its own geometry model independently, inconsistencies inevitably arise. Components disconnect at interfaces, thicknesses mismatch, and derivatives become impossible to propagate consistently across disciplines.

---

## The Central Geometry Paradigm

To resolve spatial inconsistency, modern high-fidelity MDO employs a **geometry-centric modeling approach** {cite:p}`kenway2010cad, fletcher2026implicit`.

Rather than driving individual analysis meshes directly with optimizer design variables, a single, detailed, solver-independent **central geometry representation** is defined as the central interface. All discipline-specific analysis models (CFD surfaces, FEA beam/shell meshes, acoustic observer grids) are automatically projected from and mapped to this shared source of truth.

```
                  ┌───────────────────────────────┐
                  │   Geometric Design Variables  │
                  │   (e.g., span, chord, sweep)  │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    Geometry Parameterization  │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ Central Geometry Representation│
                  │       (Analytic Surfaces)     │
                  └──────┬──────────────┬─────────┘
                         │              │
           ┌─────────────┴──────┐ ┌─────┴──────────────┐
           ▼                    ▼ ▼                    ▼
    ┌─────────────┐      ┌─────────────┐        ┌─────────────┐
    │  CFD Mesh   │      │  FEA Mesh   │        │Acoustic Mesh│
    └──────┬──────┘      └──────┬──────┘        └──────┬──────┘
           ▼                    ▼                      ▼
    ┌─────────────┐      ┌─────────────┐        ┌─────────────┐
    │Aerodynamics │      │ Structures  │        │  Acoustics  │
    └─────────────┘      └─────────────┘        └─────────────┘
```

This centralized representation guarantees:
1. **Geometric Consistency Across Disciplines:** Deflections and shape changes map back to the same master surfaces.
2. **Analytic Differentiability:** Adjoint sensitivities can be propagated from each discipline solver back to the underlying geometric definition.
3. **Multi-Fidelity Scalability:** The same central geometry can generate low-, medium-, and high-fidelity analysis meshes without altering design variables.

---

## Criteria for Effective Geometry Parameterization

While the central geometry representation provides the spatial definition, **geometry parameterization** defines the mapping from a chosen set of design variables to that representation.

In large-scale gradient-based MDO, an effective parameterization must satisfy five key criteria {cite:p}`fletcher2026implicit`:

1. **Compactness:** Efficiently map a small, low-dimensional set of design variables to a rich, high-dimensional space of potentially optimal shapes.
2. **Geometric Feasibility:** Predominantly yield feasible geometries throughout optimization, avoiding unphysical self-intersections or disjoint components that crash numerical simulation solvers.
3. **Continuous Differentiability:** Support fast, accurate analytic first and second derivatives for gradient-based optimizers and adjoint sensitivity analysis.
4. **Adaptability:** Easily adapt to increasing geometric complexity (e.g., adding components, struts, or nacelles) without restructuring the entire parameterization pipeline.
5. **Interpretability:** Provide intuitive, application-tailored parameters (such as aspect ratio, taper ratio, sweep, and span) that reflect engineering practice rather than uninterpretable mathematical coefficients.

