# Theoretical Background

`lsdo_geo` is a specialized geometry modeling and parameterization framework built for **gradient-based Multidisciplinary Design Optimization (MDO)** of complex engineering systems.

This section provides the theoretical foundation, mathematical formulations, numerical solution strategies, and case studies underlying `lsdo_geo`, based on the ASME IDETC/CIE 2026 paper: *"Implicit Nonlinear Geometry Parameterization for Multidisciplinary Design Optimization"* {cite:p}`fletcher2026implicit`.

---

## Architecture Overview

In multidisciplinary shape optimization, `lsdo_geo` formulates the parameterization mapping itself as an **inner nonlinear optimization problem**. This bridges the gap between high-level, intuitive engineering design variables and detailed, multi-component central geometry representations:

```
  ┌─────────────────────────────────────────────────────────────┐
  │ Outer MDO Optimizer (Design Variables x_g, x_ng)            │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Prescribed engineering targets x_g
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ lsdo_geo Parameterization Solver                            │
  │ min (α_s^T W α_s)  s.t. c_ge=0, c_gi<=0, x_hat_g(α_s)-x_g=0 │
  │ - Solves KKT system via exact Newton method                 │
  │ - Enforces multi-component kinematic attachment             │
  │ - Preserves feasibility via interior-point inequalities     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Converged states α_s*
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Central Geometry Representation (B-Splines & Lattices)      │
  └──────────────┬──────────────────────────────┬───────────────┘
                 │                              │
                 ▼                              ▼
  ┌──────────────────────────────┐┌─────────────────────────────┐
  │ Aerodynamic Mesh (CFD/Panel) ││ Structural Mesh (Shell/Beam)│
  └──────────────────────────────┘└─────────────────────────────┘
```

---

## Background Sections

```{toctree}
:maxdepth: 2
:numbered: 1

background/1_mdo_and_geometry
background/2_parameterization_approaches
background/3_mathematical_formulation
background/4_implicit_parameterization
background/5_solver_and_computational_graph
background/6_applications
```

---

## Bibliography

```{bibliography} references.bib
:style: unsrt
```