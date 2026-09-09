# Multidisciplinary Applications & Case Studies

The implicit nonlinear geometry parameterization framework implemented in `lsdo_geo` has been benchmarked and deployed across a diverse spectrum of large-scale multidisciplinary design optimization problems {cite:p}`fletcher2026implicit`.

---

## 6.1. Urban Air Mobility (NASA Lift+Cruise eVTOL)

Electric vertical take-off and landing (eVTOL) aircraft represent an emerging class of complex multi-component systems with tightly coupled aerodynamics, structures, acoustics, and electric powertrains.

### Benchmark Formulation & Latin Hypercube Timing
The NASA Lift+Cruise configuration features a wing, fuselage, horizontal/vertical tail, twin propeller booms, and 8 lifting rotors {cite:p}`fletcher2026implicit`:
- **Inner Map:** FFD volumes for wing, fuselage, and tail with spanwise B-spline sectional curves; rigid-body translations for booms and rotors.
- **Design Variables (Outer Optimizer):** Wing/tail planform area ($S$), aspect ratio ($AR$), taper ratio ($\lambda$), sweep angle ($\Lambda$), lifting rotor radii, and tail moment arm.
- **Constraints (Geometry Solver):** 66 geometric equality constraints enforcing that booms stay attached to the wing and rotors stay fixed on the booms.
- **Inequality Constraints:** Non-contact clearance bounds preventing rotor-to-rotor and rotor-to-fuselage collisions.
- **Numerical Robustness (300 Latin Hypercube Samples):**
  - *Without Inequalities:* 100% convergence rate, average runtime **9.3 seconds** per sample on an Intel Core i9 laptop.
  - *With Inequalities:* 100% convergence rate, average runtime **11.2 seconds** per sample.

### System-Level MDO
In full system-level optimization by Ruh et al. {cite:p}`ruh2024largescale, ruh2026airtaxi`:
- 161 total design variables, 96 outer optimizer constraints.
- 66 geometric constraints and 5 geometric variables handled entirely by the `lsdo_geo` parameterization solver.
- Achieved a **6.5% reduction in aircraft gross weight** while guaranteeing consistent structural, aerodynamic, and acoustic mesh deformation at every major iteration.

---

## 6.2. Eel-Inspired Soft Swimming Robot

Anguilliform-swimming soft robots navigate delicate underwater environments using traveling-wave undulations. Their swimming efficiency depends on complex fluid-structure interaction (FSI) between hyperelastic silicone elastomer bodies and the surrounding fluid {cite:p}`fletcher2025anguilliform`.

- **Parameterization Setup:** Mixed-state FFD formulation combining local lateral width shape variables directly on outer FFD lattice control points with module height variables enforced by the geometry solver.
- **Optimization Results:** Co-design of morphology and traveling wave actuation parameters achieved a **45% reduction in simulated energy cost per distance** while meeting swim speed requirements.

---

## 6.3. High-Altitude Laser-Powered Aircraft

Power-beaming aircraft receive energy wirelessly via laser or microwave beams from ground stations, enabling virtually indefinite flight endurance {cite:p}`orndorff2023laser`.

- **Configuration:** Twin-fuselage, high-aspect-ratio composite aircraft.
- **Kinematic Constraints:** 6 equality constraints in `lsdo_geo` enforced that the center-wing span, horizontal stabilizer, and inter-fuselage spacing remained equal while translating propeller nacelles as the tail stretched.
- **Optimization Outcome:** Minimizing vehicle mass across 50 design variables yielded an optimal, significantly enlarged wing layout and identified the optimal cruising altitude through operational sweeps.

---

## 6.4. Quadruped Robot Locomotion Co-Design

Legged robotic systems require simultaneous optimization of leg link geometries and joint trajectory controllers to maximize speed and stability over rough terrain {cite:p}`fletcher2026implicit`.

- **Geometric Variables:** Upper-leg and lower-leg link lengths.
- **Feasibility Constraints:** Ground-contact kinematic equality constraint ensuring feet maintain level contact with the terrain in the reference pose.
- **Results:** Control co-design optimization converged to a **34.5% increase in distance traveled** relative to baseline geometry with optimal control.

---

## 6.5. Blended-Wing-Body (BWB) Aircraft under Uncertainty (MDOUU)

Blended-Wing-Body aircraft integrate the fuselage and wing into an aerodynamically efficient lifting body. Scotzniovsky and Hwang applied `lsdo_geo` to an aeroelastic MDO under uncertainty (MDOUU) problem subject to atmospheric turbulence and structural stiffness variability {cite:p}`scotzniovsky2025panel`.

- **Setup:** 10 geometric variables and 61 geometric constraints handled inside the parameterization solver.
- **Graph Backtracking:** Inverse-tangent sweep and dihedral definitions were symbolically backtracked out of the inner loop, cutting solver iterations by more than half.
- **Outcome:** The robust design successfully maintained aeroelastic feasibility across the entire uncertainty domain, shifting lift distribution inward toward the center body to mitigate gust-induced bending moments.

