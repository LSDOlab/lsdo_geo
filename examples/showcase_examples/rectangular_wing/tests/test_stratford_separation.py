import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import pytest
import numpy as np
import meshio
import csdl_alpha as csdl
from examples.showcase_examples.rectangular_wing.optimization_analyses.stratford_separation import (
    build_stratford_topology,
    evaluate_stratford_separation,
    S_CRIT,
)


@pytest.fixture(scope="module")
def fast_mesh():
    return meshio.read("examples/example_geometries/rectangular_wing_naca0012_10ar.msh")


@pytest.fixture(scope="module")
def full_mesh():
    return meshio.read("examples/example_geometries/rectangular_wing_naca0012_35sect.msh")


def test_stratford_topology_shapes(fast_mesh, full_mesh):
    """Verify static topology, B-spline matrices, and slice counts for fast (20) and full (32)."""
    # Fast resolution: 5 FFD stations -> 4x5 = 20 constraints
    topo_fast = build_stratford_topology(
        points=fast_mesh.points * 7.5,
        cells_dict=fast_mesh.cells_dict,
        num_ffd_stations=5,
        scale_factor=1.0,
    )
    assert topo_fast['num_ffd_stations'] == 5
    assert topo_fast['num_mesh_stations'] == 14
    assert topo_fast['num_constraints'] == 20
    assert len(topo_fast['u_slices']) == 4
    assert len(topo_fast['v_slices']) == 5
    assert topo_fast['Mu_inv'].shape == (6, 20)
    assert topo_fast['Mv_inv_T'].shape == (15, 8)
    assert topo_fast['B_dense'].shape == (31 * 41, 48)

    # Full resolution: 8 FFD stations -> 4x8 = 32 constraints
    topo_full = build_stratford_topology(
        points=full_mesh.points * 7.5,
        cells_dict=full_mesh.cells_dict,
        num_ffd_stations=8,
        scale_factor=1.0,
    )
    assert topo_full['num_ffd_stations'] == 8
    assert topo_full['num_mesh_stations'] == 34
    assert topo_full['num_constraints'] == 32
    assert len(topo_full['u_slices']) == 4
    assert len(topo_full['v_slices']) == 8
    assert topo_full['Mu_inv'].shape == (6, 20)
    assert topo_full['Mv_inv_T'].shape == (35, 8)
    assert topo_full['B_dense'].shape == (31 * 41, 48)


def test_stratford_evaluation_and_derivatives(fast_mesh):
    """Verify evaluation outputs and exact JAX analytical derivatives vs finite differences."""
    topo = build_stratford_topology(
        points=fast_mesh.points * 7.5,
        cells_dict=fast_mesh.cells_dict,
        num_ffd_stations=5,
        scale_factor=1.0,
    )

    pts = fast_mesh.points * 7.5
    quads = fast_mesh.cells_dict['quad']
    total_quads = len(quads)
    centers = pts[quads].mean(axis=1)

    # Synthetic Cp with mild adverse recovery (attached flow: Cp from -0.3 to 0.0)
    cp_init = np.zeros(total_quads)
    for st in range(topo['num_mesh_stations']):
        u_idx = topo['ibl_topology']['upper_indices'][st]
        cp_init[u_idx] = np.linspace(-0.3, 0.0, 20)

    recorder = csdl.Recorder()
    recorder.start()

    cp_var = csdl.Variable(value=cp_init)
    centers_var = csdl.Variable(value=centers)
    v_cruise = csdl.Variable(value=230.0)
    rho_cruise = csdl.Variable(value=0.38)
    mu_air = csdl.Variable(value=1.4e-5)

    res = evaluate_stratford_separation(
        cp_node0=cp_var,
        dynamic_panel_centers=centers_var,
        v_cruise=v_cruise,
        rho_cruise=rho_cruise,
        mu_air=mu_air,
        stratford_topology=topo,
    )

    constrs = res['dv_stratford_constraints']
    margin = res['stratford_margin']
    max_s = res['max_stratford']

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[cp_var],
        additional_outputs=[constrs, margin, max_s],
        gpu=False,
    )
    sim[cp_var] = cp_init
    sim.run()

    constrs_val = sim[constrs]
    margin_val = float(np.asarray(sim[margin]).flatten()[0])
    max_s_val = float(np.asarray(sim[max_s]).flatten()[0])

    assert constrs_val.shape == (20,)
    assert np.all(np.isfinite(constrs_val))
    assert np.isfinite(margin_val)
    assert np.isfinite(max_s_val)
    # Mild attached flow: max_s < S_CRIT (0.39), margin > 0
    assert max_s_val < S_CRIT
    assert margin_val > 0.0

    # Total derivative verification vs centered finite difference
    totals = sim.check_totals(step_size=1e-5, print_results=False)
    for (of_var, wrt_var), info in totals.items():
        rel_err = float(info['rel_error'])
        assert rel_err < 1e-3, f"Derivative error too high for {info.get('of_name')} wrt {info.get('wrt_name')}: {rel_err}"


def test_stratford_favorable_gradient_derivatives(fast_mesh):
    """Verify that accelerating flow / favorable pressure gradients do not produce NaN/Inf in derivatives."""
    topo = build_stratford_topology(
        points=fast_mesh.points * 7.5,
        cells_dict=fast_mesh.cells_dict,
        num_ffd_stations=5,
        scale_factor=1.0,
    )

    pts = fast_mesh.points * 7.5
    quads = fast_mesh.cells_dict['quad']
    total_quads = len(quads)
    centers = pts[quads].mean(axis=1)

    # Cp profile with strong acceleration (favorable gradient, grad < 0) over front 4 panels
    cp_init = np.zeros(total_quads)
    for st in range(topo['num_mesh_stations']):
        u_idx = topo['ibl_topology']['upper_indices'][st]
        cp_prof = np.zeros(20)
        cp_prof[:4] = np.linspace(0.8, -1.5, 4)
        cp_prof[4:] = np.linspace(-1.5, 0.1, 16)
        cp_init[u_idx] = cp_prof

    recorder = csdl.Recorder()
    recorder.start()

    cp_var = csdl.Variable(value=cp_init)
    centers_var = csdl.Variable(value=centers)
    v_cruise = csdl.Variable(value=230.0)
    rho_cruise = csdl.Variable(value=0.38)
    mu_air = csdl.Variable(value=1.4e-5)

    res = evaluate_stratford_separation(
        cp_node0=cp_var,
        dynamic_panel_centers=centers_var,
        v_cruise=v_cruise,
        rho_cruise=rho_cruise,
        mu_air=mu_air,
        stratford_topology=topo,
    )

    constrs = res['dv_stratford_constraints']
    constrs.set_as_constraint(upper=S_CRIT)
    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[cp_var],
        additional_outputs=[constrs],
        gpu=False,
    )
    sim[cp_var] = cp_init
    sim.run()

    constrs_val = sim[constrs]
    assert np.all(np.isfinite(constrs_val))

    totals = sim.compute_totals()
    for (of_var, wrt_var), v in totals.items():
        assert not np.isnan(v).any(), f"NaN in {of_var.name} wrt {wrt_var.name}"
        assert not np.isinf(v).any(), f"Inf in {of_var.name} wrt {wrt_var.name}"

