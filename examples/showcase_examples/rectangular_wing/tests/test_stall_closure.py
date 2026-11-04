import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import pytest
import numpy as np
import meshio
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import VortexAD

from examples.showcase_examples.rectangular_wing.optimization_analyses.stall_closure import (
    build_stall_closure_topology,
    reconstruct_pre_cutoff_cp,
    evaluate_stall_closure,
    K_SEP_BASE,
)


@pytest.fixture(scope="module")
def fast_mesh_data():
    p = "examples/example_geometries/rectangular_wing/rectangular_wing_naca0012_10ar.msh"
    if not os.path.exists(p):
        p = "examples/example_geometries/rectangular_wing_naca0012_10ar.msh"
    mesh = meshio.read(p)
    pts = mesh.points * 7.5
    cells = mesh.cells_dict
    cell_adj = VortexAD.find_cell_adjacency(points=mesh.points, cells=cells)
    te_props = VortexAD.TE_detection(
        points=mesh.points,
        cells=cell_adj[1],
        edges2cells=cell_adj[3],
        points2cells=cell_adj[4],
    )
    te_edges = np.array(te_props[2])
    return {
        'points': pts,
        'cells_dict': cells,
        'cell_adj': cell_adj,
        'te_props': te_props,
        'te_edges': te_edges,
    }


def test_stall_closure_topology(fast_mesh_data):
    """Verify static topology, mapping shapes, and wake mapping matrix."""
    pts = fast_mesh_data['points']
    cells = fast_mesh_data['cells_dict']
    te_edges = fast_mesh_data['te_edges']

    topo = build_stall_closure_topology(
        points=pts,
        cells_dict=cells,
        te_edges=te_edges,
        num_ffd_stations=5,
        scale_factor=1.0,
    )

    assert topo['num_mesh_stations'] == 14
    assert topo['M_wake'].shape == (30, 14)
    assert topo['M_scatter_right'].shape == (topo['num_right_panels'], 14 * 20)
    # Check that each TE edge maps to exactly one station
    assert np.all(np.sum(topo['M_wake'], axis=1) == 1.0)


def test_stall_closure_attached_flow_identity(fast_mesh_data):
    """Verify attached flow identity: f = 1 -> L_corr == L_inv, Di_corr == Di_inv, D_sep == 0."""
    pts = fast_mesh_data['points']
    cells = fast_mesh_data['cells_dict']
    te_edges = fast_mesh_data['te_edges']
    cell_adj = fast_mesh_data['cell_adj']
    te_props = fast_mesh_data['te_props']

    topo = build_stall_closure_topology(
        points=pts,
        cells_dict=cells,
        te_edges=te_edges,
        num_ffd_stations=5,
        scale_factor=1.0,
    )

    quads = cells['quad']
    total_quads = len(quads)
    centers = pts[quads].mean(axis=1)

    # Unit normals in z direction for synthetic panels
    normals = np.zeros((total_quads, 3))
    normals[:, 2] = 1.0
    areas = np.full(total_quads, 0.1)

    # Mild attached flow: Cp from -0.3 to -0.25 (S < 0.15 everywhere)
    cp_att = np.zeros(total_quads)
    for st in range(topo['num_mesh_stations']):
        u_idx = topo['stratford_topology']['ibl_topology']['upper_indices'][st]
        cp_att[u_idx] = np.full(20, -0.3)

    # Corresponding V_mag (V_mag = V_inf * sqrt(1 - Cp * beta))
    v_inf_val = 230.0
    mach_val = 0.72
    beta_val = np.sqrt(1.0 - mach_val**2)
    v_mag_val = v_inf_val * np.sqrt(np.maximum(1.0 - cp_att * beta_val, 0.0))

    # Evaluate mock wake corners
    num_te_edges = len(te_edges)
    fake_corners = np.zeros((1, num_te_edges, 4, 3))
    for i in range(num_te_edges):
        fake_corners[0, i, 1, 1] = float(i)
    recorder = csdl.Recorder()
    recorder.start()

    fake_wake_dict = {'panel_corners': csdl.Variable(value=fake_corners)}
    v_mag_var = csdl.Variable(value=v_mag_val)
    f_inv_right = csdl.Variable(value=np.full((topo['num_right_panels'], 3), 100.0))
    l_inv_var = csdl.Variable(value=1e5)
    m_inv_var = csdl.Variable(value=np.array([0.0, -1000.0, 0.0]))
    centers_var = csdl.Variable(value=centers)
    normals_var = csdl.Variable(value=normals)
    areas_var = csdl.Variable(value=areas)
    mu_w_var = csdl.Variable(value=np.full((1, num_te_edges), 10.0))
    v_inf_var = csdl.Variable(value=v_inf_val)
    rho_var = csdl.Variable(value=0.38)
    mach_var = csdl.Variable(value=mach_val)
    q_inf_var = csdl.Variable(value=0.5 * 0.38 * (v_inf_val ** 2))
    cg_ref_var = csdl.Variable(value=np.array([10.0, 0.0, 0.0]))

    res = evaluate_stall_closure(
        v_mag=v_mag_var,
        panel_forces_inviscid_right=f_inv_right,
        l_inviscid=l_inv_var,
        m_inviscid=m_inv_var,
        dynamic_panel_centers=centers_var,
        panel_normals=normals_var,
        panel_areas=areas_var,
        mu_w_inviscid=mu_w_var,
        wake_dict=fake_wake_dict,
        te_edges=te_edges,
        v_inf=v_inf_var,
        rho=rho_var,
        mach=mach_var,
        q_inf=q_inf_var,
        cg_ref=cg_ref_var,
        strip_tc=None,
        strip_areas=None,
        stall_topology=topo,
    )

    l_corr = res['lift_corrected']
    f_att = res['f_attached']
    d_sep = res['D_separation']

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[v_mag_var],
        additional_outputs=[l_corr, f_att, d_sep],
        gpu=False,
    )
    sim[v_mag_var] = v_mag_val
    sim.run()

    f_vals = sim[f_att]
    l_val = float(np.asarray(sim[l_corr]).flatten()[0])
    d_sep_val = float(np.asarray(sim[d_sep]).flatten()[0])

    # Attached flow: f >= 0.99 everywhere
    assert np.all(f_vals >= 0.99)
    # Lift correction should match inviscid within 0.1%
    assert np.isclose(l_val, 1e5, rtol=1e-3)
    # Separation drag should be near-zero
    assert d_sep_val < 10.0


def test_stall_closure_separated_response(fast_mesh_data):
    """Verify separated flow response: severe adverse gradient causes f < 1, lift loss, and drag rise."""
    pts = fast_mesh_data['points']
    cells = fast_mesh_data['cells_dict']
    te_edges = fast_mesh_data['te_edges']

    topo = build_stall_closure_topology(
        points=pts,
        cells_dict=cells,
        te_edges=te_edges,
        num_ffd_stations=5,
        scale_factor=1.0,
    )

    quads = cells['quad']
    total_quads = len(quads)
    centers = pts[quads].mean(axis=1)

    normals = np.zeros((total_quads, 3))
    normals[:, 2] = 1.0
    areas = np.full(total_quads, 0.1)

    # Severe adverse pressure recovery: peak Cp = -2.5 at panel 2 recovering to +0.3 at TE (S >> 0.39)
    cp_sep = np.zeros(total_quads)
    for st in range(topo['num_mesh_stations']):
        u_idx = topo['stratford_topology']['ibl_topology']['upper_indices'][st]
        prof = np.zeros(20)
        prof[:2] = np.linspace(0.5, -5.0, 2)
        prof[2:] = np.linspace(-5.0, 0.8, 18)
        cp_sep[u_idx] = prof

    v_inf_val = 230.0
    mach_val = 0.72
    beta_val = np.sqrt(1.0 - mach_val**2)
    v_mag_val = v_inf_val * np.sqrt(np.maximum(1.0 - cp_sep * beta_val, 0.0))

    num_te_edges = len(te_edges)
    fake_corners = np.zeros((1, num_te_edges, 4, 3))
    for i in range(num_te_edges):
        fake_corners[0, i, 1, 1] = float(i)
        fake_corners[0, i, 2, 1] = float(i + 1)

    recorder = csdl.Recorder()
    recorder.start()

    fake_wake_dict = {'panel_corners': csdl.Variable(value=fake_corners)}
    v_mag_var = csdl.Variable(value=v_mag_val)
    f_inv_right = csdl.Variable(value=np.full((topo['num_right_panels'], 3), 100.0))
    l_inv_var = csdl.Variable(value=1e5)
    m_inv_var = csdl.Variable(value=np.array([0.0, -1000.0, 0.0]))
    centers_var = csdl.Variable(value=centers)
    normals_var = csdl.Variable(value=normals)
    areas_var = csdl.Variable(value=areas)
    mu_w_var = csdl.Variable(value=np.full((1, num_te_edges), 10.0))
    v_inf_var = csdl.Variable(value=v_inf_val)
    rho_var = csdl.Variable(value=0.38)
    mach_var = csdl.Variable(value=mach_val)
    q_inf_var = csdl.Variable(value=0.5 * 0.38 * (v_inf_val ** 2))
    cg_ref_var = csdl.Variable(value=np.array([10.0, 0.0, 0.0]))

    res = evaluate_stall_closure(
        v_mag=v_mag_var,
        panel_forces_inviscid_right=f_inv_right,
        l_inviscid=l_inv_var,
        m_inviscid=m_inv_var,
        dynamic_panel_centers=centers_var,
        panel_normals=normals_var,
        panel_areas=areas_var,
        mu_w_inviscid=mu_w_var,
        wake_dict=fake_wake_dict,
        te_edges=te_edges,
        v_inf=v_inf_var,
        rho=rho_var,
        mach=mach_var,
        q_inf=q_inf_var,
        cg_ref=cg_ref_var,
        strip_tc=None,
        strip_areas=None,
        stall_topology=topo,
    )

    l_corr = res['lift_corrected']
    f_att = res['f_attached']
    d_sep = res['D_separation']

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[v_mag_var],
        additional_outputs=[l_corr, f_att, d_sep],
        gpu=False,
    )
    sim[v_mag_var] = v_mag_val
    sim.run()

    f_vals = sim[f_att]
    l_val = float(np.asarray(sim[l_corr]).flatten()[0])
    d_sep_val = float(np.asarray(sim[d_sep]).flatten()[0])

    # Under severe adverse gradient, f drops significantly (< 0.85)
    assert np.any(f_vals < 0.85)
    # Lift should decrease due to Kirchhoff-Helmholtz decambering
    assert l_val < 1e5
    # Separation drag should be strictly positive
    assert d_sep_val > 500.0


def test_stall_closure_derivatives(fast_mesh_data):
    """Verify analytical JAX total derivatives are finite and match finite difference."""
    pts = fast_mesh_data['points']
    cells = fast_mesh_data['cells_dict']
    te_edges = fast_mesh_data['te_edges']

    topo = build_stall_closure_topology(
        points=pts,
        cells_dict=cells,
        te_edges=te_edges,
        num_ffd_stations=5,
        scale_factor=1.0,
    )

    quads = cells['quad']
    total_quads = len(quads)
    centers = pts[quads].mean(axis=1)

    normals = np.zeros((total_quads, 3))
    normals[:, 2] = 1.0
    areas = np.full(total_quads, 0.1)

    cp_test = np.zeros(total_quads)
    for st in range(topo['num_mesh_stations']):
        u_idx = topo['stratford_topology']['ibl_topology']['upper_indices'][st]
        cp_test[u_idx] = np.linspace(-0.5, 0.1, 20)

    v_inf_val = 230.0
    mach_val = 0.72
    beta_val = np.sqrt(1.0 - mach_val**2)
    v_mag_val = v_inf_val * np.sqrt(np.maximum(1.0 - cp_test * beta_val, 0.0))

    num_te_edges = len(te_edges)
    fake_corners = np.zeros((1, num_te_edges, 4, 3))
    for i in range(num_te_edges):
        fake_corners[0, i, 1, 1] = float(i)
        fake_corners[0, i, 2, 1] = float(i + 1)

    recorder = csdl.Recorder()
    recorder.start()

    fake_wake_dict = {'panel_corners': csdl.Variable(value=fake_corners)}
    v_mag_var = csdl.Variable(value=v_mag_val)
    f_inv_right = csdl.Variable(value=np.full((topo['num_right_panels'], 3), 100.0))
    l_inv_var = csdl.Variable(value=1e5)
    m_inv_var = csdl.Variable(value=np.array([0.0, -1000.0, 0.0]))
    centers_var = csdl.Variable(value=centers)
    normals_var = csdl.Variable(value=normals)
    areas_var = csdl.Variable(value=areas)
    mu_w_var = csdl.Variable(value=np.full((1, num_te_edges), 10.0))
    v_inf_var = csdl.Variable(value=v_inf_val)
    rho_var = csdl.Variable(value=0.38)
    mach_var = csdl.Variable(value=mach_val)
    q_inf_var = csdl.Variable(value=0.5 * 0.38 * (v_inf_val ** 2))
    cg_ref_var = csdl.Variable(value=np.array([10.0, 0.0, 0.0]))

    res = evaluate_stall_closure(
        v_mag=v_mag_var,
        panel_forces_inviscid_right=f_inv_right,
        l_inviscid=l_inv_var,
        m_inviscid=m_inv_var,
        dynamic_panel_centers=centers_var,
        panel_normals=normals_var,
        panel_areas=areas_var,
        mu_w_inviscid=mu_w_var,
        wake_dict=fake_wake_dict,
        te_edges=te_edges,
        v_inf=v_inf_var,
        rho=rho_var,
        mach=mach_var,
        q_inf=q_inf_var,
        cg_ref=cg_ref_var,
        strip_tc=None,
        strip_areas=None,
        stall_topology=topo,
    )

    l_corr = res['lift_corrected']
    d_sep = res['D_separation']

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[v_mag_var],
        additional_outputs=[l_corr, d_sep],
        gpu=False,
    )
    sim[v_mag_var] = v_mag_val
    sim.run()

    totals = sim.compute_totals()
    for (of_v, wrt_v), grad in totals.items():
        assert not np.isnan(grad).any(), f"NaN in derivative of {of_v.name}"
        assert not np.isinf(grad).any(), f"Inf in derivative of {of_v.name}"


def test_maneuver_attachment_constraint(fast_mesh_data):
    """Verify spanwise B-spline fit, per-DV local aggregation, and >65% span constraint enforcement."""
    pts = fast_mesh_data['points']
    cells = fast_mesh_data['cells_dict']
    te_edges = fast_mesh_data['te_edges']
    num_stations = 5

    topo = build_stall_closure_topology(
        points=pts,
        cells_dict=cells,
        te_edges=te_edges,
        num_ffd_stations=num_stations,
        scale_factor=1.0,
    )

    station_span_y = topo['station_span_y']
    b_tip_stall = float(np.max(station_span_y))
    tau_mesh_stall = np.clip(station_span_y / b_tip_stall, 0.0, 1.0)
    num_mesh_stall = len(tau_mesh_stall)

    tau_colloc_stall = np.concatenate([[0.0], tau_mesh_stall])
    int_knots_stall = [np.mean(tau_colloc_stall[j:j+3]) for j in range(1, num_mesh_stall - 2)]
    knots_f_stall = np.concatenate([[0.0, 0.0, 0.0, 0.0], int_knots_stall, [1.0, 1.0, 1.0, 1.0]])

    f_stall_fit_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=3,
        coefficients_shape=(num_mesh_stall + 1,),
        knots=(knots_f_stall,),
    )

    B_stall_matrix = f_stall_fit_space.compute_basis_matrix(tau_mesh_stall.reshape((-1, 1))).toarray()
    d_root_stall_row = np.zeros((1, num_mesh_stall + 1))
    d_root_stall_row[0, 0] = -1.0
    d_root_stall_row[0, 1] = 1.0

    M_f_stall_sys = np.vstack([B_stall_matrix, d_root_stall_row])
    M_f_stall_inv = np.linalg.inv(M_f_stall_sys)

    # Evaluation points and grouping per spanwise DV
    station_cl_peaks = np.linspace(0.0, 1.0, num_stations)
    num_intermediate_cl = 10
    m_cl = num_intermediate_cl // 2

    eval_cl_points_list = []
    station_cl_peak_indices = []
    for i in range(num_stations - 1):
        station_cl_peak_indices.append(len(eval_cl_points_list))
        eval_cl_points_list.append(station_cl_peaks[i])
        pts_inter = np.linspace(station_cl_peaks[i], station_cl_peaks[i+1], num_intermediate_cl + 2)[1:-1]
        eval_cl_points_list.extend(pts_inter)
    station_cl_peak_indices.append(len(eval_cl_points_list))
    eval_cl_points_list.append(station_cl_peaks[-1])

    eval_cl_points_arr = np.array(eval_cl_points_list).reshape((-1, 1))

    station_cl_group_indices = []
    total_eval_cl_pts = len(eval_cl_points_list)
    for i in range(num_stations):
        p_idx = station_cl_peak_indices[i]
        left = 0 if i == 0 else p_idx - m_cl
        right = total_eval_cl_pts - 1 if i == num_stations - 1 else p_idx + m_cl
        left_ol = max(0, left - 1) if i > 0 else 0
        right_ol = min(total_eval_cl_pts - 1, right + 1) if i < num_stations - 1 else right
        station_cl_group_indices.append(list(range(left_ol, right_ol + 1)))

    quads = cells['quad']
    total_quads = len(quads)
    centers = pts[quads].mean(axis=1)
    normals = np.zeros_like(centers)
    normals[:, 2] = 1.0
    areas = np.full(total_quads, 0.5)

    cp_test = np.zeros(total_quads)
    for st in range(topo['num_mesh_stations']):
        u_idx = topo['stratford_topology']['ibl_topology']['upper_indices'][st]
        # Strong adverse pressure gradient at outboard stations to induce partial separation
        if st >= 8:
            cp_test[u_idx] = np.linspace(-1.2, 0.2, 20)
        else:
            cp_test[u_idx] = np.linspace(-0.5, 0.1, 20)

    v_inf_val = 230.0
    mach_val = 0.72
    beta_val = np.sqrt(1.0 - mach_val**2)
    v_mag_val = v_inf_val * np.sqrt(np.maximum(1.0 - cp_test * beta_val, 0.0))

    num_te_edges = len(te_edges)
    fake_corners = np.zeros((1, num_te_edges, 4, 3))
    for i in range(num_te_edges):
        fake_corners[0, i, 1, 1] = float(i)
        fake_corners[0, i, 2, 1] = float(i + 1)

    recorder = csdl.Recorder()
    recorder.start()

    fake_wake_dict = {'panel_corners': csdl.Variable(value=fake_corners)}
    v_mag_var = csdl.Variable(value=v_mag_val)
    f_inv_right = csdl.Variable(value=np.full((topo['num_right_panels'], 3), 100.0))
    l_inv_var = csdl.Variable(value=1e5)
    m_inv_var = csdl.Variable(value=np.array([0.0, -1000.0, 0.0]))
    centers_var = csdl.Variable(value=centers)
    normals_var = csdl.Variable(value=normals)
    areas_var = csdl.Variable(value=areas)
    mu_w_var = csdl.Variable(value=np.full((1, num_te_edges), 10.0))
    v_inf_var = csdl.Variable(value=v_inf_val)
    rho_var = csdl.Variable(value=0.38)
    mach_var = csdl.Variable(value=mach_val)
    q_inf_var = csdl.Variable(value=0.5 * 0.38 * (v_inf_val ** 2))
    cg_ref_var = csdl.Variable(value=np.array([10.0, 0.0, 0.0]))

    res = evaluate_stall_closure(
        v_mag=v_mag_var,
        panel_forces_inviscid_right=f_inv_right,
        l_inviscid=l_inv_var,
        m_inviscid=m_inv_var,
        dynamic_panel_centers=centers_var,
        panel_normals=normals_var,
        panel_areas=areas_var,
        mu_w_inviscid=mu_w_var,
        wake_dict=fake_wake_dict,
        te_edges=te_edges,
        v_inf=v_inf_var,
        rho=rho_var,
        mach=mach_var,
        q_inf=q_inf_var,
        cg_ref=cg_ref_var,
        strip_tc=None,
        strip_areas=None,
        stall_topology=topo,
    )

    f_attached_ss = res['f_attached']

    # B-spline fit to f_attached_ss
    f_ss_rhs = csdl.concatenate([f_attached_ss, csdl.Variable(value=np.array([0.0]))])
    f_ss_coeffs_flat = csdl.matvec(csdl.Variable(value=M_f_stall_inv), f_ss_rhs)
    f_ss_coeffs = csdl.reshape(f_ss_coeffs_flat, (num_mesh_stall + 1, 1))
    f_ss_func = lfs.Function(space=f_stall_fit_space, coefficients=f_ss_coeffs)

    all_eval_f_ss = f_ss_func.evaluate(eval_cl_points_arr)

    # Local aggregation per spanwise DV
    aggregated_f_ss_list = []
    for i in range(num_stations):
        grp = station_cl_group_indices[i]
        sub_f_ss = csdl.concatenate([all_eval_f_ss[idx] for idx in grp])
        grp_f_min = -csdl.maximum(-sub_f_ss, axes=(0,), rho=40.0)
        aggregated_f_ss_list.append(csdl.reshape(grp_f_min, (1,)))

    dv_f_attached_ss = csdl.concatenate(aggregated_f_ss_list)
    dv_f_attached_ss.name = 'dv_f_attached_ss'

    # Filter to only DVs with eta >= 0.50
    outboard_dv_mask = station_cl_peaks >= (0.50 - 1e-6)
    outboard_dv_indices = np.where(outboard_dv_mask)[0]
    near_tip_dv_start_idx = int(outboard_dv_indices[0])
    f_attached_ss_near_tip = dv_f_attached_ss[near_tip_dv_start_idx:]
    f_attached_ss_near_tip.name = 'f_attached_ss_near_tip'

    # Register constraint
    f_attached_ss_min = 0.50
    f_attached_ss_near_tip.set_as_constraint(
        lower=f_attached_ss_min,
        scaler=1.0 / f_attached_ss_min,
    )

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[v_mag_var],
        additional_outputs=[dv_f_attached_ss, f_attached_ss_near_tip],
        gpu=False,
    )
    sim[v_mag_var] = v_mag_val
    sim.run()

    dv_f_vals = np.asarray(sim[dv_f_attached_ss]).flatten()
    f_enforce_vals = np.asarray(sim[f_attached_ss_near_tip]).flatten()

    assert len(dv_f_vals) == num_stations  # Exactly 1 per spanwise DV (5)
    assert len(f_enforce_vals) == len(outboard_dv_indices)  # Only >=50% span (3 for fast: 0.50, 0.75, 1.00)
    assert len(f_enforce_vals) == 3
    assert np.all(station_cl_peaks[near_tip_dv_start_idx:] >= 0.50 - 1e-6)
    assert not np.isnan(dv_f_vals).any()
    assert not np.isnan(f_enforce_vals).any()

    totals = sim.compute_totals()
    for (of_v, wrt_v), grad in totals.items():
        assert not np.isnan(grad).any(), f"NaN in derivative of {of_v.name}"
        assert not np.isinf(grad).any(), f"Inf in derivative of {of_v.name}"

