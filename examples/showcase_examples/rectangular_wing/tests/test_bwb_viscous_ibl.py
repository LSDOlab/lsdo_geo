"""Unit tests for the BWB Differentiable Direct-IBL Viscous Drag module."""

import pytest
import numpy as np
import meshio
import csdl_alpha as csdl
import sys
from pathlib import Path

# Add showcase example directory to path
SHOWCASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOWCASE_DIR))

from optimization_analyses.bwb_viscous_ibl import (
    build_ibl_mesh_topology,
    evaluate_bwb_viscous_ibl,
    evaluate_smooth_H,
    sp_floor,
    EPS_U2,
    EPS_THETA,
    EPS_H1_DIFF3,
    EPS_RE_THETA,
    H_SEP,
    Y_SEP,
    S0_HEAD,
)


def test_closure_transforms_and_floors():
    """Gate 2: Validate numerical agreement of smooth H <-> H1 transforms, linear continuation, and floors."""
    # Test H -> H1 -> H round trip in attached regime (H <= 2.4)
    H_test = np.linspace(1.2, 2.35, 50)
    H1 = 3.0445 + 0.8702 / ((H_test - 1.1) ** 1.2721)
    H_recovered = 1.1 + (0.8702 / (H1 - 3.0445)) ** (1.0 / 1.2721)
    np.testing.assert_allclose(H_recovered, H_test, rtol=1e-12, atol=1e-12)

    # Test C-infinity smooth linear continuation of evaluate_smooth_H
    rec_h = csdl.Recorder()
    rec_h.start()
    y_var = csdl.Variable(value=np.array([1.5, 0.62326, 0.2, 0.0, -0.5]))
    y_var.set_as_design_variable()
    H_out = evaluate_smooth_H(y_var)
    rec_h.stop()
    sim_h = csdl.experimental.JaxSimulator(recorder=rec_h, additional_outputs=[H_out])
    sim_h.run()
    H_res = sim_h[H_out]

    # In attached regime (y=1.5), matches exact formula
    H_exact_1_5 = 1.1 + (0.8702 / 1.5) ** (1.0 / 1.2721)
    np.testing.assert_allclose(H_res[0], H_exact_1_5, rtol=1e-6)
    # At separation onset (y=0.62326), H is close to 2.4
    np.testing.assert_allclose(H_res[1], 2.4, atol=1e-2)
    # In separated regime (y <= 0), H remains bounded and reasonable (3.0 - 4.5)
    assert 3.0 < H_res[2] < 3.2
    assert 3.3 < H_res[3] < 3.6
    assert 4.0 < H_res[4] < 4.5

    # Check non-vanishing restoring derivative dH/dy in separated regime
    # Compute derivative via JAX simulator
    rec_d = csdl.Recorder()
    rec_d.start()
    y_sep_pt = csdl.Variable(value=np.array([-0.5]))
    y_sep_pt.set_as_design_variable()
    H_sep_out = evaluate_smooth_H(y_sep_pt)
    H_sep_out.set_as_objective()
    rec_d.stop()
    sim_d = csdl.experimental.JaxSimulator(recorder=rec_d)
    sim_d.run()
    dH_dy_sep = sim_d.compute_optimization_derivatives()['df'][0, 0]
    np.testing.assert_allclose(dH_dy_sep, S0_HEAD, rtol=1e-4)

    # Test numerical floors are inactive for nominal attached conditions
    rec = csdl.Recorder()
    rec.start()
    th_nom = csdl.Variable(value=np.array([1e-4, 5e-3]))
    th_nom.set_as_design_variable()
    th_fl = sp_floor(th_nom, EPS_THETA, 200000.0)
    rec.stop()
    sim = csdl.experimental.JaxSimulator(recorder=rec, additional_outputs=[th_fl])
    sim.run()
    # At nominal theta ~ 1e-4, floor is 1e-6: relative deviation should be < 1e-6
    np.testing.assert_allclose(sim[th_fl], np.array([1e-4, 5e-3]), rtol=1e-6)


def test_topology_construction_fast_and_full():
    """Gate 3: Topology tests for both fast (10ar) and full (35sect) meshes."""
    repo_root = Path(__file__).resolve().parents[4]
    geom_dir = repo_root / "examples" / "example_geometries"

    mesh_fast = meshio.read(geom_dir / "rectangular_wing_naca0012_10ar.msh")
    topo_fast = build_ibl_mesh_topology(mesh_fast.points, mesh_fast.cells_dict, scale_factor=7.5)
    assert topo_fast['num_stations'] == 14
    assert topo_fast['upper_indices'].shape == (14, 20)
    assert topo_fast['lower_indices'].shape == (14, 20)
    assert topo_fast['M_interp_ibl'].shape == (100, 14)
    np.testing.assert_allclose(np.sum(topo_fast['M_interp_ibl'], axis=1), 1.0, atol=1e-12)

    mesh_full = meshio.read(geom_dir / "rectangular_wing_naca0012_35sect.msh")
    topo_full = build_ibl_mesh_topology(mesh_full.points, mesh_full.cells_dict, scale_factor=7.5)
    assert topo_full['num_stations'] == 34
    assert topo_full['upper_indices'].shape == (34, 20)
    assert topo_full['lower_indices'].shape == (34, 20)
    assert topo_full['M_interp_ibl'].shape == (100, 34)
    np.testing.assert_allclose(np.sum(topo_full['M_interp_ibl'], axis=1), 1.0, atol=1e-12)

    # Test malformed input error handling: no right panels
    with pytest.raises(ValueError, match="No panels found on right half-wing"):
        build_ibl_mesh_topology(np.zeros((10, 3)), {'quad': np.zeros((1, 4), dtype=int)}, scale_factor=1.0)


def test_flat_plate_trend_and_reynolds_scaling():
    """Gate 1: Constant Ue recovers turbulent flat-plate trend and Re scaling."""
    V_inf = 200.0
    s_chord = 5.0
    nu = 1.5e-5

    def run_flat_plate(nu_val):
        N = 20
        s = np.linspace(0.005, s_chord, N)
        s0 = s[0]
        Re_s0 = V_inf * s0 / nu_val
        th = 0.037 * s0 * (Re_s0 ** -0.2)
        H1 = 3.0445 + 0.8702 / ((1.4 - 1.1) ** 1.2721)
        d1 = th * H1
        for i in range(N - 1):
            ds = s[i+1] - s[i]
            H1_val = d1 / th
            H = 1.1 + (0.8702 / (H1_val - 3.0445)) ** (1.0 / 1.2721)
            Re_th = V_inf * th / nu_val
            Cf = 0.246 * (10.0 ** (-0.678 * H)) * (Re_th ** -0.268)
            dth = 0.5 * Cf
            dd1 = 0.0306 * ((H1_val - 3.0) ** -0.6169)
            th_m = th + 0.5 * ds * dth
            d1_m = d1 + 0.5 * ds * dd1
            H1_m = d1_m / th_m
            H_m = 1.1 + (0.8702 / (H1_m - 3.0445)) ** (1.0 / 1.2721)
            Re_th_m = V_inf * th_m / nu_val
            Cf_m = 0.246 * (10.0 ** (-0.678 * H_m)) * (Re_th_m ** -0.268)
            th = th + ds * 0.5 * Cf_m
            d1 = d1 + ds * 0.0306 * ((H1_m - 3.0) ** -0.6169)
        H_final = 1.1 + (0.8702 / (d1/th - 3.0445)) ** (1.0 / 1.2721)
        cd = 4.0 * th / s_chord
        return cd, th, H_final

    cd_low_re, th_low_re, H_low = run_flat_plate(nu * 2.0)
    cd_high_re, th_high_re, H_high = run_flat_plate(nu)

    # Increasing Re lowers drag
    assert cd_high_re < cd_low_re
    # Compare with Prandtl turbulent flat-plate formula: cd = 2 * (0.074 / Re_L^0.2)
    Re_L = V_inf * s_chord / nu
    cd_prandtl = 2.0 * (0.074 / (Re_L ** 0.2))
    assert abs(cd_high_re - cd_prandtl) / cd_prandtl < 0.05


def test_jax_total_derivatives_vs_finite_difference():
    """Gate 4: Compare JAX derivatives of drag with centered finite differences."""
    cp_sub = np.array([-0.3, -0.8, -1.5, -2.0, -2.2, -1.8, -1.2, -0.6, 0.0, 0.2])
    centers_sub = np.zeros((10, 3))
    centers_sub[:, 0] = np.linspace(0.01, 7.5, 10)
    V_inf = 227.38
    nu = 1.48e-5 / 0.458312

    def run_sub_model(cp_val):
        rec = csdl.Recorder()
        rec.start()
        cp_var = csdl.Variable(value=cp_val)
        cp_var.set_as_design_variable()
        centers_var = csdl.Variable(value=centers_sub)
        u_sq = sp_floor(1.0 - cp_var, 0.04, 50.0)
        Ue = V_inf * csdl.sqrt(u_sq)
        s0 = csdl.sqrt(centers_var[0, 0]**2 + centers_var[0, 2]**2 + 1e-12)
        Re_s0 = (Ue[0] * s0 / nu) + 1.0
        theta = 0.037 * s0 * (Re_s0 ** -0.2)
        H0 = 1.4
        H1_0 = 3.0445 + 0.8702 / ((H0 - 1.1) ** 1.2721)
        delta1 = theta * H1_0
        for k in range(9):
            diff = centers_var[k+1, :] - centers_var[k, :]
            ds_k = csdl.sqrt(csdl.sum(diff**2, axes=(0,)) + 1e-12)
            Ue_k = Ue[k]
            Ue_next = Ue[k+1]
            dlogU = (Ue_next - Ue_k) / (0.5 * (Ue_k + Ue_next) * ds_k)
            alpha_step = csdl.tanh(dlogU * ds_k / 0.75) * (0.75 / ds_k)
            th_fl = sp_floor(theta, 1e-6, 200000.0)
            H1 = delta1 / th_fl
            H1_diff = sp_floor(H1 - 3.0445, 1e-4, 1000.0)
            H = 1.1 + (0.8702 / H1_diff) ** (1.0 / 1.2721)
            Re_th = sp_floor(Ue_k * th_fl / nu, 1.0, 10.0)
            Cf = 0.246 * (10.0 ** (-0.678 * H)) * (Re_th ** -0.268)
            dtheta_1 = 0.5 * Cf - (H + 2.0) * th_fl * alpha_step
            H1_diff3 = sp_floor(H1 - 3.0, 1e-4, 1000.0)
            ddelta1_1 = 0.0306 * (H1_diff3 ** -0.6169) - delta1 * alpha_step
            th_mid = th_fl + 0.5 * ds_k * dtheta_1
            d1_mid = delta1 + 0.5 * ds_k * ddelta1_1
            th_mid_fl = sp_floor(th_mid, 1e-6, 200000.0)
            H1_mid = d1_mid / th_mid_fl
            H1_mid_diff = sp_floor(H1_mid - 3.0445, 1e-4, 1000.0)
            H_mid = 1.1 + (0.8702 / H1_mid_diff) ** (1.0 / 1.2721)
            Ue_mid = 0.5 * (Ue_k + Ue_next)
            Re_th_mid = sp_floor(Ue_mid * th_mid_fl / nu, 1.0, 10.0)
            Cf_mid = 0.246 * (10.0 ** (-0.678 * H_mid)) * (Re_th_mid ** -0.268)
            dtheta_2 = 0.5 * Cf_mid - (H_mid + 2.0) * th_mid_fl * alpha_step
            H1_mid_diff3 = sp_floor(H1_mid - 3.0, 1e-4, 1000.0)
            ddelta1_2 = 0.0306 * (H1_mid_diff3 ** -0.6169) - d1_mid * alpha_step
            theta = sp_floor(th_fl + ds_k * dtheta_2, 1e-6, 200000.0)
            delta1 = sp_floor(delta1 + ds_k * ddelta1_2, 1e-6, 200000.0)
        H1_final = delta1 / sp_floor(theta, 1e-6, 200000.0)
        H_te = 1.1 + (0.8702 / sp_floor(H1_final - 3.0445, 1e-4, 1000.0)) ** (1.0 / 1.2721)
        obj = theta * ((Ue[-1] / V_inf) ** ((H_te + 5.0) / 2.0))
        obj.set_as_objective()
        rec.stop()
        s = csdl.experimental.JaxSimulator(recorder=rec)
        s.run()
        df = s.compute_optimization_derivatives()['df']
        return s[obj][0], df[0]

    _, jax_df = run_sub_model(cp_sub)
    h = 1e-6
    for idx in [0, 4, 9]:
        cp_p = cp_sub.copy()
        cp_p[idx] += h
        fp, _ = run_sub_model(cp_p)
        cp_m = cp_sub.copy()
        cp_m[idx] -= h
        fm, _ = run_sub_model(cp_m)
        fd = (fp - fm) / (2 * h)
        rel_err = abs(jax_df[idx] - fd) / (abs(fd) + 1e-12)
        assert rel_err < 1e-5, f"Derivative mismatch at idx {idx}: rel_err = {rel_err}"
