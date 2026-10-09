"""Differentiable Static-Separation Stall Closure and Corrected Trefftz Drag.

Implements Stratford-based turbulent separation detection, Kirchhoff-Helmholtz
normal force deficit closure, corrected surface forces/moments, and corrected
2D Trefftz plane induced drag via virtual wake decambering.
"""

from __future__ import annotations
import numpy as np
import csdl_alpha as csdl
from typing import Dict, Any, Tuple
from VortexAD.core.pm.trefftz_plane import trefftz_plane_drag_2d
from .bwb_viscous_ibl import sp_floor
from .stratford_separation import (
    build_stratford_topology,
    S_CRIT,
    RHO_SMOOTH_MAX,
)

# Physical & numerical constants
K_SEP_BASE = 0.18         # Calibrated fixed-transition NACA 0012 post-stall drag constant
TC_REF = 0.12             # Reference thickness-to-chord ratio (NACA 0012)
K_LOGISTIC = 40.0         # Sharpness of logistic separation activation
F_ATTACHED_FLOOR = 0.01   # Smooth floor for attached fraction f
EPS_AREA_SEP = 1e-6       # Area floor for separated normal force redistribution


def build_stall_closure_topology(
    points: np.ndarray,
    cells_dict: Dict[str, np.ndarray],
    te_edges: np.ndarray,
    num_ffd_stations: int,
    scale_factor: float = 1.0,
    b_tip_ref: float | None = None,
) -> Dict[str, Any]:
    """Construct static topology and mapping matrices for stall closure.

    Parameters
    ----------
    points : np.ndarray
        Mesh points.
    cells_dict : dict
        Mesh cells dictionary containing 'quad' connectivity.
    te_edges : np.ndarray
        Trailing-edge node pairs from VortexAD TE detection, shape (num_TE_edges, 2).
    num_ffd_stations : int
        Number of FFD design variable spanwise stations (5 fast, 8 full).
    scale_factor : float
        Wing scale factor.
    b_tip_ref : float, optional
        Reference wing tip y-coordinate.

    Returns
    -------
    dict
        Static matrices and index maps for stall closure.
    """
    stratford_topo = build_stratford_topology(
        points=points,
        cells_dict=cells_dict,
        num_ffd_stations=num_ffd_stations,
        scale_factor=scale_factor,
        b_tip_ref=b_tip_ref,
    )

    num_mesh_stations = stratford_topo['num_mesh_stations']
    ibl_topo = stratford_topo['ibl_topology']
    station_span_y = ibl_topo['station_span_y']
    upper_indices = ibl_topo['upper_indices']  # shape (num_mesh_stations, 20)

    pts_scaled = points * scale_factor
    quads = cells_dict['quad']
    total_quads = len(quads)
    centers = pts_scaled[quads].mean(axis=1)

    # Number of panels on right wing (y > 0)
    num_right_panels = int(np.sum(centers[:, 1] > 1e-4))

    # Construct static wake mapping matrix: maps r_gamma (num_mesh_stations,)
    # to all trailing edge edges (num_TE_edges,)
    num_te_edges = len(te_edges)
    te_pts = pts_scaled[te_edges]
    te_centers_y = te_pts.mean(axis=1)[:, 1]

    M_wake = np.zeros((num_te_edges, num_mesh_stations))
    for edge_i, y_e in enumerate(te_centers_y):
        y_abs = abs(y_e)
        diffs = np.abs(station_span_y - y_abs)
        closest_st = int(np.argmin(diffs))
        M_wake[edge_i, closest_st] = 1.0

    # Scatter matrix: maps upper gathered panel force increments
    # of shape (num_mesh_stations * 20,) to flat right panels of shape (num_right_panels,)
    flat_upper_idx = upper_indices.flatten()
    M_scatter_right = np.zeros((num_right_panels, len(flat_upper_idx)))
    for col_i, panel_idx in enumerate(flat_upper_idx):
        if panel_idx < num_right_panels:
            M_scatter_right[panel_idx, col_i] = 1.0

    return {
        'stratford_topology': stratford_topo,
        'num_mesh_stations': num_mesh_stations,
        'num_right_panels': num_right_panels,
        'total_quads': total_quads,
        'M_wake': M_wake,
        'M_scatter_right': M_scatter_right,
        'station_span_y': station_span_y,
    }


def reconstruct_pre_cutoff_cp(
    v_mag: csdl.Variable,
    v_inf: csdl.Variable,
    mach: csdl.Variable,
) -> csdl.Variable:
    """Reconstruct unclipped inviscid pressure coefficient from velocity magnitude.

    Applies the Prandtl-Glauert compressibility correction without artificial
    numerical Cp clipping to resolve physical suction peaks.
    """
    v_ratio_sq = (v_mag / v_inf) ** 2
    cp_incomp = 1.0 - v_ratio_sq
    mach_sq = mach ** 2
    beta = csdl.sqrt(sp_floor(1.0 - mach_sq, 0.05, 50.0))
    return cp_incomp / beta


def evaluate_stall_closure(
    v_mag: csdl.Variable,
    panel_forces_inviscid_right: csdl.Variable,
    l_inviscid: csdl.Variable,
    m_inviscid: csdl.Variable,
    dynamic_panel_centers: csdl.Variable,
    panel_normals: csdl.Variable,
    panel_areas: csdl.Variable,
    mu_w_inviscid: csdl.Variable,
    wake_dict: dict,
    te_edges: np.ndarray,
    v_inf: csdl.Variable,
    rho: csdl.Variable,
    mach: csdl.Variable,
    q_inf: csdl.Variable,
    cg_ref: csdl.Variable,
    strip_tc: csdl.Variable | None,
    strip_areas: csdl.Variable | None,
    stall_topology: dict,
) -> dict:
    """Differentiable static separation stall closure across aerodynamic surfaces and wake.

    Parameters
    ----------
    v_mag : csdl.Variable
        Collocation velocity magnitude on panels, shape (num_panels,).
    panel_forces_inviscid_right : csdl.Variable
        Inviscid 3D panel forces on right half-wing, shape (num_right_panels, 3).
    l_inviscid : csdl.Variable
        Inviscid total lift (N), scalar.
    m_inviscid : csdl.Variable
        Inviscid pitching moment (Nm), shape (3,) or scalar.
    dynamic_panel_centers : csdl.Variable
        Deformed panel center coordinates, shape (num_panels, 3).
    panel_normals : csdl.Variable
        Panel outward normal unit vectors, shape (num_panels, 3).
    panel_areas : csdl.Variable
        Panel surface areas (m^2), shape (num_panels,).
    mu_w_inviscid : csdl.Variable
        Inviscid trailing-edge doublet jump vector, shape (1, num_TE_edges).
    wake_dict : dict
        VortexAD wake mesh dictionary containing 'panel_corners'.
    te_edges : np.ndarray
        Trailing-edge node pairs.
    v_inf : csdl.Variable
        Freestream airspeed (m/s).
    rho : csdl.Variable
        Air density (kg/m^3).
    mach : csdl.Variable
        Freestream Mach number.
    q_inf : csdl.Variable
        Dynamic pressure (Pa).
    cg_ref : csdl.Variable
        Moment reference center of mass [x_cg, 0, z_cg].
    strip_tc : csdl.Variable, optional
        Sectional thickness-to-chord ratios.
    strip_areas : csdl.Variable, optional
        Sectional strip planform areas.
    stall_topology : dict
        Static topology dictionary from build_stall_closure_topology.

    Returns
    -------
    dict
        Corrected forces, moments, wake, Trefftz drag, and diagnostics.
    """
    strat_topo = stall_topology['stratford_topology']
    ibl_topo = strat_topo['ibl_topology']
    num_mesh_stations = stall_topology['num_mesh_stations']
    num_right_panels = stall_topology['num_right_panels']
    station_span_y = stall_topology['station_span_y']

    M_gather_var = csdl.Variable(value=ibl_topo['M_gather'])
    M_wake_var = csdl.Variable(value=stall_topology['M_wake'])
    M_scatter_var = csdl.Variable(value=stall_topology['M_scatter_right'])

    # 1. Reconstruct unclipped Cp
    cp_uncut = reconstruct_pre_cutoff_cp(v_mag=v_mag, v_inf=v_inf, mach=mach)

    # 2. Gather Cp, normals, areas, and centers on upper surface panel chains
    cp_flat = csdl.matvec(M_gather_var, cp_uncut)
    cp_paths = csdl.reshape(cp_flat, (2, num_mesh_stations, 20))
    cp_upper = cp_paths[0]  # Shape (num_mesh_stations, 20)

    centers_flat = csdl.matmat(M_gather_var, dynamic_panel_centers)
    centers_paths = csdl.reshape(centers_flat, (2, num_mesh_stations, 20, 3))
    centers_upper = centers_paths[0]  # Shape (num_mesh_stations, 20, 3)

    normals_flat = csdl.matmat(M_gather_var, panel_normals)
    normals_paths = csdl.reshape(normals_flat, (2, num_mesh_stations, 20, 3))
    normals_upper = normals_paths[0]  # Shape (num_mesh_stations, 20, 3)

    areas_flat = csdl.matvec(M_gather_var, panel_areas)
    areas_paths = csdl.reshape(areas_flat, (2, num_mesh_stations, 20))
    areas_upper = areas_paths[0]  # Shape (num_mesh_stations, 20)

    # 3. Evaluate Stratford separation criterion along upper chains
    cp_min_st = -csdl.maximum(-cp_upper, axes=(1,), rho=RHO_SMOOTH_MAX)
    cp_min_2d = csdl.expand(cp_min_st, (num_mesh_stations, 20), 'j->ji')
    cp_rec = sp_floor(cp_upper - cp_min_2d, 0.0, 50.0) / sp_floor(1.0 - cp_min_2d, 0.5, 50.0)

    diff_le = centers_paths[0, :, 0, :] - centers_paths[1, :, 0, :]
    le_dist = csdl.sqrt(csdl.sum(diff_le**2, axes=(1,)) + 1e-12)
    s0 = 0.5 * le_dist

    # Arclengths and Stratford parameter
    S_list = [csdl.Variable(value=np.zeros(num_mesh_stations))]
    ds_list = [s0]
    s_curr = s0

    mu_air_val = 1.4e-5  # Standard air dynamic viscosity Pa*s

    for k in range(19):
        diff = centers_upper[:, k + 1, :] - centers_upper[:, k, :]
        ds_k = csdl.sqrt(csdl.sum(diff**2, axes=(1,)) + 1e-12)
        ds_list.append(ds_k)
        s_mid = s_curr + 0.5 * ds_k
        s_curr = s_curr + ds_k

        grad_k = (cp_rec[:, k + 1] - cp_rec[:, k]) / ds_k
        adverse_k = sp_floor(s_mid * grad_k, 0.0, 50.0)
        cp_rec_mid = 0.5 * (cp_rec[:, k] + cp_rec[:, k + 1])

        Re_k = (rho * v_inf * s_mid) / mu_air_val
        re_factor = (1e-6 * sp_floor(Re_k, 1000.0, 0.01)) ** (-0.1)

        S_k = cp_rec_mid * csdl.sqrt(adverse_k + 1e-12) * re_factor
        S_list.append(S_k)

    # Stack S and ds along chord: shape (20, num_mesh_stations)
    S_stack = csdl.concatenate([csdl.reshape(s, (1, num_mesh_stations)) for s in S_list], axis=0)
    ds_stack = csdl.concatenate([csdl.reshape(ds, (1, num_mesh_stations)) for ds in ds_list], axis=0)
    s_total = csdl.sum(ds_stack, axes=(0,)) + 1e-12  # Shape (num_mesh_stations,)

    # 4. Cumulative downstream monotonic propagation of Stratford parameter
    S_cum_list = [csdl.reshape(S_stack[0], (1, num_mesh_stations))]
    for k in range(1, 20):
        sub_S = csdl.concatenate([csdl.reshape(S_stack[j], (1, num_mesh_stations)) for j in range(k + 1)], axis=0)
        m = csdl.maximum(sub_S, axes=(0,), rho=RHO_SMOOTH_MAX)
        S_cum_list.append(csdl.reshape(m, (1, num_mesh_stations)))

    S_cum = csdl.concatenate(S_cum_list, axis=0)  # Shape (20, num_mesh_stations)

    # Logistic separation activation (sharpness k=40, S_crit=0.39)
    # Applied to cumulative peak Stratford parameter along chord
    act_cum = 1.0 / (1.0 + csdl.exp(-K_LOGISTIC * (S_cum - S_CRIT)))  # Shape (20, num_mesh_stations)
    act_cum_transpose = csdl.transpose(act_cum)                       # Shape (num_mesh_stations, 20)

    # 5. Integrated attached fraction f in [0.01, 1.0]
    sep_length = csdl.sum(act_cum * ds_stack, axes=(0,))  # Shape (num_mesh_stations,)
    f_raw = 1.0 - (sep_length / s_total)
    f_attached = sp_floor(f_raw, F_ATTACHED_FLOOR, 50.0)  # Shape (num_mesh_stations,)
    f_attached.name = 'f_attached'

    # Minimum attached fraction diagnostic
    f_min = -csdl.maximum(-f_attached, axes=(0,), rho=RHO_SMOOTH_MAX)
    f_min.name = 'f_attached_min'

    # 6. Kirchhoff-Helmholtz normal force target & deficit
    # Circulation ratio r_gamma = ((1 + sqrt(f)) / 2)^2
    sqrt_f = csdl.sqrt(f_attached)
    r_gamma = ((1.0 + sqrt_f) / 2.0) ** 2  # Shape (num_mesh_stations,)
    r_gamma.name = 'r_gamma'

    # Normal force target ratio: r_gamma <= 1.0
    # Normal force deficit factor (r_gamma - 1.0) <= 0.0 (identically 0 in attached flow)
    delta_kh_factor = r_gamma - 1.0

    # Upper panel normal force projection (vertical z component of normal)
    # normals_upper has shape (num_mesh_stations, 20, 3)
    nz_upper = normals_upper[:, :, 2]
    # Inviscid normal force on upper chain: N_inviscid = sum(-Cp * nz * Area)
    n_panel_upper = -cp_upper * nz_upper * areas_upper
    N_upper_inviscid = csdl.sum(n_panel_upper, axes=(1,))  # Shape (num_mesh_stations,)

    # Normal force deficit Delta N = N_inviscid * (r_gamma - 1.0)
    # Guaranteed identically 0 when f = 1 (attached flow)
    delta_N = N_upper_inviscid * delta_kh_factor  # Shape (num_mesh_stations,)

    # 7. Normal force deficit redistribution across separated panels
    # Denominator: A_sep_nz = sum(act_cum * nz * Area)
    a_sep_nz = csdl.sum(act_cum_transpose * nz_upper * areas_upper, axes=(1,))
    a_sep_eff = sp_floor(a_sep_nz, EPS_AREA_SEP, 50.0)  # Shape (num_mesh_stations,)

    # Pressure deficit delta_Cp on each panel:
    # delta_Cp = -(act_cum / a_sep_eff) * delta_N
    delta_N_expand = csdl.expand(delta_N, (num_mesh_stations, 20), 'i->ij')
    a_sep_expand = csdl.expand(a_sep_eff, (num_mesh_stations, 20), 'i->ij')
    delta_cp_upper = -(act_cum_transpose / a_sep_expand) * delta_N_expand  # Shape (num_mesh_stations, 20)

    # 8. Corrected panel forces on right wing
    # Force increment delta_F_k = -delta_Cp * Area * Normal
    areas_exp = csdl.expand(areas_upper, (num_mesh_stations, 20, 3), 'ij->ijk')
    delta_cp_exp = csdl.expand(delta_cp_upper, (num_mesh_stations, 20, 3), 'ij->ijk')
    delta_F_upper = -delta_cp_exp * areas_exp * normals_upper  # Shape (num_mesh_stations, 20, 3)

    delta_F_flat = csdl.reshape(delta_F_upper, (num_mesh_stations * 20, 3))
    delta_F_right = csdl.matmat(M_scatter_var, delta_F_flat)  # Shape (num_right_panels, 3)

    # Corrected panel forces on right wing:
    panel_forces_corr_right = panel_forces_inviscid_right + delta_F_right
    panel_forces_corr_right.name = 'panel_forces_corr_right'

    # Corrected lift: full aircraft has factor 2 for symmetry
    delta_L = 2.0 * csdl.sum(delta_F_right[:, 2])
    L_corrected = l_inviscid + delta_L
    L_corrected.name = 'lift_corrected'

    # Corrected pitching moment about CG:
    # r = centers_right - cg_ref
    centers_right = dynamic_panel_centers[:num_right_panels, :]
    cg_exp = csdl.expand(cg_ref, (num_right_panels, 3), 'j->ij')
    r_arm = centers_right - cg_exp
    delta_M_cross = csdl.cross(r_arm, delta_F_right, axis=1)
    delta_My = 2.0 * csdl.sum(delta_M_cross[:, 1])  # Pitching moment about y-axis
    if hasattr(m_inviscid, 'shape') and len(m_inviscid.shape) > 0:
        M_corrected_y = m_inviscid[1] + delta_My
    else:
        M_corrected_y = m_inviscid + delta_My
    M_corrected_y.name = 'pitch_moment_corrected'

    # 9. Corrected Trefftz virtual wake & induced drag
    # Map r_gamma to all 30 TE edges using static symmetry matrix M_wake
    r_gamma_wake = csdl.matvec(M_wake_var, r_gamma)  # Shape (num_TE_edges,)
    r_gamma_wake_exp = csdl.reshape(r_gamma_wake, (1, stall_topology['M_wake'].shape[0]))
    mu_w_stall = r_gamma_wake_exp * mu_w_inviscid
    mu_w_stall.name = 'mu_w_stall'

    # Evaluate 2D Trefftz plane induced drag with decambered wake
    Di_Trefftz_stall, L_wake_stall = trefftz_plane_drag_2d(
        mesh_dict={'TE_edges': te_edges},
        wake_mesh_dict=wake_dict,
        mu=csdl.Variable(value=np.zeros((1, 1))),
        sigma=None,
        mu_w=mu_w_stall,
        rho=rho,
        constant_geometry=True,
        Q_inf=v_inf,
        return_wake_lift=True,
    )

    # Reconcile Trefftz induced drag with surface lift:
    lift_ratio_stall = L_corrected / (L_wake_stall + 1e-6)
    lift_ratio_stall.name = 'lift_ratio_stall'

    Di_Trefftz_reconciled = Di_Trefftz_stall * (lift_ratio_stall ** 2)
    Di_Trefftz_reconciled.name = 'Di_Trefftz_reconciled'

    # Pre-reconciliation residual diagnostic:
    res_wake_lift = csdl.absolute(L_corrected - L_wake_stall) / (csdl.absolute(L_corrected) + 1e-6)
    res_wake_lift.name = 'res_wake_lift'

    # 10. Separated profile drag D_separation
    # Calibrated NACA 0012 constant with local thickness scaling:
    # cd_sep = K_SEP_BASE * (tc / 0.12) * (1 - f)^2
    unattached = 1.0 - f_attached  # Shape (num_mesh_stations,)
    if strip_tc is not None and strip_areas is not None:
        # Scale by local thickness ratio if available
        tc_factor = strip_tc[:num_mesh_stations] / TC_REF
        cd_sep_st = K_SEP_BASE * tc_factor * (unattached ** 2)
        D_separation = 2.0 * q_inf * csdl.sum(cd_sep_st * strip_areas[:num_mesh_stations])
    else:
        # Uniform reference thickness fallback
        cd_sep_st = K_SEP_BASE * (unattached ** 2)
        D_separation = 2.0 * q_inf * csdl.sum(cd_sep_st * (s_total * (station_span_y[1] - station_span_y[0])))
    D_separation.name = 'D_separation'

    # 11. Transonic exposure diagnostic (local Mach > 1.0)
    cp_min_wing = -csdl.maximum(-cp_uncut, axes=(0,), rho=RHO_SMOOTH_MAX)
    transonic_exposure = sp_floor(-cp_min_wing - 1.5, 0.0, 20.0)
    transonic_exposure.name = 'transonic_exposure'

    return {
        'panel_forces_corr_right': panel_forces_corr_right,
        'panel_forces_corrected_right': panel_forces_corr_right,
        'lift_corrected': L_corrected,
        'pitch_moment_corrected': M_corrected_y,
        'f_attached': f_attached,
        'f_attached_min': f_min,
        'r_gamma': r_gamma,
        'mu_w_stall': mu_w_stall,
        'Di_Trefftz_reconciled': Di_Trefftz_reconciled,
        'lift_ratio_stall': lift_ratio_stall,
        'k_reconcile': lift_ratio_stall,
        'res_wake_lift': res_wake_lift,
        'D_separation': D_separation,
        'transonic_exposure': transonic_exposure,
        'cp_uncut': cp_uncut,
    }
