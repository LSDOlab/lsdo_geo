"""Differentiable Direct-IBL Viscous Drag post-processing analysis for BWB wings.

Implements Head's turbulent entrainment method with Ludwieg-Tillmann skin-friction
closure and Squire-Young momentum-deficit relation, fully vectorized in CSDL.
"""

from __future__ import annotations
import numpy as np
import csdl_alpha as csdl
from typing import Dict, Any, Tuple


# Numerical guard constants
EPS_U2 = 0.04           # Velocity-squared floor for smooth_positive(1 - Cp)
BETA_U2 = 50.0          # Softplus sharpness for velocity-squared floor
EPS_THETA = 1.0e-6      # Momentum thickness floor [m]
BETA_THETA = 200000.0    # Sharpness for momentum thickness floor
EPS_H1_DIFF3 = 1.0e-4   # Floor for (H1 - 3.0)
BETA_H1 = 1000.0        # Sharpness for H1 differences
EPS_RE_THETA = 1.0      # Reynolds number Re_theta floor
BETA_RE_THETA = 10.0    # Sharpness for Re_theta floor
STEP_LIMIT_RATIO = 0.95 # Dimensionless convective velocity step limit for alpha_step (preserves physical gradients with algebraic limiter)

# Head correlation inverse constants for C-infinity smooth linear continuation
H_SEP = 2.4             # Turbulent separation threshold
P_HEAD = 1.0 / 1.2721   # Exponent 0.786086
Y_SEP = 0.8702 / ((H_SEP - 1.1) ** 1.2721)  # ~0.62326 (H1 - 3.0445 at H=2.4)
S0_HEAD = -P_HEAD * (H_SEP - 1.1) / Y_SEP   # ~ -1.63955 (dH/dy at separation onset)
BETA_CONTINUATION = 20.0  # Smooth transition sharpness for linear continuation


def sp_floor(x, floor: float, beta: float):
    """Normalized smooth softplus floor: smooth_floor(x, floor) >= floor."""
    u = x - floor
    return floor + (1.0 / beta) * csdl.softplus(beta * u)


def evaluate_smooth_H(H1_diff: csdl.Variable) -> csdl.Variable:
    """C-infinity smooth linear continuation of Head's shape factor H(H1).
    
    For attached flow (H1 - 3.0445 >= Y_SEP, i.e. H <= 2.4), this reproduces the
    exact physical Head correlation: H = 1.1 + (0.8702 / (H1 - 3.0445))^(1/1.2721).
    For separated flow (H1 - 3.0445 < Y_SEP), it continues along the tangent line
    with constant restoring slope dH/dH1 = -1.63955, guaranteeing non-vanishing
    gradients and bounded constraint scaling (H ~ 2.4 to 4.5).
    """
    y_smooth = Y_SEP + (1.0 / BETA_CONTINUATION) * csdl.softplus(BETA_CONTINUATION * (H1_diff - Y_SEP))
    f_smooth = 1.1 + (0.8702 / y_smooth) ** P_HEAD
    return f_smooth + S0_HEAD * (H1_diff - y_smooth)


def build_ibl_mesh_topology(
    points: np.ndarray,
    cells_dict: Dict[str, np.ndarray],
    scale_factor: float = 1.0,
    num_drag_strips: int = 100,
    y_drag_centers: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Construct immutable static topology index maps and interpolation matrices.
    
    Parameters
    ----------
    points : np.ndarray
        Mesh points (unscaled or scaled).
    cells_dict : dict
        Mesh cells dictionary containing 'quad' connectivity.
    scale_factor : float
        Wing scaling factor applied to points (default 1.0).
    num_drag_strips : int
        Number of spanwise drag strips (default 100).
    y_drag_centers : np.ndarray, optional
        Precomputed spanwise drag strip center coordinates. If None, derived from
        the full span of station_span_y.
        
    Returns
    -------
    topology : dict
        Dictionary containing index matrices, gather matrices, and interpolation map.
    """
    pts = points * scale_factor
    quad_cells = cells_dict['quad']
    total_quads = len(quad_cells)
    centers = pts[quad_cells].mean(axis=1)

    # Restrict to right half-wing (y > 0)
    right_indices = np.where(centers[:, 1] > 1e-4)[0]
    if len(right_indices) == 0:
        raise ValueError("No panels found on right half-wing (y > 0).")

    y_right = centers[right_indices, 1]
    y_round = np.round(y_right, 3)
    unique_y, counts = np.unique(y_round, return_counts=True)
    
    # 40-panel spanwise stations
    station_ys = unique_y[counts == 40]
    num_stations = len(station_ys)
    if num_stations < 2:
        raise ValueError(
            f"Expected at least 2 valid span stations with 40 quads, found {num_stations}."
        )

    upper_indices = np.zeros((num_stations, 20), dtype=int)
    lower_indices = np.zeros((num_stations, 20), dtype=int)
    station_span_y = np.zeros(num_stations)

    for i, y_val in enumerate(station_ys):
        st_panels = right_indices[y_round == y_val]
        if len(st_panels) != 40:
            raise ValueError(
                f"Station at y={y_val} has {len(st_panels)} panels instead of 40."
            )
        c_st = centers[st_panels]
        station_span_y[i] = np.mean(c_st[:, 1])

        upper_mask = c_st[:, 2] >= 0.0
        lower_mask = c_st[:, 2] < 0.0

        u_idx = st_panels[upper_mask]
        l_idx = st_panels[lower_mask]

        if len(u_idx) != 20 or len(l_idx) != 20:
            raise ValueError(
                f"Station at y={y_val} has {len(u_idx)} upper and {len(l_idx)} lower panels; expected 20 each."
            )

        # Sort from LE to TE (increasing x coordinate)
        u_sorted = u_idx[np.argsort(centers[u_idx, 0])]
        l_sorted = l_idx[np.argsort(centers[l_idx, 0])]

        upper_indices[i] = u_sorted
        lower_indices[i] = l_sorted

    # Construct static gather matrix M_gather of shape (2 * num_stations * 20, total_quads)
    # paths_idx shape is (2, num_stations, 20) where path 0 = upper, path 1 = lower
    paths_idx = np.stack([upper_indices, lower_indices], axis=0)
    flat_paths_idx = paths_idx.flatten()
    n_gathered = len(flat_paths_idx)

    M_gather = np.zeros((n_gathered, total_quads))
    for row_i, panel_idx in enumerate(flat_paths_idx):
        M_gather[row_i, panel_idx] = 1.0

    # Build interpolation matrix from IBL stations to refined drag strips
    if y_drag_centers is None:
        b_tip_ref = float(np.max(station_span_y))
        num_drag_nodes = num_drag_strips + 1
        eta_drag = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, num_drag_nodes))
        y_drag_span = eta_drag * b_tip_ref
        y_drag_centers = 0.5 * (y_drag_span[:-1] + y_drag_span[1:])
    else:
        y_drag_centers = np.asarray(y_drag_centers, dtype=float)

    M_interp_ibl = np.zeros((num_drag_strips, num_stations))
    for i, yd in enumerate(y_drag_centers):
        if yd <= station_span_y[0]:
            M_interp_ibl[i, 0] = 1.0
        elif yd >= station_span_y[-1]:
            M_interp_ibl[i, -1] = 1.0
        else:
            k = np.searchsorted(station_span_y, yd) - 1
            t = (yd - station_span_y[k]) / (station_span_y[k+1] - station_span_y[k])
            M_interp_ibl[i, k] = 1.0 - t
            M_interp_ibl[i, k+1] = t

    return {
        'num_stations': num_stations,
        'station_span_y': station_span_y,
        'upper_indices': upper_indices,
        'lower_indices': lower_indices,
        'paths_idx': paths_idx,
        'M_gather': M_gather,
        'M_interp_ibl': M_interp_ibl,
    }


def evaluate_bwb_viscous_ibl(
    cp_node0: csdl.Variable,
    dynamic_panel_centers: csdl.Variable,
    topology: Dict[str, Any],
    v_cruise: csdl.Variable,
    rho_cruise: csdl.Variable,
    mu_air: csdl.Variable,
    q_inf: csdl.Variable,
    strip_area: csdl.Variable,
    total_strip_area: csdl.Variable,
) -> Dict[str, csdl.Variable]:
    """Evaluate CSDL-native direct Head integral boundary layer and profile drag.
    
    Parameters
    ----------
    cp_node0 : csdl.Variable
        VortexAD pressure coefficient array at cruise (node 0), shape (total_quads,).
    dynamic_panel_centers : csdl.Variable
        Deformed panel center coordinates, shape (total_quads, 3).
    topology : dict
        Static topology outputs from build_ibl_mesh_topology.
    v_cruise : csdl.Variable
        Freestream cruise speed [m/s].
    rho_cruise : csdl.Variable
        Cruise air density [kg/m^3].
    mu_air : csdl.Variable
        Dynamic air viscosity [Pa*s].
    q_inf : csdl.Variable
        Freestream dynamic pressure [Pa].
    strip_area : csdl.Variable
        Strip areas for the 100 drag strips [m^2].
    total_strip_area : csdl.Variable
        Total reference planform area of drag strips [m^2].
        
    Returns
    -------
    dict
        Dictionary of CSDL variables: CD_viscous, D_viscous, cd_viscous_elem,
        cd_section, H_max_ibl, ibl_attachment_margin, ibl_min_cp, ibl_cp_cutoff_margin,
        theta_te_upper, theta_te_lower.
    """
    num_stations = topology['num_stations']
    M_gather_var = csdl.Variable(value=topology['M_gather'])
    M_interp_var = csdl.Variable(value=topology['M_interp_ibl'])

    # Kinematic viscosity nu = mu / rho
    nu = mu_air / rho_cruise

    # Gather Cp and panel centers
    cp_flat = csdl.matvec(M_gather_var, cp_node0)
    cp_paths = csdl.reshape(cp_flat, (2, num_stations, 20))

    centers_flat = csdl.matmat(M_gather_var, dynamic_panel_centers)
    centers_paths = csdl.reshape(centers_flat, (2, num_stations, 20, 3))

    # Pressure cutoff diagnostics: min(Cp) and margin from -5.0
    ibl_min_cp = -csdl.maximum(-cp_node0, axes=(0,), rho=50.0)
    ibl_min_cp.name = 'ibl_min_cp'
    ibl_cp_cutoff_margin = ibl_min_cp - (-5.0)
    ibl_cp_cutoff_margin.name = 'ibl_cp_cutoff_margin'

    # External velocity Ue = V_inf * sqrt(smooth_positive(1 - Cp))
    u_sq = sp_floor(1.0 - cp_paths, EPS_U2, BETA_U2)
    Ue = v_cruise * csdl.sqrt(u_sq)

    # Arclength s_0 from leading edge to first center:
    # Measure physical distance across upper and lower first panel centers at the leading edge nose
    diff_le = centers_paths[0, :, 0, :] - centers_paths[1, :, 0, :]
    le_dist = csdl.sqrt(csdl.sum(diff_le**2, axes=(1,)) + 1e-12)
    s0_station = 0.5 * le_dist
    s0 = csdl.expand(s0_station, (2, num_stations), 'j->ij')

    # Initialize turbulent attached state at first center
    Ue0 = Ue[:, :, 0]
    Re_s0 = (Ue0 * s0 / nu) + 1.0
    theta_0 = 0.037 * s0 * (Re_s0 ** -0.2)
    H_0_val = 1.4
    H1_0_val = 3.0445 + 0.8702 / ((H_0_val - 1.1) ** 1.2721)
    delta1_0 = theta_0 * H1_0_val

    theta = theta_0
    delta1 = delta1_0

    # Collect chordwise upper and lower surface H values across all steps to compute localized section max
    H_upper_steps = [csdl.Variable(value=np.full((num_stations,), H_0_val))]
    H_lower_steps = [csdl.Variable(value=np.full((num_stations,), H_0_val))]
    all_H = [csdl.reshape(csdl.Variable(value=np.full((2, num_stations), H_0_val)), (-1,))]

    # March 19 chordwise panel intervals with second-order Runge-Kutta
    for k in range(19):
        diff = centers_paths[:, :, k+1, :] - centers_paths[:, :, k, :]
        ds_k = csdl.sqrt(csdl.sum(diff**2, axes=(2,)) + 1e-12)

        Ue_k = Ue[:, :, k]
        Ue_next = Ue[:, :, k+1]

        dlogU = (Ue_next - Ue_k) / (0.5 * (Ue_k + Ue_next) * ds_k)
        u_lim = dlogU * ds_k / STEP_LIMIT_RATIO
        alpha_step = (u_lim / csdl.sqrt(1.0 + u_lim**2)) * (STEP_LIMIT_RATIO / ds_k)

        # RK2 Stage 1
        th_fl = sp_floor(theta, EPS_THETA, BETA_THETA)
        H1 = delta1 / th_fl
        H = evaluate_smooth_H(H1 - 3.0445)
        all_H.append(csdl.reshape(H, (-1,)))
        H_upper_steps.append(H[0, :])
        H_lower_steps.append(H[1, :])

        Re_th = sp_floor(Ue_k * th_fl / nu, EPS_RE_THETA, BETA_RE_THETA)
        Cf = 0.246 * (10.0 ** (-0.678 * H)) * (Re_th ** -0.268)

        dtheta_1 = 0.5 * Cf - (H + 2.0) * th_fl * alpha_step
        H1_diff3 = sp_floor(H1 - 3.0, EPS_H1_DIFF3, BETA_H1)
        ddelta1_1 = 0.0306 * (H1_diff3 ** -0.6169) - delta1 * alpha_step

        th_mid = th_fl + 0.5 * ds_k * dtheta_1
        d1_mid = delta1 + 0.5 * ds_k * ddelta1_1
        th_mid_fl = sp_floor(th_mid, EPS_THETA, BETA_THETA)

        # RK2 Stage 2
        H1_mid = d1_mid / th_mid_fl
        H_mid = evaluate_smooth_H(H1_mid - 3.0445)

        Ue_mid = 0.5 * (Ue_k + Ue_next)
        Re_th_mid = sp_floor(Ue_mid * th_mid_fl / nu, EPS_RE_THETA, BETA_RE_THETA)
        Cf_mid = 0.246 * (10.0 ** (-0.678 * H_mid)) * (Re_th_mid ** -0.268)

        dtheta_2 = 0.5 * Cf_mid - (H_mid + 2.0) * th_mid_fl * alpha_step
        H1_mid_diff3 = sp_floor(H1_mid - 3.0, EPS_H1_DIFF3, BETA_H1)
        ddelta1_2 = 0.0306 * (H1_mid_diff3 ** -0.6169) - d1_mid * alpha_step

        theta = sp_floor(th_fl + ds_k * dtheta_2, EPS_THETA, BETA_THETA)
        delta1 = sp_floor(delta1 + ds_k * ddelta1_2, EPS_THETA, BETA_THETA)

    # Trailing edge states
    th_final_fl = sp_floor(theta, EPS_THETA, BETA_THETA)
    H1_final = delta1 / th_final_fl
    H_te = evaluate_smooth_H(H1_final - 3.0445)

    theta_te_upper = th_final_fl[0, :]
    theta_te_lower = th_final_fl[1, :]
    theta_te_upper.name = 'theta_te_upper'
    theta_te_lower.name = 'theta_te_lower'

    H_te_upper = H_te[0, :]
    H_te_lower = H_te[1, :]
    Ue_te_upper = Ue[0, :, -1]
    Ue_te_lower = Ue[1, :, -1]

    # Local chord c = x_te - x_le:
    # Streamwise distance between first and last panel centers plus leading and trailing edge offsets
    dx_centers = centers_paths[0, :, -1, 0] - centers_paths[0, :, 0, 0]
    diff_last = centers_paths[0, :, -1, :] - centers_paths[0, :, -2, :]
    ds_last = csdl.sqrt(csdl.sum(diff_last**2, axes=(1,)) + 1e-12)
    c_local = dx_centers + s0_station + 0.5 * ds_last

    # Squire-Young momentum-deficit relation
    cd_section = (2.0 / c_local) * (
        theta_te_upper * ((Ue_te_upper / v_cruise) ** ((H_te_upper + 5.0) / 2.0)) +
        theta_te_lower * ((Ue_te_lower / v_cruise) ** ((H_te_lower + 5.0) / 2.0))
    )
    cd_section.name = 'CD_ibl_section'

    # Interpolate drag to the 100 drag strips
    cd_viscous_elem = csdl.matvec(M_interp_var, cd_section)

    # Local chordwise smooth-max per IBL station over upper and lower surfaces
    # Reshaping to (20, num_stations) allows chordwise smooth-max along axis 0
    H_upper_matrix = csdl.reshape(csdl.concatenate(H_upper_steps), (20, num_stations))
    H_upper_matrix.name = 'H_upper_matrix'
    H_lower_matrix = csdl.reshape(csdl.concatenate(H_lower_steps), (20, num_stations))
    H_lower_matrix.name = 'H_lower_matrix'
    H_section_upper = csdl.maximum(H_upper_matrix, axes=(0,), rho=20.0)
    H_section_upper.name = 'H_section_upper'
    H_section_lower = csdl.maximum(H_lower_matrix, axes=(0,), rho=20.0)
    H_section_lower.name = 'H_section_lower'

    # Section maximum across both surfaces (retained for diagnostics)
    H_both_matrix = csdl.reshape(csdl.concatenate([H_section_upper, H_section_lower]), (2, num_stations))
    H_section_both = csdl.maximum(H_both_matrix, axes=(0,), rho=20.0)
    H_section_both.name = 'H_section_both'

    # Interpolate upper surface section maximum H onto the 100 drag strips.
    # On lifting wings, physical boundary layer separation occurs on the upper (suction) surface.
    # Lower surfaces have forward flow from the stagnation point into the nose, which makes a
    # LE-to-TE marching path exhibit an artificial adverse gradient at positive lift.
    H_strip = csdl.matvec(M_interp_var, H_section_upper)
    H_strip.name = 'H_strip'

    # Global diagnostic scalar attachment margin (upper surface where physical separation occurs)
    H_max_upper_vec = csdl.concatenate(H_upper_steps)
    H_max_ibl = csdl.maximum(H_max_upper_vec, axes=(0,), rho=20.0)
    H_max_ibl.name = 'H_max_ibl'
    ibl_attachment_margin = 2.4 - H_max_ibl
    ibl_attachment_margin.name = 'ibl_attachment_margin'

    # Viscous drag force and coefficient
    D_viscous = q_inf * csdl.sum(cd_viscous_elem * strip_area)
    D_viscous.name = 'D_viscous'
    CD_viscous = csdl.sum(cd_viscous_elem * strip_area) / total_strip_area
    CD_viscous.name = 'CD_viscous'

    return {
        'CD_viscous': CD_viscous,
        'D_viscous': D_viscous,
        'cd_viscous_elem': cd_viscous_elem,
        'CD_ibl_section': cd_section,
        'H_max_ibl': H_max_ibl,
        'ibl_attachment_margin': ibl_attachment_margin,
        'ibl_min_cp': ibl_min_cp,
        'ibl_cp_cutoff_margin': ibl_cp_cutoff_margin,
        'theta_te_upper': theta_te_upper,
        'theta_te_lower': theta_te_lower,
        'H_upper_matrix': H_upper_matrix,
        'H_section_upper': H_section_upper,
        'H_section_lower': H_section_lower,
        'H_section_both': H_section_both,
        'H_strip': H_strip,
    }
