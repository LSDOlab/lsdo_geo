"""Transonic Korn-Lock wave-drag model for BWB strip theory.

This module provides a differentiable, geometry- and load-aware transonic
wave-drag correction evaluated along 100 drag strips.

Physical model:
    c_i = norm(TE_i - LE_i)
    (t/c)_i = max_x(||r_upper(x_i) - r_lower(x_i)||) / c_i
    r_qc = 0.75 * LE + 0.25 * TE
    Lambda_qc derived from adjacent spanwise quarter-chord points using atan2
    Lprime_i = L_strip_i / dy_aero_dynamic_i
    cl_i = Lprime_i / (q_node * c_i)
    abs_cl_i = sqrt(cl_i**2 + eps_cl**2)

Korn-Lock correlation:
    kappa_A = 0.87 (frozen conventional-airfoil technology factor)
    delta_M_dd_to_crit = (0.1 / 80.0)**(1/3)  # approx 0.1077217345015942
    cos_Lambda = cos(Lambda_qc)
    M_dd_i = kappa_A / cos_Lambda - (t/c)_i / cos_Lambda**2 - abs_cl_i / (10.0 * cos_Lambda**3)
    M_crit_i = M_dd_i - delta_M_dd_to_crit
    delta_M_i = M_node - M_crit_i
    delta_M_eff_i = softplus(100.0 * delta_M_i) / 100.0
    cd_wave_i = 20.0 * delta_M_eff_i**4
    CD_wave_node = sum(cd_wave_i * strip_area_i) / total_strip_area
    D_wave_node = CD_wave_node * q_node * planform_area
"""

from __future__ import annotations

from typing import Union, Tuple
import numpy as np
import csdl_alpha as csdl

# Fixed physical / correlation constants
KAPPA_A = 0.87                         # Conventional-airfoil technology factor
LOCK_COEFFICIENT = 20.0                # Lock 4th-power scaling coefficient
LOCK_DRAG_DIVERGENCE_SLOPE = 0.1       # dCD/dM at drag divergence
LOCK_ACTIVATION_SHARPNESS = 100.0      # Scaled softplus sharpness parameter
DELTA_M_DD_TO_CRIT = (0.1 / 80.0)**(1.0 / 3.0)  # 0.1077217345015942
EPS_CL = 1.0e-5                        # Floor for differentiable absolute value of cl
EPS_GEOM = 1.0e-8                      # Numerical floor for geometry denominators
EPS_GAP = 1.0e-12                      # Numerical floor for differentiable Euclidean gap distance
RHO_MAX_TC = 50.0                      # CSDL smooth maximum aggregation sharpness


def setup_strip_projection_points(
    y_strip_centers: np.ndarray,
    scale_factor: float,
    num_chord_fractions: int = 41,
    z_offset_ratio: float = 0.05,
):
    """Generate static seed points for paired upper and lower surface projections.

    Parameters
    ----------
    y_strip_centers : np.ndarray
        Initial spanwise y-coordinates for each drag-strip center (shape: num_drag_strips,).
    scale_factor : float
        Overall geometric scaling factor.
    num_chord_fractions : int
        Number of chordwise stations per strip (default 41).
    z_offset_ratio : float
        Seed point vertical offset as fraction of reference chord.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[int, int]]
        upper_seed_pts: shape (num_chord_fractions * num_strips, 3)
        lower_seed_pts: shape (num_chord_fractions * num_strips, 3)
        grid_shape: (num_chord_fractions, num_strips)
    """
    num_strips = len(y_strip_centers)
    x_fracs = np.linspace(0.0, 1.0, num_chord_fractions)
    
    # 2D grid: (num_chord_fractions, num_strips)
    x_grid, y_grid = np.meshgrid(x_fracs * scale_factor, y_strip_centers, indexing='ij')
    grid_shape = (num_chord_fractions, num_strips)

    upper_seed_pts = np.column_stack([
        x_grid.ravel(),
        y_grid.ravel(),
        np.full(x_grid.size, z_offset_ratio * scale_factor),
    ])
    lower_seed_pts = np.column_stack([
        x_grid.ravel(),
        y_grid.ravel(),
        np.full(x_grid.size, -z_offset_ratio * scale_factor),
    ])
    return upper_seed_pts, lower_seed_pts, grid_shape


def compute_strip_sweep(r_pts: csdl.Variable):
    """Compute local sweep angle along drag strips from a spanwise point chain.

    Parameters
    ----------
    r_pts : csdl.Variable
        Coordinates of strip points for each drag strip (shape: (num_strips, 3)).
        For quarter-chord sweep, r_pts = 0.75 * LE + 0.25 * TE.

    Returns
    -------
    csdl.Variable
        Sweep angle in radians for each drag strip (shape: (num_strips,)).
    """
    # dx, dy along the span (ordered by positive y)
    # Interior: centered differences r[i+1] - r[i-1]
    dx_int = r_pts[2:, 0] - r_pts[:-2, 0]
    dy_int = r_pts[2:, 1] - r_pts[:-2, 1]

    # Root boundary: forward difference r[1] - r[0]
    dx_0 = csdl.reshape(r_pts[1, 0] - r_pts[0, 0], (1,))
    dy_0 = csdl.reshape(r_pts[1, 1] - r_pts[0, 1], (1,))

    # Tip boundary: backward difference r[-1] - r[-2]
    dx_tip = csdl.reshape(r_pts[-1, 0] - r_pts[-2, 0], (1,))
    dy_tip = csdl.reshape(r_pts[-1, 1] - r_pts[-2, 1], (1,))

    dx = csdl.concatenate([dx_0, dx_int, dx_tip])
    dy = csdl.concatenate([dy_0, dy_int, dy_tip])

    # atan2(dx, dy) where x is chordwise (streamwise) and y is spanwise
    sweep = csdl.arctan2(dx, dy)
    return sweep


def compute_quarter_chord_sweep(r_qc: csdl.Variable):
    """Compute local quarter-chord sweep angle Lambda_qc along drag strips.

    Parameters
    ----------
    r_qc : csdl.Variable
        Coordinates of quarter-chord points for each drag strip (shape: (num_strips, 3)).

    Returns
    -------
    csdl.Variable
        Quarter-chord sweep angle in radians for each drag strip (shape: (num_strips,)).
    """
    return compute_strip_sweep(r_qc)


def compute_half_chord_sweep(r_midchord: csdl.Variable):
    """Compute local half-chord sweep angle Lambda_0.5 along drag strips.

    Parameters
    ----------
    r_midchord : csdl.Variable
        Coordinates of mid-chord points for each drag strip (shape: (num_strips, 3)).

    Returns
    -------
    csdl.Variable
        Half-chord sweep angle in radians for each drag strip (shape: (num_strips,)).
    """
    return compute_strip_sweep(r_midchord)


def compute_strip_thickness_to_chord(
    upper_surface_pts: csdl.Variable,
    lower_surface_pts: csdl.Variable,
    grid_shape: tuple[int, int],
    chord_lengths: csdl.Variable,
    rho: float = RHO_MAX_TC,
):
    """Compute thickness-to-chord ratio (t/c)_i for each drag strip using live surface points.

    Parameters
    ----------
    upper_surface_pts : csdl.Variable
        Evaluated upper surface projection points (shape: (N_pts, 3)).
    lower_surface_pts : csdl.Variable
        Evaluated lower surface projection points (shape: (N_pts, 3)).
    grid_shape : tuple[int, int]
        (num_chord_fractions, num_strips).
    chord_lengths : csdl.Variable
        Chord length c_i for each strip (shape: (num_strips,)).
    rho : float
        Softmax / smooth maximum sharpness parameter.

    Returns
    -------
    csdl.Variable
        Thickness-to-chord ratio (t/c)_i for each drag strip (shape: (num_strips,)).
    """
    nx, ny = grid_shape
    upper_grid = csdl.reshape(upper_surface_pts, (nx, ny, 3))
    lower_grid = csdl.reshape(lower_surface_pts, (nx, ny, 3))

    gap_vectors = upper_grid - lower_grid
    # Differentiable Euclidean distance with numerical floor to avoid 0/0 gradient at sharp LE/TE
    gap_distances = csdl.sqrt(csdl.sum(gap_vectors**2, axes=(2,)) + EPS_GAP)  # shape: (nx, ny)

    # Smooth maximum gap across all chordwise fractions for each strip
    max_gap = csdl.maximum(gap_distances, axes=(0,), rho=rho)  # shape: (ny,)
    tc_ratio = max_gap / (chord_lengths + EPS_GEOM)
    return tc_ratio


def evaluate_wave_drag(
    strip_tc: csdl.Variable,
    strip_sweep: csdl.Variable = None,
    strip_cl: csdl.Variable = None,
    strip_area: csdl.Variable = None,
    total_strip_area: csdl.Variable = None,
    mach_node: Union[float, csdl.Variable] = None,
    q_node: Union[float, csdl.Variable] = None,
    planform_area: csdl.Variable = None,
    strip_sweep_halfchord: csdl.Variable = None,
    strip_sweep_qc: csdl.Variable = None,
):
    """Evaluate Korn-Lock transonic wave drag on strip level and integrated for a flight node.

    Parameters
    ----------
    strip_tc : csdl.Variable
        Thickness-to-chord ratio (t/c)_i for each strip (shape: (num_strips,)).
    strip_sweep : csdl.Variable, optional
        Quarter-chord (or local) sweep angle Lambda in radians for each strip (shape: (num_strips,)).
    strip_cl : csdl.Variable
        Sectional lift coefficient cl_i for each strip (shape: (num_strips,)).
    strip_area : csdl.Variable
        Full-wing strip planform area for each strip (shape: (num_strips,)).
    total_strip_area : csdl.Variable
        Total planform area summed across strips (scalar).
    mach_node : float or csdl.Variable
        Flight condition Mach number for this node.
    q_node : float or csdl.Variable
        Dynamic pressure for this node [Pa].
    planform_area : csdl.Variable
        Aircraft reference planform area [m^2].
    strip_sweep_halfchord : csdl.Variable, optional
        Legacy alias for strip_sweep (half-chord sweep).
    strip_sweep_qc : csdl.Variable, optional
        Explicit quarter-chord sweep alias for strip_sweep.

    Returns
    -------
    dict
        Dictionary containing CSDL variables:
        - 'CD_wave': integrated wave drag coefficient
        - 'D_wave': integrated wave drag force [N]
        - 'cd_wave_strip': strip wave drag coefficients (shape: (num_strips,))
        - 'M_dd': drag divergence Mach numbers (shape: (num_strips,))
        - 'M_crit': critical Mach numbers (shape: (num_strips,))
        - 'delta_M': M - M_crit (shape: (num_strips,))
        - 'delta_M_eff': scaled softplus of delta_M (shape: (num_strips,))
        - 'abs_cl': differentiable |cl| (shape: (num_strips,))
    """
    if strip_sweep is None:
        if strip_sweep_qc is not None:
            strip_sweep = strip_sweep_qc
        elif strip_sweep_halfchord is not None:
            strip_sweep = strip_sweep_halfchord
        else:
            raise ValueError("Must provide strip_sweep (or strip_sweep_qc / strip_sweep_halfchord) to evaluate_wave_drag.")

    # Differentiable |cl|
    abs_cl = csdl.sqrt(strip_cl**2 + EPS_CL**2)

    cos_lambda = csdl.cos(strip_sweep)
    cos2 = cos_lambda**2
    cos3 = cos_lambda**3

    m_dd = (KAPPA_A / cos_lambda) - (strip_tc / cos2) - (abs_cl / (10.0 * cos3))
    m_crit = m_dd - DELTA_M_DD_TO_CRIT
    
    delta_m = mach_node - m_crit
    delta_m_eff = csdl.softplus(LOCK_ACTIVATION_SHARPNESS * delta_m) / LOCK_ACTIVATION_SHARPNESS
    cd_wave_strip = LOCK_COEFFICIENT * (delta_m_eff**4)

    cd_wave_node = csdl.sum(cd_wave_strip * strip_area) / total_strip_area
    d_wave_node = cd_wave_node * q_node * planform_area

    return {
        'CD_wave': cd_wave_node,
        'D_wave': d_wave_node,
        'cd_wave_strip': cd_wave_strip,
        'M_dd': m_dd,
        'M_crit': m_crit,
        'delta_M': delta_m,
        'delta_M_eff': delta_m_eff,
        'abs_cl': abs_cl,
    }


def build_panel_strip_wave_mapping(
    W_strip: np.ndarray,
    M_interp_drag: np.ndarray,
    y_strip_centers: np.ndarray,
    y_drag_centers: np.ndarray,
) -> dict:
    """Build static interpolation and panel-slicing mappings for Karman-Tsien RH wave drag.

    Parameters
    ----------
    W_strip : np.ndarray
        Selection matrix mapping right-half panels to aero strips (shape: (num_aero_strips, num_right_panels)).
    M_interp_drag : np.ndarray
        Interpolation matrix from aero strips to drag strips (shape: (num_drag_strips, num_aero_strips)).
    y_strip_centers : np.ndarray
        Spanwise coordinates of aero strip centers (shape: (num_aero_strips,)).
    y_drag_centers : np.ndarray
        Spanwise coordinates of drag strip centers (shape: (num_drag_strips,)).

    Returns
    -------
    dict
        Dictionary containing:
        - 'M_drag_to_aero': shape (num_aero_strips, num_drag_strips)
        - 'M_drag_to_panel': shape (num_right_panels, num_drag_strips)
        - 'strip_panel_slices': list of (start, end) index tuples per aero strip
    """
    num_aero = len(y_strip_centers)
    num_drag = len(y_drag_centers)
    num_panels = W_strip.shape[1]

    # Linear interpolation mapping from drag strip centers to aero strip centers
    M_drag_to_aero = np.zeros((num_aero, num_drag))
    for i in range(num_aero):
        y_val = y_strip_centers[i]
        if y_val <= y_drag_centers[0]:
            M_drag_to_aero[i, 0] = 1.0
        elif y_val >= y_drag_centers[-1]:
            M_drag_to_aero[i, -1] = 1.0
        else:
            k = np.searchsorted(y_drag_centers, y_val) - 1
            k = max(0, min(k, num_drag - 2))
            dy = y_drag_centers[k + 1] - y_drag_centers[k]
            t = (y_val - y_drag_centers[k]) / (dy + 1e-15)
            M_drag_to_aero[i, k] = 1.0 - t
            M_drag_to_aero[i, k + 1] = t

    # Map drag strip quantities to individual panels via aero strip assignment
    M_drag_to_panel = W_strip.T @ M_drag_to_aero  # shape: (num_right_panels, num_drag_strips)

    # Contiguous panel slices per aero strip
    strip_panel_slices = []
    for s in range(num_aero):
        idx = np.where(W_strip[s] > 0.0)[0]
        if len(idx) > 0:
            strip_panel_slices.append((int(idx[0]), int(idx[-1] + 1)))
        else:
            strip_panel_slices.append((0, 0))

    return {
        'M_drag_to_aero': M_drag_to_aero,
        'M_drag_to_panel': M_drag_to_panel,
        'strip_panel_slices': strip_panel_slices,
    }


def evaluate_wave_drag_karman_tsien_rh(
    *,
    cp_right: csdl.Variable,
    panel_area_right: csdl.Variable,
    strip_sweep_halfchord: csdl.Variable,
    strip_area: csdl.Variable,
    total_strip_area: csdl.Variable,
    mach_node: Union[float, csdl.Variable],
    q_node: Union[float, csdl.Variable],
    planform_area: csdl.Variable,
    wave_panel_mapping: dict,
    M_interp_drag: np.ndarray,
    W_strip: np.ndarray,
    dy_aero_dyn: csdl.Variable,
    local_chord_drag: csdl.Variable,
    c_cal: float = 1.0,
    beta_s: float = 20.0,
    m_n_buffet_limit: float = 1.30,
    rho_strip_max: float = 20.0,
    d_min: float = 0.15,
    beta_d: float = 50.0,
    p_min: float = 0.05,
    beta_p: float = 50.0,
    gamma: float = 1.4,
    delta_s_max: float = 1.0,
) -> dict:
    """Evaluate Karman-Tsien + Surface-Integrated Rankine-Hugoniot wave drag.

    Parameters
    ----------
    cp_right : csdl.Variable
        Surface pressure coefficients on right-half panels (shape: (num_right_panels,)).
    panel_area_right : csdl.Variable
        Area of right-half panels [m^2] (shape: (num_right_panels,)).
    strip_sweep_halfchord : csdl.Variable
        Local half-chord sweep angle in radians for each drag strip (shape: (num_drag_strips,)).
    strip_area : csdl.Variable
        Full-wing planform area of drag strips (shape: (num_drag_strips,)).
    total_strip_area : csdl.Variable
        Total reference planform area of drag strips (scalar).
    mach_node : float or csdl.Variable
        Flight condition freestream Mach number.
    q_node : float or csdl.Variable
        Flight condition dynamic pressure [Pa].
    planform_area : csdl.Variable
        Aircraft reference planform area [m^2].
    wave_panel_mapping : dict
        Mapping dictionary from build_panel_strip_wave_mapping.
    M_interp_drag : np.ndarray
        Interpolation matrix from aero strips to drag strips (shape: (num_drag_strips, num_aero_strips)).
    W_strip : np.ndarray
        Selection matrix mapping right-half panels to aero strips (shape: (num_aero_strips, num_right_panels)).
    dy_aero_dyn : csdl.Variable
        Dynamic spanwise spacing of aero strips.
    local_chord_drag : csdl.Variable
        Chord length at each drag strip [m] (shape: (num_drag_strips,)).
    c_cal : float
        Wave drag calibration scaling factor (default: 1.0).
    beta_s : float
        Softplus sharpness parameter for supersonic activation (default: 20.0).
    m_n_buffet_limit : float
        Buffet-onset normal Mach number threshold (default: 1.30).
    rho_strip_max : float
        CSDL smooth maximum sharpness for per-strip peak Mach aggregation (default: 20.0).

    Returns
    -------
    dict
        Dictionary containing CSDL variables with exact parity to Korn-Lock outputs:
        - 'CD_wave': integrated full-wing wave drag coefficient
        - 'D_wave': integrated wave drag force [N]
        - 'cd_wave_strip': sectional wave drag coefficients on 100 drag strips (shape: (num_drag_strips,))
        - 'M_dd': effective drag divergence Mach numbers (shape: (num_drag_strips,))
        - 'M_crit': effective critical Mach numbers (shape: (num_drag_strips,))
        - 'delta_M': M - M_crit (shape: (num_drag_strips,))
        - 'delta_M_eff': scaled softplus of delta_M (shape: (num_drag_strips,))
        - 'strip_M_n_max': peak normal Mach number per drag strip (shape: (num_drag_strips,))
        - 'abs_cl': None (placeholder for interface compatibility)
    """
    # 1. Recover incompressible Cp0 from Prandtl-Glauert corrected Cp
    beta_inf = csdl.sqrt(1.0 - mach_node**2 + 1e-12)
    cp0 = beta_inf * cp_right

    # 2. Karman-Tsien denominator with differentiable smooth floor
    d_kt = beta_inf + (mach_node**2 / (2.0 * (1.0 + beta_inf))) * cp0
    d_kt_safe = d_min + (1.0 / beta_d) * csdl.softplus(beta_d * (d_kt - d_min))
    cp_kt = cp0 / d_kt_safe

    # 3. Static-to-freestream pressure ratio with differentiable smooth floor
    p_ratio = 1.0 + 0.5 * gamma * (mach_node**2) * cp_kt
    p_ratio_safe = p_min + (1.0 / beta_p) * csdl.softplus(beta_p * (p_ratio - p_min))

    # 4. Isentropic local Mach number squared (avoiding sqrt to prevent derivative singularities)
    gm1 = gamma - 1.0
    M_loc_sq = (2.0 / gm1) * ((1.0 + 0.5 * gm1 * mach_node**2) * (p_ratio_safe ** (-gm1 / gamma)) - 1.0)

    # 5. Map half-chord sweep from drag strips to each panel
    M_drag_to_panel = wave_panel_mapping['M_drag_to_panel']
    sweep_panel = csdl.matvec(csdl.Variable(value=M_drag_to_panel), strip_sweep_halfchord)
    cos_sw = csdl.cos(sweep_panel)

    # 6. Normal Mach number squared
    M_n_sq = M_loc_sq * (cos_sw ** 2)

    # 7. Differentiable supersonic excess with smooth physical weak-shock saturation
    delta_s_raw = (1.0 / beta_s) * csdl.softplus(beta_s * (M_n_sq - 1.0))
    delta_s = delta_s_max * csdl.tanh(delta_s_raw / delta_s_max)

    # 8. Weak-shock Rankine-Hugoniot entropy integrand per panel
    rh_coeff = 2.0 * gamma / (3.0 * ((gamma + 1.0) ** 2))  # ~0.162037
    dcd_w = c_cal * rh_coeff * (delta_s ** 3) * cos_sw
    panel_dw_area = dcd_w * panel_area_right  # [m^2] wave drag area per panel

    # 9. Global full-wing wave drag (factor of 2 for symmetry)
    CD_wave = 2.0 * csdl.sum(panel_dw_area) / planform_area
    D_wave = q_node * planform_area * CD_wave

    # 10. Strip-wise sectional wave drag breakdown on 100 drag strips
    aero_strip_dw_area = csdl.matvec(csdl.Variable(value=W_strip), panel_dw_area)  # shape: (num_aero_strips,)
    dw_area_prime_aero = (2.0 * aero_strip_dw_area) / dy_aero_dyn                # [m^2 / m = m]
    dw_area_prime_drag = csdl.matvec(csdl.Variable(value=M_interp_drag), dw_area_prime_aero)
    cd_wave_strip = dw_area_prime_drag / (local_chord_drag + 1e-12)              # shape: (num_drag_strips,)

    # 11. Per-strip peak normal Mach number aggregation via contiguous panel slices
    strip_panel_slices = wave_panel_mapping['strip_panel_slices']
    m_sq_list = []
    for start, end in strip_panel_slices:
        if end > start:
            sub = M_n_sq[start:end]
            m_max_s = csdl.maximum(sub, axes=(0,), rho=rho_strip_max)
        else:
            m_max_s = csdl.Variable(value=np.array([0.0]))
        m_sq_list.append(csdl.reshape(m_max_s, (1,)))

    aero_M_n_sq_max = csdl.concatenate(m_sq_list)  # shape: (num_aero_strips,)
    aero_M_n_sq_safe = csdl.softplus(rho_strip_max * aero_M_n_sq_max) / rho_strip_max + 1e-12
    aero_M_n_max = csdl.sqrt(aero_M_n_sq_safe)
    strip_M_n_max = csdl.matvec(csdl.Variable(value=M_interp_drag), aero_M_n_max)  # shape: (num_drag_strips,)

    # 12. Buffet & critical Mach diagnostics (matching Korn-Lock conventions)
    m_dd = mach_node + (m_n_buffet_limit - strip_M_n_max)
    m_crit = mach_node + (1.0 - strip_M_n_max)
    delta_m = strip_M_n_max - 1.0
    delta_m_eff = (1.0 / beta_s) * csdl.softplus(beta_s * (strip_M_n_max - 1.0))

    return {
        'CD_wave': CD_wave,
        'D_wave': D_wave,
        'cd_wave_strip': cd_wave_strip,
        'M_dd': m_dd,
        'M_crit': m_crit,
        'delta_M': delta_m,
        'delta_M_eff': delta_m_eff,
        'strip_M_n_max': strip_M_n_max,
        'abs_cl': None,
    }


