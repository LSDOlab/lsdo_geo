"""Transonic Korn-Lock wave-drag model for BWB strip theory.

This module provides a differentiable, geometry- and load-aware transonic
wave-drag correction evaluated along 100 drag strips.

Physical model:
    c_i = norm(TE_i - LE_i)
    (t/c)_i = max_x(||r_upper(x_i) - r_lower(x_i)||) / c_i
    r_0.5 = 0.5 * (LE + TE)
    Lambda_0.5 derived from adjacent spanwise mid-chord points using atan2
    Lprime_i = L_strip_i / dy_aero_dynamic_i
    cl_i = Lprime_i / (q_node * c_i)
    abs_cl_i = sqrt(cl_i**2 + eps_cl**2)

Korn-Lock correlation:
    kappa_A = 0.87 (frozen conventional-airfoil technology factor)
    delta_M_dd_to_crit = (0.1 / 80.0)**(1/3)  # approx 0.1077217345015942
    cos_Lambda = cos(Lambda_0.5)
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
    # dx, dy along the span (ordered by positive y)
    # Interior: centered differences r[i+1] - r[i-1]
    dx_int = r_midchord[2:, 0] - r_midchord[:-2, 0]
    dy_int = r_midchord[2:, 1] - r_midchord[:-2, 1]

    # Root boundary: forward difference r[1] - r[0]
    dx_0 = csdl.reshape(r_midchord[1, 0] - r_midchord[0, 0], (1,))
    dy_0 = csdl.reshape(r_midchord[1, 1] - r_midchord[0, 1], (1,))

    # Tip boundary: backward difference r[-1] - r[-2]
    dx_tip = csdl.reshape(r_midchord[-1, 0] - r_midchord[-2, 0], (1,))
    dy_tip = csdl.reshape(r_midchord[-1, 1] - r_midchord[-2, 1], (1,))

    dx = csdl.concatenate([dx_0, dx_int, dx_tip])
    dy = csdl.concatenate([dy_0, dy_int, dy_tip])

    # atan2(dx, dy) where x is chordwise (streamwise) and y is spanwise
    sweep = csdl.arctan2(dx, dy)
    return sweep


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
    strip_sweep_halfchord: csdl.Variable,
    strip_cl: csdl.Variable,
    strip_area: csdl.Variable,
    total_strip_area: csdl.Variable,
    mach_node: Union[float, csdl.Variable],
    q_node: Union[float, csdl.Variable],
    planform_area: csdl.Variable,
):
    """Evaluate Korn-Lock transonic wave drag on strip level and integrated for a flight node.

    Parameters
    ----------
    strip_tc : csdl.Variable
        Thickness-to-chord ratio (t/c)_i for each strip (shape: (num_strips,)).
    strip_sweep_halfchord : csdl.Variable
        Half-chord sweep angle Lambda_0.5 in radians for each strip (shape: (num_strips,)).
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
    # Differentiable |cl|
    abs_cl = csdl.sqrt(strip_cl**2 + EPS_CL**2)

    cos_lambda = csdl.cos(strip_sweep_halfchord)
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

