"""Differentiable 2D B-spline surface Stratford separation analysis.

Evaluates Stratford's canonical turbulent separation criterion along upper-surface
panel chains and enforces flow attachment (S <= 0.39) via a 2D tensor-product
B-spline surface fit (degree 2 chordwise, degree 3 spanwise) with regional smooth-max
aggregation centered at camber DV influence peaks (and trailing edge) and spanwise stations.
"""

from __future__ import annotations
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
from typing import Dict, Any, List, Tuple
from .bwb_viscous_ibl import build_ibl_mesh_topology, sp_floor


# Numerical constants
S_CRIT = 0.39            # Canonical Stratford turbulent separation threshold
RHO_SMOOTH_MAX = 30.0   # Sharpness for smooth maximum aggregations
NU_DENSE = 31           # Chordwise dense evaluation points
NV_DENSE = 41           # Spanwise dense evaluation points
N_CP_U = 6              # Chordwise control points (degree 2)
N_CP_V = 8              # Spanwise control points (degree 3)


def build_stratford_topology(
    points: np.ndarray,
    cells_dict: Dict[str, np.ndarray],
    num_ffd_stations: int,
    scale_factor: float = 1.0,
    b_tip_ref: float | None = None,
) -> Dict[str, Any]:
    """Construct static topology, B-spline spaces, and 2D evaluation/aggregation structures.

    Parameters
    ----------
    points : np.ndarray
        Mesh points.
    cells_dict : dict
        Mesh cells dictionary containing 'quad' connectivity.
    num_ffd_stations : int
        Number of FFD design variable spanwise stations (5 in fast, 8 in full).
    scale_factor : float
        Wing scale factor.
    b_tip_ref : float, optional
        Reference wing tip y-coordinate. If None, computed from station_span_y.

    Returns
    -------
    dict
        Static matrices, slices, and metadata for Stratford evaluation.
    """
    # Base IBL mesh topology provides station indices and gather matrix
    ibl_topo = build_ibl_mesh_topology(
        points=points,
        cells_dict=cells_dict,
        scale_factor=scale_factor,
    )
    num_mesh_stations = ibl_topo['num_stations']
    station_span_y = ibl_topo['station_span_y']

    if b_tip_ref is None:
        b_tip_ref = float(np.max(station_span_y))

    # Normalized collocation coordinates for the mesh
    # Chordwise: 20 upper panel centers mapped to normalized [0, 1]
    # At interval midpoints with LE at 0.0:
    pts_scaled = points * scale_factor
    quads = cells_dict['quad']
    centers = pts_scaled[quads].mean(axis=1)
    
    st0_u_idx = ibl_topo['upper_indices'][0]
    st0_centers = centers[st0_u_idx]
    x_le = st0_centers[0, 0]
    x_te = st0_centers[-1, 0]
    chord_len = max(x_te - x_le, 1e-3)
    u_colloc = np.clip((st0_centers[:, 0] - x_le) / chord_len, 0.0, 1.0)
    # Ensure strictly increasing for B-spline evaluation
    u_colloc = np.linspace(0.0, 1.0, 20)

    # Spanwise collocation coordinates: station_span_y normalized by b_tip_ref
    v_colloc = np.clip(station_span_y / b_tip_ref, 0.0, 1.0)

    # 1. Chordwise B-Spline Space: Degree 2 (Quadratic), N_CP_U = 6
    # Clamped knot vector: 3 zeros, 3 interior knots, 3 ones
    knots_u = np.concatenate([
        [0.0, 0.0, 0.0],
        np.linspace(0.0, 1.0, 5)[1:-1],
        [1.0, 1.0, 1.0]
    ])
    space_u = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=2,
        coefficients_shape=(N_CP_U,),
        knots=(knots_u,),
    )
    Bu = space_u.compute_basis_matrix(u_colloc.reshape(-1, 1)).toarray()
    Mu_inv = np.linalg.pinv(Bu)  # Shape (6, 20)

    # 2. Spanwise B-Spline Space: Degree 3 (Cubic), N_CP_V = 8
    # Clamped knot vector: 4 zeros, 4 interior knots, 4 ones
    knots_v = np.concatenate([
        [0.0, 0.0, 0.0, 0.0],
        np.linspace(0.0, 1.0, 6)[1:-1],
        [1.0, 1.0, 1.0, 1.0]
    ])
    space_v = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=3,
        coefficients_shape=(N_CP_V,),
        knots=(knots_v,),
    )
    Bv = space_v.compute_basis_matrix(v_colloc.reshape(-1, 1)).toarray()
    # Enforce root symmetry: dS/dv(0) = 0 -> C_{:, 0} = C_{:, 1}
    d_root_v = np.zeros((1, N_CP_V))
    d_root_v[0, 0] = -1.0
    d_root_v[0, 1] = 1.0
    Bv_sys = np.vstack([Bv, d_root_v])
    Mv_inv = np.linalg.pinv(Bv_sys)  # Shape (8, num_mesh_stations + 1)
    Mv_inv_T = Mv_inv.T              # Shape (num_mesh_stations + 1, 8)

    # 3. 2D Tensor-Product B-Spline Space for dense grid evaluation
    space_2d = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(2, 3),
        coefficients_shape=(N_CP_U, N_CP_V),
        knots=(knots_u, knots_v),
    )
    u_dense = np.linspace(0.0, 1.0, NU_DENSE)
    v_dense = np.linspace(0.0, 1.0, NV_DENSE)
    U_grid, V_grid = np.meshgrid(u_dense, v_dense, indexing='ij')
    dense_pts = np.column_stack([U_grid.flatten(), V_grid.flatten()])
    B_dense = space_2d.compute_basis_matrix(dense_pts).toarray()  # Shape (NU_DENSE * NV_DENSE, 48)

    # 4. Regional Aggregation Slices
    # Chordwise centers correspond to camber DV peak influences (Greville abscissae) + TE:
    # [1/6, 1/2, 5/6, 1.0]
    u_centers = [1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0]
    u_mids = [0.0] + [0.5 * (u_centers[i] + u_centers[i + 1]) for i in range(len(u_centers) - 1)] + [1.0]
    u_overlap = 0.02

    u_slices = []
    for i in range(len(u_centers)):
        low = max(0.0, u_mids[i] - (u_overlap if i > 0 else 0.0))
        high = min(1.0, u_mids[i + 1] + (u_overlap if i < len(u_centers) - 1 else 0.0))
        idx = np.where((u_dense >= low - 1e-5) & (u_dense <= high + 1e-5))[0]
        u_slices.append(slice(int(idx[0]), int(idx[-1]) + 1))

    # Spanwise centers correspond to FFD design variable station peaks:
    v_centers = np.linspace(0.0, 1.0, num_ffd_stations)
    v_mids = [0.0] + [0.5 * (v_centers[j] + v_centers[j + 1]) for j in range(num_ffd_stations - 1)] + [1.0]
    v_overlap = 0.02

    v_slices = []
    for j in range(num_ffd_stations):
        low = max(0.0, v_mids[j] - (v_overlap if j > 0 else 0.0))
        high = min(1.0, v_mids[j + 1] + (v_overlap if j < num_ffd_stations - 1 else 0.0))
        idx = np.where((v_dense >= low - 1e-5) & (v_dense <= high + 1e-5))[0]
        v_slices.append(slice(int(idx[0]), int(idx[-1]) + 1))

    num_constraints = len(u_slices) * len(v_slices)

    return {
        'ibl_topology': ibl_topo,
        'num_mesh_stations': num_mesh_stations,
        'num_ffd_stations': num_ffd_stations,
        'num_constraints': num_constraints,
        'Mu_inv': Mu_inv,
        'Mv_inv_T': Mv_inv_T,
        'B_dense': B_dense,
        'u_dense': u_dense,
        'v_dense': v_dense,
        'u_slices': u_slices,
        'v_slices': v_slices,
        'u_centers': u_centers,
        'v_centers': v_centers,
    }


def evaluate_stratford_separation(
    cp_node0: csdl.Variable,
    dynamic_panel_centers: csdl.Variable,
    v_cruise: csdl.Variable,
    rho_cruise: csdl.Variable,
    mu_air: csdl.Variable,
    stratford_topology: Dict[str, Any],
) -> Dict[str, csdl.Variable]:
    """Evaluate CSDL-native Stratford separation criterion and 2D regional constraints.

    Parameters
    ----------
    cp_node0 : csdl.Variable
        VortexAD pressure coefficient array at cruise node 0, shape (total_quads,).
    dynamic_panel_centers : csdl.Variable
        Deformed panel center coordinates, shape (total_quads, 3).
    v_cruise : csdl.Variable
        Freestream cruise speed [m/s].
    rho_cruise : csdl.Variable
        Cruise air density [kg/m^3].
    mu_air : csdl.Variable
        Dynamic air viscosity [Pa*s].
    stratford_topology : dict
        Static topology dictionary from build_stratford_topology.

    Returns
    -------
    dict
        Dictionary containing:
        - 'dv_stratford_constraints': shape (num_constraints,) <= S_CRIT
        - 'stratford_margin': scalar minimum margin (0.39 - max(S))
        - 'S_grid': panel-level cumulative Stratford matrix, shape (20, num_mesh_stations)
        - 'S_dense_2d': dense evaluated surface, shape (NU_DENSE, NV_DENSE)
    """
    topo = stratford_topology['ibl_topology']
    num_mesh_stations = stratford_topology['num_mesh_stations']
    u_slices = stratford_topology['u_slices']
    v_slices = stratford_topology['v_slices']

    M_gather_var = csdl.Variable(value=topo['M_gather'])
    Mu_inv_var = csdl.Variable(value=stratford_topology['Mu_inv'])
    Mv_inv_T_var = csdl.Variable(value=stratford_topology['Mv_inv_T'])
    B_dense_var = csdl.Variable(value=stratford_topology['B_dense'])

    # Gather Cp and panel centers
    cp_flat = csdl.matvec(M_gather_var, cp_node0)
    cp_paths = csdl.reshape(cp_flat, (2, num_mesh_stations, 20))
    cp_upper = cp_paths[0]  # Shape (num_mesh_stations, 20)

    centers_flat = csdl.matmat(M_gather_var, dynamic_panel_centers)
    centers_paths = csdl.reshape(centers_flat, (2, num_mesh_stations, 20, 3))
    centers_upper = centers_paths[0]  # Shape (num_mesh_stations, 20, 3)

    # Station suction peak (minimum Cp) and normalized pressure recovery
    cp_min_st = -csdl.maximum(-cp_upper, axes=(1,), rho=RHO_SMOOTH_MAX)
    cp_min_2d = csdl.expand(cp_min_st, (num_mesh_stations, 20), 'j->ji')
    cp_rec = sp_floor(cp_upper - cp_min_2d, 0.0, 50.0) / sp_floor(1.0 - cp_min_2d, 0.5, 50.0)

    # Leading edge nose distance
    diff_le = centers_paths[0, :, 0, :] - centers_paths[1, :, 0, :]
    le_dist = csdl.sqrt(csdl.sum(diff_le**2, axes=(1,)) + 1e-12)
    s0 = 0.5 * le_dist

    # Compute Stratford parameter on 19 chordwise intervals
    S_list = [csdl.Variable(value=np.zeros(num_mesh_stations))]  # S=0 at LE
    s_curr = s0

    for k in range(19):
        diff = centers_upper[:, k + 1, :] - centers_upper[:, k, :]
        ds_k = csdl.sqrt(csdl.sum(diff**2, axes=(1,)) + 1e-12)
        s_mid = s_curr + 0.5 * ds_k
        s_curr = s_curr + ds_k

        grad_k = (cp_rec[:, k + 1] - cp_rec[:, k]) / ds_k
        adverse_k = sp_floor(s_mid * grad_k, 0.0, 50.0)
        cp_rec_mid = 0.5 * (cp_rec[:, k] + cp_rec[:, k + 1])

        Re_k = (rho_cruise * v_cruise * s_mid) / mu_air
        re_factor = (1e-6 * sp_floor(Re_k, 1000.0, 0.01)) ** (-0.1)

        S_k = cp_rec_mid * csdl.sqrt(adverse_k + 1e-12) * re_factor
        S_list.append(S_k)

    # Cumulative downstream maximum along chord to enforce monotonic separation
    S_cum_list = [csdl.reshape(S_list[0], (1, num_mesh_stations))]
    for k in range(1, 20):
        sub = csdl.concatenate([csdl.reshape(s, (1, num_mesh_stations)) for s in S_list[:k + 1]], axis=0)
        m = csdl.maximum(sub, axes=(0,), rho=RHO_SMOOTH_MAX)
        S_cum_list.append(csdl.reshape(m, (1, num_mesh_stations)))

    # S_grid has shape (20, num_mesh_stations) -> 20 chordwise, num_mesh_stations spanwise
    S_grid = csdl.concatenate(S_cum_list, axis=0)
    S_grid.name = 'S_grid'

    # 2D B-Spline surface fit
    # S_aug appends root derivative symmetry zero column: shape (20, num_mesh_stations + 1)
    zero_col = csdl.Variable(value=np.zeros((20, 1)))
    S_aug = csdl.concatenate([S_grid, zero_col], axis=1)

    # Tensor product solve: C = Mu_inv @ S_aug @ Mv_inv.T
    temp = csdl.matmat(Mu_inv_var, S_aug)
    C_surface = csdl.matmat(temp, Mv_inv_T_var)
    C_flat = csdl.reshape(C_surface, (N_CP_U * N_CP_V,))

    # Evaluate on dense grid (NU_DENSE, NV_DENSE)
    S_dense_flat = csdl.matvec(B_dense_var, C_flat)
    S_dense_2d = csdl.reshape(S_dense_flat, (NU_DENSE, NV_DENSE))
    S_dense_2d.name = 'S_dense_2d'

    # Regional smooth-max aggregation: 4 chordwise x num_ffd_stations spanwise
    constrs = []
    for us in u_slices:
        for vs in v_slices:
            cell_sub = S_dense_2d[us, vs]
            cell_max = csdl.maximum(cell_sub, axes=(0, 1), rho=RHO_SMOOTH_MAX)
            constrs.append(csdl.reshape(cell_max, (1,)))

    dv_stratford_constraints = csdl.concatenate(constrs)
    dv_stratford_constraints.name = 'dv_stratford_constraints'

    # Global maximum Stratford diagnostic and margin
    max_stratford = csdl.maximum(dv_stratford_constraints, axes=(0,), rho=RHO_SMOOTH_MAX)
    max_stratford.name = 'max_stratford'
    stratford_margin = S_CRIT - max_stratford
    stratford_margin.name = 'stratford_margin'

    return {
        'dv_stratford_constraints': dv_stratford_constraints,
        'max_stratford': max_stratford,
        'stratford_margin': stratford_margin,
        'S_grid': S_grid,
        'S_dense_2d': S_dense_2d,
    }

