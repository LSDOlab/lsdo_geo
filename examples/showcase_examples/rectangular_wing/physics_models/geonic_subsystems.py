"""
Subsystem geometric non-interference (geonic) modeling and partial-aggregation
constraint formulation for BWB aircraft packaging.

Supports:
1. Nose Landing Gear (NLG) bay (centerline, y=0).
2. Symmetric Main Landing Gear (MLG) bays (left/right, y=+/-mlg_y).
3. Optional outboard elevon actuator housings (chord-scaled, locally twist/sweep rotated).
4. Coupled dynamic sizing:
   - Landing gear bay dimensions scale with (W_total / W_ref)^(1/3) and mass with W_total / W_ref.
   - Actuator housing dimensions scale with local wing chord c_local / c_ref.
5. Encapsulated analytical BoxSignedDistanceFunction using native CSDL operations
   (smooth log-sum-exp max/min) and structured face-grid partial aggregation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import csdl_alpha as csdl
import numpy as np

from physics_models.geonic_payload import (
    GEONIC_CLEARANCE_M,
    PAYLOAD_HEIGHT_M,
    PAYLOAD_LENGTH_M,
    PAYLOAD_WIDTH_M,
    PALLET_HEIGHT_M,
    PALLET_L_LONG_M,
    PALLET_L_SHORT_M,
    PALLET_STACK_FRONT_M,
    PALLET_STACK_REFERENCE_CG_X,
    PALLET_WIDTH_M,
)

# Valid modes
VALID_GEONIC_SUBSYSTEM_MODES: set[str] = {'none', 'landing_gear', 'full'}
VALID_GEONIC_AGGREGATION_MODES: set[str] = {
    'partial_aggregation',
    'pairwise',
    'full_aggregation',
    'unaggregated',
    'volume_only',
}

# -----------------------------------------------------------------------------
# Full-Scale (scale_factor = 7.5) Baseline Subsystem Dimensions & Masses
# -----------------------------------------------------------------------------
# NLG Bay (length, width, height) in metres and mass in kg
NLG_BASE_DIMS_M: np.ndarray = np.array([2.4, 1.0, 1.0], dtype=float)
NLG_BASE_MASS_KG: float = 650.0

# MLG Bay per side (length, width, height) in metres and mass per side in kg
MLG_BASE_DIMS_M: np.ndarray = np.array([3.4, 1.5, 1.2], dtype=float)
MLG_BASE_MASS_KG: float = 1950.0

# Outboard Elevon Actuator Housing per unit (length, width, height) in metres and mass in kg
# Standard thin electromechanical / rotary actuator dimensions for outboard elevon installation
ACTUATOR_BASE_DIMS_M: np.ndarray = np.array([1.0, 0.30, 0.08], dtype=float)
ACTUATOR_MASS_KG: float = 25.0
ACTUATOR_SPAN_FRACTIONS: Tuple[float, ...] = (0.65, 0.85)
ACTUATOR_CHORD_FRACTION: float = 0.65

# Dedicated clearance buffer for compact internal actuators (30-40 mm)
GEONIC_ACTUATOR_CLEARANCE_M: float = 0.04


def parse_geonic_subsystem_mode(
    mode: Optional[str] = None,
    env: Optional[dict] = None,
) -> str:
    """Parse and validate geonic_subsystem_mode ('none', 'landing_gear', 'full')."""
    if env is None:
        env = os.environ
    if mode is None:
        mode = env.get('GEONIC_SUBSYSTEM_MODE', 'none')
    mode = str(mode).strip().lower()
    if mode not in VALID_GEONIC_SUBSYSTEM_MODES:
        raise ValueError(
            f"Invalid geonic_subsystem_mode: '{mode}'. Must be one of {sorted(VALID_GEONIC_SUBSYSTEM_MODES)}."
        )
    return mode


def parse_geonic_aggregation_mode(
    mode: Optional[str] = None,
    env: Optional[dict] = None,
) -> str:
    """Parse and validate geonic_aggregation_mode."""
    if env is None:
        env = os.environ
    if mode is None:
        mode = env.get('GEONIC_AGGREGATION_MODE', 'partial_aggregation')
    mode = str(mode).strip().lower()
    if mode not in VALID_GEONIC_AGGREGATION_MODES:
        raise ValueError(
            f"Invalid geonic_aggregation_mode: '{mode}'. Must be one of {sorted(VALID_GEONIC_AGGREGATION_MODES)}."
        )
    return mode


def csdl_smooth_max(
    x: csdl.Variable,
    axes: Optional[Tuple[int, ...]] = None,
    rho: float = 50.0,
) -> csdl.Variable:
    """Smooth maximum using CSDL's native log-sum-exp implementation."""
    return csdl.maximum(x, axes=axes, rho=rho)


def csdl_smooth_min(
    x: csdl.Variable,
    axes: Optional[Tuple[int, ...]] = None,
    rho: float = 50.0,
) -> csdl.Variable:
    """Smooth minimum using CSDL's native log-sum-exp implementation."""
    return csdl.minimum(x, axes=axes, rho=rho)


@dataclass
class BoxSignedDistanceFunction:
    """
    Analytical signed distance function (SDF) for an axis-aligned or rotated box
    implemented via CSDL operations.

    Convention (matches lsdo_geo signed distance):
      - Negative inside the box
      - Zero on the box boundary
      - Positive outside the box
    """

    half_lengths: Union[np.ndarray, csdl.Variable]
    center: Optional[Union[np.ndarray, csdl.Variable]] = None
    rotation: Optional[Union[np.ndarray, csdl.Variable]] = None
    sharpness: float = 200.0
    eps_reg: float = 1e-12

    def __post_init__(self) -> None:
        if self.half_lengths.shape != (3,):
            raise ValueError(f"half_lengths must have shape (3,), got {self.half_lengths.shape}")
        if self.center is not None and self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")
        if self.rotation is not None and self.rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {self.rotation.shape}")
        if self.sharpness <= 0:
            raise ValueError("sharpness must be positive")

    def __call__(self, points: Union[np.ndarray, csdl.Variable]) -> csdl.Variable:
        return self.evaluate(points)

    def evaluate(self, points: Union[np.ndarray, csdl.Variable]) -> csdl.Variable:
        """
        Evaluate smooth box SDF for query points of shape (N, 3) or (3,).
        Returns shape (N,) or scalar.
        """
        p_world = csdl.Variable(value=points) if isinstance(points, np.ndarray) else points
        if p_world.shape[-1] != 3:
            raise ValueError(f"points must have last dimension 3, got shape {p_world.shape}")

        is_batch = len(p_world.shape) > 1
        if self.center is None:
            p_diff = p_world
        else:
            center_var = csdl.Variable(value=self.center) if isinstance(self.center, np.ndarray) else self.center
            if is_batch:
                center_expanded = csdl.expand(center_var, p_world.shape, 'i->ji')
            else:
                center_expanded = center_var
            p_diff = p_world - center_expanded

        if self.rotation is not None:
            rot_var = csdl.Variable(value=self.rotation) if isinstance(self.rotation, np.ndarray) else self.rotation
            if is_batch:
                p_local = csdl.matmat(p_diff, rot_var)
            else:
                p_local = csdl.matvec(csdl.transpose(rot_var), p_diff)
        else:
            p_local = p_diff

        hl_var = csdl.Variable(value=self.half_lengths) if isinstance(self.half_lengths, np.ndarray) else self.half_lengths
        if is_batch:
            hl_expanded = csdl.expand(hl_var, p_world.shape, 'i->ji')
        else:
            hl_expanded = hl_var

        # Smooth abs: sqrt(x^2 + eps^2) - eps (exact zero at zero, finite derivatives everywhere)
        abs_p = csdl.sqrt(p_local ** 2 + self.eps_reg ** 2) - self.eps_reg
        q = abs_p - hl_expanded

        zero_q = csdl.Variable(value=np.zeros(q.shape))
        outside = csdl.maximum(q, zero_q, rho=self.sharpness)

        last_axis = len(p_world.shape) - 1
        outside_norm = csdl.sqrt(csdl.sum(outside ** 2, axes=(last_axis,)) + self.eps_reg ** 2) - self.eps_reg

        max_q = csdl.maximum(q, axes=(last_axis,), rho=self.sharpness)
        zero_s = csdl.Variable(value=np.zeros(max_q.shape))
        inside = csdl.minimum(max_q, zero_s, rho=self.sharpness)

        return outside_norm + inside


def compute_landing_gear_sizing(
    W_total: csdl.Variable,
    W_ref_lg: float,
    scale_factor: float = 7.5,
    dynamic_sizing: bool = True,
) -> Dict[str, csdl.Variable]:
    """
    Compute NLG and MLG half-lengths and masses, optionally scaled dynamically
    with total aircraft weight W_total.

    Dimensions scale with (W_total / W_ref_lg)^(1/3); mass scales linearly with W_total / W_ref_lg.
    """
    sf_ratio = float(scale_factor) / 7.5
    if dynamic_sizing:
        w_ratio = W_total / float(W_ref_lg)
        dim_scale = w_ratio ** (1.0 / 3.0)
        mass_scale = w_ratio
    else:
        dim_scale = csdl.Variable(value=1.0)
        mass_scale = csdl.Variable(value=1.0)

    nlg_base_hl = 0.5 * NLG_BASE_DIMS_M * sf_ratio
    mlg_base_hl = 0.5 * MLG_BASE_DIMS_M * sf_ratio

    nlg_half_lengths = csdl.Variable(value=nlg_base_hl) * csdl.expand(csdl.reshape(dim_scale, (1,)), (3,))
    mlg_half_lengths = csdl.Variable(value=mlg_base_hl) * csdl.expand(csdl.reshape(dim_scale, (1,)), (3,))

    nlg_mass = (NLG_BASE_MASS_KG * (sf_ratio ** 3)) * mass_scale
    mlg_mass = (MLG_BASE_MASS_KG * (sf_ratio ** 3)) * mass_scale

    return {
        'nlg_half_lengths': nlg_half_lengths,
        'mlg_half_lengths': mlg_half_lengths,
        'nlg_mass': csdl.reshape(nlg_mass, (1,)),
        'mlg_mass': csdl.reshape(mlg_mass, (1,)),
        'lg_dim_scale': csdl.reshape(dim_scale, (1,)),
    }


def compute_actuator_sizing(
    c_local: csdl.Variable,
    c_ref_local: float,
    scale_factor: float = 7.5,
    dynamic_sizing: bool = True,
) -> csdl.Variable:
    """
    Compute outboard elevon actuator housing half-lengths (shape (3,)),
    optionally scaled with local chord c_local / c_ref_local.
    """
    sf_ratio = float(scale_factor) / 7.5
    if dynamic_sizing:
        c_ratio = c_local / float(c_ref_local)
        act_scale = 0.5 + 0.5 * c_ratio
    else:
        act_scale = csdl.Variable(value=1.0)

    base_hl = 0.5 * ACTUATOR_BASE_DIMS_M * sf_ratio
    return csdl.Variable(value=base_hl) * csdl.expand(csdl.reshape(act_scale, (1,)), (3,))


def build_local_frame_from_chord_vector(c_vec: csdl.Variable) -> csdl.Variable:
    """
    Construct a 3x3 orthonormal rotation matrix R = [e_x, e_y, e_z] aligned with
    the local airfoil chord vector c_vec = te_pt - le_pt.

    e_x points along local chord (incorporating twist and sweep);
    e_y is orthogonal to e_x in the spanwise direction;
    e_z = e_x x e_y points roughly upward normal to the local chord plane.
    """
    c_len = csdl.sqrt(csdl.sum(c_vec ** 2) + 1e-24)
    e_x = c_vec / csdl.expand(csdl.reshape(c_len, (1,)), (3,))

    y_ref = csdl.Variable(value=np.array([0.0, 1.0, 0.0]))
    e_z_raw = csdl.cross(e_x, y_ref, axis=0)
    e_z_len = csdl.sqrt(csdl.sum(e_z_raw ** 2) + 1e-24)
    e_z = e_z_raw / csdl.expand(csdl.reshape(e_z_len, (1,)), (3,))

    e_y = csdl.cross(e_z, e_x, axis=0)

    R = csdl.concatenate([
        csdl.reshape(e_x, (3, 1)),
        csdl.reshape(e_y, (3, 1)),
        csdl.reshape(e_z, (3, 1)),
    ], axis=1)
    return R


def overlapping_partition_slices(num_points: int, num_partitions: int = 2) -> List[slice]:
    """Split a 1D grid axis into overlapping partitions sharing boundary indices."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    num_partitions = min(num_partitions, num_points - 1)
    interval_groups = np.array_split(np.arange(num_points - 1), num_partitions)
    slices: List[slice] = []
    for group in interval_groups:
        start = int(group[0])
        stop = int(group[-1]) + 2
        slices.append(slice(start, min(stop, num_points)))
    return slices


def build_box_surface_samples(
    center: csdl.Variable,
    half_lengths: csdl.Variable,
    rotation: Optional[csdl.Variable] = None,
    grid_res: Tuple[int, int] = (3, 3),
    num_partitions_per_axis: int = 2,
    symmetry_y_nonneg: bool = False,
) -> Tuple[csdl.Variable, List[np.ndarray]]:
    """
    Generate structured surface sample points on the boundary of a box
    and return (points_csdl, partition_index_groups).

    Parameters
    ----------
    center : csdl.Variable, shape (3,)
    half_lengths : csdl.Variable, shape (3,)
    rotation : csdl.Variable, shape (3, 3), optional
    grid_res : (nu, nv) per face
    num_partitions_per_axis : number of overlapping partitions per face axis
    symmetry_y_nonneg : if True, restricts local y coordinates to [0, +half_y]
        (for centerline boxes symmetric about y=0).
    """
    nu, nv = grid_res
    u_vals = np.linspace(-1.0, 1.0, nu)
    v_vals = np.linspace(-1.0, 1.0, nv)

    # Local parametric coordinates in [-1, 1]^3 for each face:
    # Each face is represented as (nu, nv, 3) normalized multipliers of half_lengths
    faces_local_norm: List[np.ndarray] = []

    if symmetry_y_nonneg:
        y_vals_pos = np.linspace(0.0, 1.0, nu)
        # +/- x faces (y in [0, 1], z in [-1, 1])
        for sx in (-1.0, 1.0):
            yy, zz = np.meshgrid(y_vals_pos, v_vals, indexing='ij')
            faces_local_norm.append(np.stack([np.full_like(yy, sx), yy, zz], axis=-1))
        # +y face (x in [-1, 1], z in [-1, 1])
        xx, zz = np.meshgrid(u_vals, v_vals, indexing='ij')
        faces_local_norm.append(np.stack([xx, np.ones_like(xx), zz], axis=-1))
        # +/- z faces (x in [-1, 1], y in [0, 1])
        for sz in (-1.0, 1.0):
            xx, yy = np.meshgrid(u_vals, y_vals_pos, indexing='ij')
            faces_local_norm.append(np.stack([xx, yy, np.full_like(xx, sz)], axis=-1))
    else:
        # +/- x faces
        for sx in (-1.0, 1.0):
            yy, zz = np.meshgrid(u_vals, v_vals, indexing='ij')
            faces_local_norm.append(np.stack([np.full_like(yy, sx), yy, zz], axis=-1))
        # +/- y faces
        for sy in (-1.0, 1.0):
            xx, zz = np.meshgrid(u_vals, v_vals, indexing='ij')
            faces_local_norm.append(np.stack([xx, np.full_like(xx, sy), zz], axis=-1))
        # +/- z faces
        for sz in (-1.0, 1.0):
            xx, yy = np.meshgrid(u_vals, v_vals, indexing='ij')
            faces_local_norm.append(np.stack([xx, yy, np.full_like(xx, sz)], axis=-1))

    all_norm = np.concatenate([f.reshape(-1, 3) for f in faces_local_norm], axis=0)
    num_pts = all_norm.shape[0]

    # Build partition index groups on each face
    u_slices = overlapping_partition_slices(nu, num_partitions=num_partitions_per_axis)
    v_slices = overlapping_partition_slices(nv, num_partitions=num_partitions_per_axis)
    partition_groups: List[np.ndarray] = []
    pts_per_face = nu * nv
    for f_idx in range(len(faces_local_norm)):
        base_offset = f_idx * pts_per_face
        idx_grid = np.arange(pts_per_face).reshape(nu, nv) + base_offset
        for us in u_slices:
            for vs in v_slices:
                partition_groups.append(idx_grid[us, vs].ravel())

    norm_var = csdl.Variable(value=all_norm)
    hl_exp = csdl.expand(half_lengths, (num_pts, 3), 'i->ji')
    p_local = norm_var * hl_exp

    if rotation is not None:
        p_rot = csdl.matmat(p_local, csdl.transpose(rotation))
    else:
        p_rot = p_local

    center_exp = csdl.expand(center, (num_pts, 3), 'i->ji')
    points = center_exp + p_rot
    return points, partition_groups


def _as_csdl_var(
    val: Union[np.ndarray, Sequence[float], csdl.Variable],
    expected_shape: Tuple[int, ...],
) -> csdl.Variable:
    """Ensure input is a CSDL Variable with `expected_shape`."""
    if isinstance(val, csdl.Variable):
        var = val
    else:
        var = csdl.Variable(value=np.asarray(val, dtype=float))
    if var.shape != expected_shape:
        var = csdl.reshape(var, expected_shape)
    return var


@dataclass
class BoxRepresentation:
    """Dual explicit (B-spline lsdo_geo.Geometry) + implicit (BoxSignedDistanceFunction)
    representation of a 3D rectangular prism payload or subsystem component.
    """
    name: str
    center: csdl.Variable
    half_lengths: csdl.Variable
    implicit_sdf: BoxSignedDistanceFunction
    explicit_geometry: Any
    rotation: Optional[csdl.Variable] = None


def build_box_bspline_geometry(
    half_lengths: Union[np.ndarray, Sequence[float], csdl.Variable],
    center: Optional[Union[np.ndarray, Sequence[float], csdl.Variable]] = None,
    rotation: Optional[Union[np.ndarray, csdl.Variable]] = None,
    name: str = 'box',
) -> Any:
    """
    Construct a watertight 6-patch degree-(1, 1) B-spline `lsdo_geo.Geometry`
    representing a rectangular box centered at `center` with axis half-lengths
    `half_lengths` and optional 3x3 rotation matrix `rotation` (local-to-body).
    """
    import lsdo_function_spaces as lfs
    import lsdo_geo

    hl_var = _as_csdl_var(half_lengths, (3,))
    c_var = _as_csdl_var(center, (3,)) if center is not None else csdl.Variable(value=np.zeros((3,)))
    rot_var = _as_csdl_var(rotation, (3, 3)) if rotation is not None else None

    # Standard bilinear B-spline space for each planar face: 2x2 control net
    face_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),
        knots=(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])),
    )

    # 8 box corners ordered as (sx in {-1,1}, sy in {-1,1}, sz in {-1,1})
    corner_signs = np.array([
        [sx, sy, sz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ], dtype=float)  # shape (8, 3)

    hl_exp = csdl.expand(hl_var, (8, 3), 'i->ji')
    local_corners = csdl.Variable(value=corner_signs) * hl_exp  # (8, 3)
    if rot_var is not None:
        # local @ R^T  -> shape (8, 3)
        rot_corners = csdl.einsum(local_corners, rot_var, action='ij,kj->ik')
    else:
        rot_corners = local_corners
    corners = rot_corners + csdl.expand(c_var, (8, 3), 'i->ji')

    # Face corner indices shaped (2, 2) with consistent outward normals:
    # Corner index table: idx = 4*(sx>0) + 2*(sy>0) + (sz>0)
    # 0:(-,-,-), 1:(-,-,+), 2:(-,+,-), 3:(-,+,+), 4:(+,-,-), 5:(+,-,+), 6:(+,+,-), 7:(+,+,+)
    face_specs = [
        ('neg_x', np.array([[0, 1], [2, 3]])),  # x = -hx
        ('pos_x', np.array([[4, 6], [5, 7]])),  # x = +hx
        ('neg_y', np.array([[0, 4], [1, 5]])),  # y = -hy
        ('pos_y', np.array([[2, 3], [6, 7]])),  # y = +hy
        ('neg_z', np.array([[0, 2], [4, 6]])),  # z = -hz
        ('pos_z', np.array([[1, 5], [3, 7]])),  # z = +hz
    ]

    functions: Dict[int, lfs.Function] = {}
    function_names: Dict[int, str] = {}
    for f_idx, (face_label, idx_grid) in enumerate(face_specs):
        row0 = csdl.concatenate([
            csdl.reshape(corners[int(idx_grid[0, 0]), :], (1, 1, 3)),
            csdl.reshape(corners[int(idx_grid[0, 1]), :], (1, 1, 3)),
        ], axis=1)
        row1 = csdl.concatenate([
            csdl.reshape(corners[int(idx_grid[1, 0]), :], (1, 1, 3)),
            csdl.reshape(corners[int(idx_grid[1, 1]), :], (1, 1, 3)),
        ], axis=1)
        face_coeffs = csdl.concatenate([row0, row1], axis=0)  # (2, 2, 3)
        functions[f_idx] = lfs.Function(space=face_space, coefficients=face_coeffs, name=f'{name}_{face_label}')
        function_names[f_idx] = f'{name}_{face_label}'

    return lsdo_geo.Geometry(
        functions=functions,
        function_names=function_names,
        name=name,
        space=face_space,
    )


def build_box_representation(
    name: str,
    half_lengths: Union[np.ndarray, Sequence[float], csdl.Variable],
    center: Optional[Union[np.ndarray, Sequence[float], csdl.Variable]] = None,
    rotation: Optional[Union[np.ndarray, csdl.Variable]] = None,
    sharpness: float = 200.0,
) -> BoxRepresentation:
    """Construct BOTH implicit SDF and explicit B-spline representations for a box."""
    hl_var = _as_csdl_var(half_lengths, (3,))
    c_var = _as_csdl_var(center, (3,)) if center is not None else csdl.Variable(value=np.zeros((3,)))
    rot_var = _as_csdl_var(rotation, (3, 3)) if rotation is not None else None
    implicit_sdf = BoxSignedDistanceFunction(
        half_lengths=hl_var,
        center=c_var,
        rotation=rot_var,
        sharpness=sharpness,
    )
    explicit_geom = build_box_bspline_geometry(
        half_lengths=hl_var,
        center=c_var,
        rotation=rot_var,
        name=name,
    )
    return BoxRepresentation(
        name=name,
        center=c_var,
        half_lengths=hl_var,
        implicit_sdf=implicit_sdf,
        explicit_geometry=explicit_geom,
        rotation=rot_var,
    )


def build_payload_representations(
    geonic_payload_mode: str,
    payload_center_x: Optional[csdl.Variable] = None,
    pallet_stack_center_x: Optional[csdl.Variable] = None,
    sharpness: float = 200.0,
) -> Dict[str, List[BoxRepresentation]]:
    """
    Construct dual explicit (B-spline `lsdo_geo.Geometry`) + implicit
    (`BoxSignedDistanceFunction`) representations for every active payload box.
    """
    result: Dict[str, List[BoxRepresentation]] = {'oversized': [], 'pallets': []}
    zero_s = csdl.Variable(value=0.0)

    if geonic_payload_mode in {'oversized', 'both'} and payload_center_x is not None:
        center = csdl.concatenate([
            csdl.reshape(payload_center_x, (1,)),
            csdl.reshape(zero_s, (1,)),
            csdl.reshape(zero_s, (1,)),
        ])
        hl = np.array([0.5 * PAYLOAD_LENGTH_M, 0.5 * PAYLOAD_WIDTH_M, 0.5 * PAYLOAD_HEIGHT_M])
        result['oversized'].append(
            build_box_representation(
                name='oversized_payload',
                half_lengths=hl,
                center=center,
                sharpness=sharpness,
            )
        )

    if geonic_payload_mode in {'pallets', 'both'} and pallet_stack_center_x is not None:
        x_shift = pallet_stack_center_x - PALLET_STACK_REFERENCE_CG_X
        w = PALLET_WIDTH_M
        h = PALLET_HEIGHT_M
        l_s = PALLET_L_SHORT_M
        l_l = PALLET_L_LONG_M
        base_x = PALLET_STACK_FRONT_M

        # Full 6-pallet stack specs: (label, cx_offset, cy, half_l, half_w, half_h)
        specs = [
            ('pallet_front_short', 0.5 * l_s, 0.0, 0.5 * l_s, 0.5 * w, 0.5 * h),
            ('pallet_second_short_right', 1.5 * l_s, 0.5 * w, 0.5 * l_s, 0.5 * w, 0.5 * h),
            ('pallet_second_short_left', 1.5 * l_s, -0.5 * w, 0.5 * l_s, 0.5 * w, 0.5 * h),
            ('pallet_center_long', 2.0 * l_s + 0.5 * l_l, 0.0, 0.5 * l_l, 0.5 * w, 0.5 * h),
            ('pallet_outer_long_right', 2.0 * l_s + 0.5 * l_l, w, 0.5 * l_l, 0.5 * w, 0.5 * h),
            ('pallet_outer_long_left', 2.0 * l_s + 0.5 * l_l, -w, 0.5 * l_l, 0.5 * w, 0.5 * h),
        ]
        for pal_name, cx_off, cy, hl_x, hl_y, hl_z in specs:
            cx = base_x + cx_off + x_shift
            c_vec = csdl.concatenate([
                csdl.reshape(cx, (1,)),
                csdl.Variable(value=np.array([cy])),
                csdl.Variable(value=np.array([0.0])),
            ])
            result['pallets'].append(
                build_box_representation(
                    name=pal_name,
                    half_lengths=np.array([hl_x, hl_y, hl_z]),
                    center=c_vec,
                    sharpness=sharpness,
                )
            )

    return result


def build_payload_box_sdfs(
    geonic_payload_mode: str,
    payload_center_x: Optional[csdl.Variable] = None,
    pallet_stack_center_x: Optional[csdl.Variable] = None,
    sharpness: float = 200.0,
) -> Dict[str, List[BoxSignedDistanceFunction]]:
    """
    Construct BoxSignedDistanceFunction objects for active payload loading cases
    so permanent subsystems (NLG, MLG) can enforce exterior non-interference against them.
    """
    reps = build_payload_representations(
        geonic_payload_mode=geonic_payload_mode,
        payload_center_x=payload_center_x,
        pallet_stack_center_x=pallet_stack_center_x,
        sharpness=sharpness,
    )
    return {
        'oversized': [rep.implicit_sdf for rep in reps['oversized']],
        'pallets': [rep.implicit_sdf for rep in reps['pallets']],
    }


def aggregate_constraints_by_mode(
    raw_residuals: csdl.Variable,
    partition_groups: List[np.ndarray],
    aggregation_mode: str,
    rho_local: float = 50.0,
    rho_global: float = 50.0,
) -> csdl.Variable:
    """
    Aggregate a 1D residual vector g <= 0 according to geonic_aggregation_mode:
      - 'partial_aggregation': smooth max over each overlapping face partition
      - 'full_aggregation': single global smooth max
      - 'unaggregated' / 'pairwise': raw point-wise residuals unchanged
    """
    if aggregation_mode == 'partial_aggregation':
        agg_list = []
        for idx_arr in partition_groups:
            subset = csdl.concatenate([csdl.reshape(raw_residuals[int(i)], (1,)) for i in idx_arr])
            agg_list.append(csdl.reshape(csdl_smooth_max(subset, axes=(0,), rho=rho_local), (1,)))
        return csdl.concatenate(agg_list)
    elif aggregation_mode == 'full_aggregation':
        return csdl.reshape(csdl_smooth_max(raw_residuals, axes=(0,), rho=rho_global), (1,))
    elif aggregation_mode in {'unaggregated', 'pairwise'}:
        return raw_residuals
    else:
        raise ValueError(f"Unsupported aggregation_mode '{aggregation_mode}' for residual aggregation.")


def evaluate_exterior_non_interference(
    query_points: csdl.Variable,
    obstacle_sdfs: Sequence[BoxSignedDistanceFunction],
    clearance_buffer: float = GEONIC_CLEARANCE_M,
    rho_union: float = 50.0,
    aggregation_mode: str = 'partial_aggregation',
) -> Tuple[csdl.Variable, csdl.Variable]:
    """
    Evaluate exterior-exterior non-interference between query_points on component j
    and a list of obstacle SDFs (b != j).

    Returns
    -------
    (raw_residuals, pairwise_or_union_distances)
      raw_residuals:
        - In 'pairwise' mode: concatenated (-d_b(p) + clearance) for all b.
        - Otherwise: (-smooth_min_b(d_b(p)) + clearance) of shape (N,).
    """
    if len(obstacle_sdfs) == 0:
        raise ValueError("obstacle_sdfs cannot be empty")

    d_list = [csdl.reshape(sdf.evaluate(query_points), (1, query_points.shape[0])) for sdf in obstacle_sdfs]
    d_stacked = csdl.concatenate(d_list, axis=0)  # (num_obstacles, N)

    if aggregation_mode == 'pairwise':
        # Exhaustive pairwise residuals without smooth union
        raw_residuals = -d_stacked.reshape((-1,)) + clearance_buffer
        min_dist = csdl_smooth_min(d_stacked, axes=(0,), rho=rho_union)
        return raw_residuals, min_dist

    if len(obstacle_sdfs) == 1:
        union_dist = d_stacked[0, :]
    else:
        union_dist = csdl_smooth_min(d_stacked, axes=(0,), rho=rho_union)

    raw_residuals = -union_dist + clearance_buffer
    return raw_residuals, union_dist


def get_box_corners_3d(
    center: np.ndarray,
    half_lengths: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return the 8 3D corner coordinates (shape (8, 3)) of a box in body frame
    for PyVista wireframe/surface visualization.
    """
    hx, hy, hz = float(half_lengths[0]), float(half_lengths[1]), float(half_lengths[2])
    local_corners = np.array([
        [sx * hx, sy * hy, sz * hz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ], dtype=float)
    if rotation is not None:
        local_corners = local_corners @ rotation.T
    return local_corners + np.asarray(center, dtype=float).reshape(1, 3)


def plot_geometry_with_payloads(
    wing_geometry: Any,
    payload_representations: Optional[Union[Dict[str, List[BoxRepresentation]], Sequence[BoxRepresentation]]] = None,
    subsystem_representations: Optional[Sequence[BoxRepresentation]] = None,
    sample_points: Optional[Sequence[Tuple[str, Union[csdl.Variable, np.ndarray]]]] = None,
    title: str = 'BWB OML Geometry & Payload / Subsystem Packaging',
    interactive: bool = True,
    screenshot: str = '',
    show: bool = True,
) -> list:
    """
    Plot the aircraft OML B-spline geometry alongside all active explicit B-spline
    payload and subsystem geometries (plus optional SDF sample point clouds) using
    LSDO_GEO / PyVista.
    """
    import lsdo_function_spaces.utils.plotting_functions as pf

    plotting_elements: list = []
    plotting_elements = wing_geometry.plot(
        point_types=['evaluated_points'],
        plot_types=['function'],
        opacity=0.25,
        color='#00629B',
        additional_plotting_elements=plotting_elements,
        show=False,
    )

    if isinstance(payload_representations, dict):
        oversized_list = list(payload_representations.get('oversized', []))
        pallet_list = list(payload_representations.get('pallets', []))
    elif payload_representations is not None:
        oversized_list = [r for r in payload_representations if 'pallet' not in r.name]
        pallet_list = [r for r in payload_representations if 'pallet' in r.name]
    else:
        oversized_list, pallet_list = [], []

    for rep in oversized_list:
        plotting_elements = rep.explicit_geometry.plot(
            point_types=['evaluated_points'],
            plot_types=['function', 'wireframe'],
            opacity=0.55,
            color='#9467bd',
            line_width=2.0,
            additional_plotting_elements=plotting_elements,
            show=False,
        )

    for rep in pallet_list:
        plotting_elements = rep.explicit_geometry.plot(
            point_types=['evaluated_points'],
            plot_types=['function', 'wireframe'],
            opacity=0.55,
            color='cyan',
            line_width=1.5,
            additional_plotting_elements=plotting_elements,
            show=False,
        )

    if subsystem_representations is not None:
        for rep in subsystem_representations:
            if 'nlg' in rep.name:
                color_str = 'orange'
            elif 'mlg' in rep.name:
                color_str = 'limegreen'
            else:
                color_str = 'gold'
            plotting_elements = rep.explicit_geometry.plot(
                point_types=['evaluated_points'],
                plot_types=['function', 'wireframe'],
                opacity=0.65,
                color=color_str,
                line_width=2.0,
                additional_plotting_elements=plotting_elements,
                show=False,
            )

    if sample_points is not None:
        for pt_label, pts in sample_points:
            pts_val = np.asarray(pts.value if hasattr(pts, 'value') else pts, dtype=float)
            if pts_val.size == 0:
                continue
            pts_val = pts_val.reshape((-1, 3))
            pt_color = '#d62728' if 'subsys' in pt_label or 'lg' in pt_label or 'act' in pt_label else '#1f77b4'
            plotting_elements = pf.plot_points(
                pts_val,
                opacity=0.95,
                color=pt_color,
                size=8.0,
                additional_plotting_elements=plotting_elements,
                show=False,
            )

    if show:
        pf.show_plot(
            plotting_elements,
            title=title,
            axes=True,
            view_up='z',
            interactive=interactive,
            screenshot=screenshot,
        )
    return plotting_elements
