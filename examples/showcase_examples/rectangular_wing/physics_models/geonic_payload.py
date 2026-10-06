"""
TCP0 payload geometric non-interference (geonic) containment models.

Supports:
1. TCP0 oversized single cargo box (33 x 13 x 10 ft).
2. TCP0 6-pallet cargo stack (88 x 108 x 96 in shorts and 440 x 108 x 96 in longs).

Provides dimensional constants, CSDL sample builders with symmetry reduction,
clearance/margin calculations, and rigid box corner generation for visualization.
"""
from __future__ import annotations

import os
from typing import Literal
import numpy as np
import csdl_alpha as csdl

# Named clearance buffer: 0.2 m minimum interior clearance
GEONIC_CLEARANCE_M: float = 0.2

# Valid geonic payload modes
VALID_GEONIC_PAYLOAD_MODES: set[str] = {'none', 'oversized', 'pallets', 'both'}

# -----------------------------------------------------------------------------
# 1. OVERSIZED PAYLOAD DIMENSIONS (33 x 13 x 10 ft)
# -----------------------------------------------------------------------------
FT2M: float = 1.0 / 3.28084
PAYLOAD_LENGTH_M: float = 33.0 * FT2M   # ~10.058399 m
PAYLOAD_WIDTH_M: float = 13.0 * FT2M    # ~3.962400 m
PAYLOAD_HEIGHT_M: float = 10.0 * FT2M   # ~3.048000 m

# -----------------------------------------------------------------------------
# 2. PALLET STACK DIMENSIONS (88 x 108 x 96 in shorts, 5x length longs)
# -----------------------------------------------------------------------------
IN2M: float = 1.0 / 39.3701
PALLET_L_SHORT_M: float = 88.0 * IN2M    # ~2.235199 m
PALLET_WIDTH_M: float = 108.0 * IN2M     # ~2.743198 m
PALLET_HEIGHT_M: float = 96.0 * IN2M     # ~2.438399 m
PALLET_L_LONG_M: float = 5.0 * PALLET_L_SHORT_M  # ~11.175994 m
PALLET_STACK_FRONT_M: float = 5.0

# Exact volume-weighted reference CG x-coordinate of the 6-pallet stack:
# Box volumes: 1 front short (V), 2 second shorts (2V), 1 center long (5V), 2 outer longs (10V) -> Total = 18V
# Weighted CG_x = 5.0 + (71 / 18) * L_short = 13.816617461248795 m
PALLET_STACK_REFERENCE_CG_X: float = 13.816617461248795


def parse_geonic_payload_mode(
    mode: str | None = None,
    env: dict | None = None,
) -> tuple[str, bool]:
    """
    Parse and validate the geonic payload mode.

    Parameters
    ----------
    mode : str | None
        Explicit payload mode string ('none', 'oversized', 'pallets', 'both').
    env : dict | None
        Environment mapping (defaults to os.environ).

    Returns
    -------
    tuple[str, bool]
        (geonic_payload_mode, use_geonic)
    """
    if env is None:
        env = os.environ

    if mode is None:
        if 'GEONIC_PAYLOAD_MODE' in env:
            mode = str(env['GEONIC_PAYLOAD_MODE']).strip().lower()
        elif 'USE_GEONIC' in env:
            mode = 'oversized' if str(env['USE_GEONIC']).strip() == '1' else 'none'
        else:
            mode = 'none'
    else:
        mode = str(mode).strip().lower()

    if mode not in VALID_GEONIC_PAYLOAD_MODES:
        raise ValueError(
            f"Invalid geonic_payload_mode: '{mode}'. Must be one of {sorted(VALID_GEONIC_PAYLOAD_MODES)}."
        )

    use_geonic = (mode != 'none')
    return mode, use_geonic


def build_geonic_payload_sample_points(
    payload_center_x: csdl.Variable,
    num_chord_points: int = 5,
) -> csdl.Variable:
    """
    Construct CSDL sample points for the TCP0 oversized payload box
    in the aircraft body frame, discretized in the chordwise direction.

    By default, discretizes the box into 5 chordwise stations (0%, 25%, 50%, 75%, 100%)
    along the top and bottom faces at the symmetry reduction planes (y in {0, width/2}),
    producing 20 sample points to capture chordwise OML thickness/camber variation.

    Passing num_chord_points=2 yields the legacy 8-point corner configuration.

    Parameters
    ----------
    payload_center_x : csdl.Variable
        Scalar CSDL variable for payload center x-coordinate in metres.
    num_chord_points : int, optional
        Number of chordwise stations (>= 2). Default is 5 (20 sample points).

    Returns
    -------
    csdl.Variable
        CSDL variable of shape (num_chord_points * 4, 3) containing sample points.
    """
    half_l = 0.5 * PAYLOAD_LENGTH_M
    half_w = 0.5 * PAYLOAD_WIDTH_M
    half_h = 0.5 * PAYLOAD_HEIGHT_M

    if num_chord_points < 2:
        raise ValueError(f"num_chord_points must be at least 2, got {num_chord_points}")

    fractions = np.linspace(0.0, 1.0, num_chord_points)
    x_front = payload_center_x - half_l

    pts = []
    # Deterministic ordering: chordwise (front to rear), root (y=0) to right (y=half_w), bottom to top
    for frac in fractions:
        x_expr = x_front + frac * PAYLOAD_LENGTH_M
        for y_val in [0.0, half_w]:
            for z_val in [-half_h, half_h]:
                pt = csdl.reshape(
                    csdl.concatenate([
                        csdl.reshape(x_expr, (1,)),
                        csdl.Variable(value=np.array([y_val])),
                        csdl.Variable(value=np.array([z_val])),
                    ]),
                    (1, 3),
                )
                pts.append(pt)

    payload_sample_points = csdl.concatenate(pts)
    return payload_sample_points


def build_geonic_pallet_sample_points(
    pallet_stack_center_x: csdl.Variable,
    num_chord_points_short: int = 3,
    num_chord_points_long: int = 6,
) -> csdl.Variable:
    """
    Construct CSDL sample points for the TCP0 6-pallet stack in the aircraft
    body frame, using wing symmetry reduction and chordwise discretization.

    The 4 representative modeled boxes on the right / center half-wing are:
      1. Front short (centerline): x in [5.0, 5.0 + L_short], y in {0, +W/2}, z in {-H/2, +H/2}
      2. Right second-row short:   x in [5.0 + L_short, 5.0 + 2*L_short], y in {0, +W}, z in {-H/2, +H/2}
      3. Center long:              x in [5.0 + 2*L_short, 5.0 + 7*L_short], y in {0, +W/2}, z in {-H/2, +H/2}
      4. Right outer long:         x in [5.0 + 2*L_short, 5.0 + 7*L_short], y in {+W/2, +3W/2}, z in {-H/2, +H/2}

    All pallets are shifted in x by:
      pallet_stack_x_shift = pallet_stack_center_x - PALLET_STACK_REFERENCE_CG_X

    By default:
      - Short boxes (length ~2.24m) use num_chord_points_short=3 (spacing ~1.12m)
      - Long boxes (length ~11.18m) use num_chord_points_long=6 (spacing ~2.24m matching L_short)
      This produces (3 + 3 + 6 + 6) * 4 = 72 sample points, capturing chordwise thickness variation.

    Passing num_chord_points_short=2 and num_chord_points_long=2 yields the legacy 32-point configuration.

    Parameters
    ----------
    pallet_stack_center_x : csdl.Variable
        Scalar CSDL variable for pallet stack CG x-coordinate in metres.
    num_chord_points_short : int, optional
        Number of chordwise stations per short box (>= 2). Default is 3.
    num_chord_points_long : int, optional
        Number of chordwise stations per long box (>= 2). Default is 6.

    Returns
    -------
    csdl.Variable
        CSDL variable containing the sample points.
    """
    if num_chord_points_short < 2 or num_chord_points_long < 2:
        raise ValueError("Chordwise point counts must be at least 2")

    x_shift = pallet_stack_center_x - PALLET_STACK_REFERENCE_CG_X
    w = PALLET_WIDTH_M
    half_w = 0.5 * w
    half_h = 0.5 * PALLET_HEIGHT_M
    l_s = PALLET_L_SHORT_M
    l_l = PALLET_L_LONG_M
    base_x = PALLET_STACK_FRONT_M

    # Box definitions: (x_start_offset, length, [y_plane_0, y_plane_1], num_chord_pts)
    boxes_def = [
        # 1. Front short: x in [5.0, 5.0 + L_s], y in {0, W/2}
        (0.0, l_s, [0.0, half_w], num_chord_points_short),
        # 2. Right second-row short: x in [5.0 + L_s, 5.0 + 2*L_s], y in {0, W}
        (l_s, l_s, [0.0, w], num_chord_points_short),
        # 3. Center long: x in [5.0 + 2*L_s, 5.0 + 2*L_s + L_l], y in {0, W/2}
        (2.0 * l_s, l_l, [0.0, half_w], num_chord_points_long),
        # 4. Right outer long: x in [5.0 + 2*L_s, 5.0 + 2*L_s + L_l], y in {W/2, 1.5*W}
        (2.0 * l_s, l_l, [half_w, 1.5 * w], num_chord_points_long),
    ]

    pts = []
    for x_offset, length, y_planes, n_chord in boxes_def:
        fractions = np.linspace(0.0, 1.0, n_chord)
        for frac in fractions:
            x_expr = base_x + x_offset + frac * length + x_shift
            for y_val in y_planes:
                for z_val in [-half_h, half_h]:
                    pt = csdl.reshape(
                        csdl.concatenate([
                            csdl.reshape(x_expr, (1,)),
                            csdl.Variable(value=np.array([y_val])),
                            csdl.Variable(value=np.array([z_val])),
                        ]),
                        (1, 3),
                    )
                    pts.append(pt)

    pallet_sample_points = csdl.concatenate(pts)
    return pallet_sample_points


def compute_geonic_clearance_and_margin(
    payload_signed_distance: csdl.Variable,
    clearance_buffer: float = GEONIC_CLEARANCE_M,
    rho: float = 50.0,
) -> tuple[csdl.Variable, csdl.Variable, csdl.Variable]:
    """
    Compute geonic constraint values, per-point clearance, and smooth margin.

    With BSM3 enclosed convention:
      payload_signed_distance: negative inside, positive outside
      geonic_constraint_values = payload_signed_distance + clearance_buffer <= 0
      geonic_clearance_per_point = -payload_signed_distance
      geonic_margin = -csdl.maximum(geonic_constraint_values, axes=(0,), rho=rho)

    A geonic_margin >= 0 indicates all sample points satisfy the minimum buffer.
    """
    geonic_constraint_values = payload_signed_distance + clearance_buffer
    geonic_clearance_per_point = -payload_signed_distance
    geonic_margin = -csdl.maximum(geonic_constraint_values, axes=(0,), rho=rho)
    return geonic_constraint_values, geonic_clearance_per_point, geonic_margin


def get_full_payload_box_corners(
    center_x: float,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> np.ndarray:
    """
    Return all 8 3D corner coordinates (shape (8, 3)) of the full rigid oversized payload box
    in the body frame (including both left and right sides).
    """
    half_l = 0.5 * PAYLOAD_LENGTH_M
    half_w = 0.5 * PAYLOAD_WIDTH_M
    half_h = 0.5 * PAYLOAD_HEIGHT_M

    corners = []
    for x in [center_x - half_l, center_x + half_l]:
        for y in [-half_w, half_w]:
            for z in [center_z - half_h, center_z + half_h]:
                corners.append([x, y, z])
    return np.asarray(corners, dtype=float)


def get_full_pallet_boxes_corners(pallet_stack_center_x: float) -> list[np.ndarray]:
    """
    Return the 8 3D corner coordinates (shape (8, 3)) for each of the 6 full rigid
    pallet boxes in the body frame for 3D visualization.

    Returns
    -------
    list[np.ndarray]
        List of 6 NumPy arrays of shape (8, 3).
    """
    x_shift = pallet_stack_center_x - PALLET_STACK_REFERENCE_CG_X
    w = PALLET_WIDTH_M
    half_w = 0.5 * w
    half_h = 0.5 * PALLET_HEIGHT_M
    l_s = PALLET_L_SHORT_M
    l_l = PALLET_L_LONG_M
    base_x = PALLET_STACK_FRONT_M

    # 6 full pallet boxes: (x_start, length, y_min, y_max)
    boxes_spec = [
        # 1. Front short: centerline
        (base_x + x_shift, l_s, -half_w, half_w),
        # 2. Second short: left
        (base_x + l_s + x_shift, l_s, -w, 0.0),
        # 3. Second short: right
        (base_x + l_s + x_shift, l_s, 0.0, w),
        # 4. Center long: centerline
        (base_x + 2.0 * l_s + x_shift, l_l, -half_w, half_w),
        # 5. Outer long: left
        (base_x + 2.0 * l_s + x_shift, l_l, -1.5 * w, -half_w),
        # 6. Outer long: right
        (base_x + 2.0 * l_s + x_shift, l_l, half_w, 1.5 * w),
    ]

    pallet_boxes = []
    for x_start, length, y_min, y_max in boxes_spec:
        corners = []
        for x in [x_start, x_start + length]:
            for y in [y_min, y_max]:
                for z in [-half_h, half_h]:
                    corners.append([x, y, z])
        pallet_boxes.append(np.asarray(corners, dtype=float))

    return pallet_boxes
