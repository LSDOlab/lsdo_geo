"""
TCP0 oversized-payload geometric non-interference (geonic) containment model.

Provides dimensional constants, CSDL 8-point sample builder with symmetry reduction,
clearance/margin calculations, and rigid box corner generation for visualization.
"""
from __future__ import annotations

import numpy as np
import csdl_alpha as csdl

# TCP0 oversized-payload dimensions in metres (converted from feet)
FT2M: float = 1.0 / 3.28084
PAYLOAD_LENGTH_M: float = 33.0 * FT2M   # ~10.058399 m
PAYLOAD_WIDTH_M: float = 13.0 * FT2M    # ~3.962400 m
PAYLOAD_HEIGHT_M: float = 10.0 * FT2M   # ~3.048000 m

# Named clearance buffer: 0.2 m minimum interior clearance
GEONIC_CLEARANCE_M: float = 0.2


def build_geonic_payload_sample_points(payload_center_x: csdl.Variable) -> csdl.Variable:
    """
    Construct the eight CSDL sample points for the TCP0 oversized payload box
    in the aircraft body frame.

    x in {payload_center_x - payload_length / 2, payload_center_x + payload_length / 2}
    y in {0, payload_width / 2}
    z in {-payload_height / 2, payload_height / 2}

    The four y = payload_width / 2 points are right-side box corners.
    The four y = 0 points are root-plane samples constraining centerline pocket thickness.
    Negative-y mirrors are omitted due to wing symmetry.

    Parameters
    ----------
    payload_center_x : csdl.Variable
        Scalar CSDL variable for payload center x-coordinate in metres.

    Returns
    -------
    csdl.Variable
        CSDL variable of shape (8, 3) containing the 8 sample points.
    """
    half_l = 0.5 * PAYLOAD_LENGTH_M
    half_w = 0.5 * PAYLOAD_WIDTH_M
    half_h = 0.5 * PAYLOAD_HEIGHT_M

    x_front = payload_center_x - half_l
    x_rear = payload_center_x + half_l

    pts = []
    # Deterministic ordering: front to rear, root (y=0) to right (y=half_w), bottom to top
    for x_expr in [x_front, x_rear]:
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
    Return all 8 3D corner coordinates (shape (8, 3)) of the full rigid payload box
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
