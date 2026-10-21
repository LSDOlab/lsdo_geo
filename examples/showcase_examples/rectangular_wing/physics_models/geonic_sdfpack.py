"""
Geometric non-interference enforcement using sdfpack.

Provides continuous volumetric signed-distance interference metrics
for TCP0 payload containment in aircraft outer mold lines (OML):
1. Oversized payload cargo box (33 x 13 x 10 ft)
2. 6-pallet cargo stack (shorts and longs)

Implements CSDL CustomExplicitOperation classes with exact analytic/adjoint
sensitivities with respect to LSDO_GEO B-spline OML surface coefficients and
payload position design variables.
"""
from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Tuple
import numpy as np
import csdl_alpha as csdl

try:
    from physics_models.geonic_payload import (
        GEONIC_CLEARANCE_M,
        PAYLOAD_LENGTH_M,
        PAYLOAD_WIDTH_M,
        PAYLOAD_HEIGHT_M,
        PALLET_L_SHORT_M,
        PALLET_WIDTH_M,
        PALLET_HEIGHT_M,
        PALLET_L_LONG_M,
        PALLET_STACK_FRONT_M,
        PALLET_STACK_REFERENCE_CG_X,
    )
except ImportError:
    from .geonic_payload import (
        GEONIC_CLEARANCE_M,
        PAYLOAD_LENGTH_M,
        PAYLOAD_WIDTH_M,
        PAYLOAD_HEIGHT_M,
        PALLET_L_SHORT_M,
        PALLET_WIDTH_M,
        PALLET_HEIGHT_M,
        PALLET_L_LONG_M,
        PALLET_STACK_FRONT_M,
        PALLET_STACK_REFERENCE_CG_X,
    )


def compute_box_sdf_and_grad(
    points: np.ndarray,
    center: np.ndarray,
    half_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact analytical signed-distance function and point gradient for an axis-aligned box.

    Parameters
    ----------
    points : np.ndarray of shape (N, 3)
        Query points in the box coordinate frame.
    center : np.ndarray of shape (3,)
        Box center coordinates.
    half_lengths : np.ndarray of shape (3,)
        Box half-lengths (half_x, half_y, half_z).

    Returns
    -------
    sdf : np.ndarray of shape (N,)
        Signed distance values (negative inside, positive outside).
    d_sdf_d_p : np.ndarray of shape (N, 3)
        Spatial gradient of the signed distance with respect to query point coordinates.
        Note: d_sdf / d_center = - d_sdf / d_p.
    """
    points = np.atleast_2d(points)
    p_diff = points - center
    sign_p = np.sign(p_diff)
    sign_p[sign_p == 0.0] = 1.0
    abs_p = np.abs(p_diff)
    q = abs_p - half_lengths

    max_q0 = np.maximum(q, 0.0)
    norm_max = np.linalg.norm(max_q0, axis=-1)
    max_q = np.max(q, axis=-1)
    argmax_q = np.argmax(q, axis=-1)

    outside = norm_max
    inside = np.minimum(max_q, 0.0)
    sdf = outside + inside

    is_outside = norm_max > 1e-12
    safe_norm = np.where(is_outside, norm_max, 1.0)[:, None]
    grad_outside = (max_q0 / safe_norm) * sign_p
    eye3 = np.eye(3)
    grad_inside = eye3[argmax_q] * sign_p
    d_sdf_d_p = np.where(is_outside[:, None], grad_outside, grad_inside)

    return sdf, d_sdf_d_p


def get_pallet_boxes_spec(
    stack_center_x: float,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Return centers and half-lengths for all 6 full rigid pallet boxes.
    """
    x_shift = stack_center_x - PALLET_STACK_REFERENCE_CG_X
    w = PALLET_WIDTH_M
    half_w = 0.5 * w
    half_h = 0.5 * PALLET_HEIGHT_M
    l_s = PALLET_L_SHORT_M
    l_l = PALLET_L_LONG_M
    base_x = PALLET_STACK_FRONT_M

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

    boxes = []
    for x_start, length, y_min, y_max in boxes_spec:
        c = np.array([x_start + 0.5 * length, 0.5 * (y_min + y_max), 0.0])
        h = np.array([0.5 * length, 0.5 * (y_max - y_min), half_h])
        boxes.append((c, h))
    return boxes


def pallet_stack_sdf_and_grad(
    points: np.ndarray,
    stack_center_x: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute signed distance and gradient for the 6-pallet stack union.
    """
    points = np.atleast_2d(points)
    boxes = get_pallet_boxes_spec(stack_center_x)
    sdfs = []
    grads = []
    for c, h in boxes:
        s, g = compute_box_sdf_and_grad(points, c, h)
        sdfs.append(s)
        grads.append(g)
    sdfs_arr = np.array(sdfs)  # (6, N)
    grads_arr = np.array(grads)  # (6, N, 3)
    best_box = np.argmin(sdfs_arr, axis=0)
    n_pts = points.shape[0]
    min_sdf = sdfs_arr[best_box, np.arange(n_pts)]
    min_grad = grads_arr[best_box, np.arange(n_pts), :]
    return min_sdf, min_grad


class SDFPackOversizedInterferenceOperation(csdl.CustomExplicitOperation):
    """
    CSDL Custom Explicit Operation for volumetric non-interference of the
    TCP0 oversized payload box inside the aircraft OML using sdfpack.
    """

    def __init__(
        self,
        projection_model,
        clearance: float = GEONIC_CLEARANCE_M,
        max_grid_subdivisions: int = 5,
        smoothing_radius: float = 0.01,
        bbox: Optional[Tuple[np.ndarray, np.ndarray] | List[np.ndarray]] = None,
    ):
        super().__init__()
        from lsdo_geo import FunctionSetSignedDistanceOperation
        if not isinstance(projection_model, FunctionSetSignedDistanceOperation):
            self.projection_model = FunctionSetSignedDistanceOperation(
                function_set=projection_model,
                sign_mode="winding_number",
                winding_chord_tolerance=float(clearance) / 10.0,
            )
        else:
            self.projection_model = projection_model
        self.clearance = float(clearance)
        self.max_grid_subdivisions = int(max_grid_subdivisions)
        self.smoothing_radius = float(smoothing_radius)
        self.half_lengths = 0.5 * np.array([PAYLOAD_LENGTH_M, PAYLOAD_WIDTH_M, PAYLOAD_HEIGHT_M])
        if bbox is not None:
            self.bbox = [np.asarray(bbox[0], dtype=float), np.asarray(bbox[1], dtype=float)]
        else:
            # Fixed domain covering payload translations cx in [0, 10] m plus half-length (5.03 m) and padding
            self.bbox = [np.array([-6.0, -3.0, -2.5]), np.array([16.0, 3.0, 2.5])]

    def evaluate(
        self,
        coefficients: csdl.Variable,
        payload_center_x: csdl.Variable,
    ) -> csdl.Variable:
        self.declare_input("coefficients", coefficients)
        self.declare_input("payload_center_x", payload_center_x)

        interference = self.create_output("interference", shape=(1,))

        self.declare_derivative_parameters("interference", "coefficients")
        self.declare_derivative_parameters("interference", "payload_center_x")

        return interference

    def _eval(self, inputs: Dict[str, np.ndarray], compute_grad: bool = False):
        from sdfpack.metrics import InterferenceMetric, WeightedMeanSquare

        coeffs = np.asarray(inputs["coefficients"], dtype=float).reshape(-1, 3)
        cx = float(np.asarray(inputs["payload_center_x"]).flatten()[0])
        center = np.array([cx, 0.0, 0.0])
        half_lengths = self.half_lengths

        def payload_sdf(p):
            sdf, _ = compute_box_sdf_and_grad(p, center, half_lengths)
            return sdf

        def wall_sdf(p):
            p = np.atleast_2d(p)
            d, _ = self.projection_model.project(coeffs, p)
            return -(d + self.clearance)

        im = InterferenceMetric(
            sdfs=[wall_sdf, payload_sdf],
            bbox=self.bbox,
            max_grid_subdivisions=self.max_grid_subdivisions,
            max_sdf_cell_evaluations=1e6,
            smoothing_radius=self.smoothing_radius,
            compute_gradient=compute_grad,
            aggregation=WeightedMeanSquare(),
        )
        val = float(im.value)

        if not compute_grad:
            return val, None, None

        if im.gradient is None or val == 0.0:
            return val, np.zeros_like(coeffs), 0.0

        # Body 0: OML wall
        row0 = im.gradient.getrow(0)
        if row0.data.size > 0:
            pts0 = im.points[row0.indices]
            w0 = row0.data
            _, forward_state = self.projection_model.project(coeffs, pts0)
            _, d_val_dcoeffs = self.projection_model.compute_vjp(coeffs, pts0, -w0, forward_state)
        else:
            d_val_dcoeffs = np.zeros_like(coeffs)

        # Body 1: Payload box
        row1 = im.gradient.getrow(1)
        if row1.data.size > 0:
            pts1 = im.points[row1.indices]
            w1 = row1.data
            _, d_sdf_d_p = compute_box_sdf_and_grad(pts1, center, half_lengths)
            d_val_dcx = -np.sum(w1 * d_sdf_d_p[:, 0])
        else:
            d_val_dcx = 0.0

        return val, d_val_dcoeffs, d_val_dcx

    def compute(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]) -> None:
        val, _, _ = self._eval(inputs, compute_grad=False)
        outputs["interference"] = np.array([val])

    def compute_derivatives(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        derivatives: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        _, d_val_dcoeffs, d_val_dcx = self._eval(inputs, compute_grad=True)
        coeffs = inputs["coefficients"]
        derivatives["interference", "coefficients"] = d_val_dcoeffs.reshape((1,) + coeffs.shape)
        derivatives["interference", "payload_center_x"] = np.array([[d_val_dcx]])


class SDFPackPalletsInterferenceOperation(csdl.CustomExplicitOperation):
    """
    CSDL Custom Explicit Operation for volumetric non-interference of the
    TCP0 6-pallet cargo stack inside the aircraft OML using sdfpack.
    """

    def __init__(
        self,
        projection_model,
        clearance: float = GEONIC_CLEARANCE_M,
        max_grid_subdivisions: int = 5,
        smoothing_radius: float = 0.01,
        bbox: Optional[Tuple[np.ndarray, np.ndarray] | List[np.ndarray]] = None,
    ):
        super().__init__()
        from lsdo_geo import FunctionSetSignedDistanceOperation
        if not isinstance(projection_model, FunctionSetSignedDistanceOperation):
            self.projection_model = FunctionSetSignedDistanceOperation(
                function_set=projection_model,
                sign_mode="winding_number",
                winding_chord_tolerance=float(clearance) / 10.0,
            )
        else:
            self.projection_model = projection_model
        self.clearance = float(clearance)
        self.max_grid_subdivisions = int(max_grid_subdivisions)
        self.smoothing_radius = float(smoothing_radius)
        if bbox is not None:
            self.bbox = [np.asarray(bbox[0], dtype=float), np.asarray(bbox[1], dtype=float)]
        else:
            # Fixed domain covering pallet stack translations cx in [0, 10] m
            self.bbox = [np.array([-10.0, -5.0, -2.5]), np.array([18.0, 5.0, 2.5])]

    def evaluate(
        self,
        coefficients: csdl.Variable,
        pallet_stack_center_x: csdl.Variable,
    ) -> csdl.Variable:
        self.declare_input("coefficients", coefficients)
        self.declare_input("pallet_stack_center_x", pallet_stack_center_x)

        interference = self.create_output("interference", shape=(1,))

        self.declare_derivative_parameters("interference", "coefficients")
        self.declare_derivative_parameters("interference", "pallet_stack_center_x")

        return interference

    def _eval(self, inputs: Dict[str, np.ndarray], compute_grad: bool = False):
        from sdfpack.metrics import InterferenceMetric, WeightedMeanSquare

        coeffs = np.asarray(inputs["coefficients"], dtype=float).reshape(-1, 3)
        cx = float(np.asarray(inputs["pallet_stack_center_x"]).flatten()[0])

        def pallet_sdf(p):
            sdf, _ = pallet_stack_sdf_and_grad(p, cx)
            return sdf

        def wall_sdf(p):
            p = np.atleast_2d(p)
            d, _ = self.projection_model.project(coeffs, p)
            return -(d + self.clearance)

        im = InterferenceMetric(
            sdfs=[wall_sdf, pallet_sdf],
            bbox=self.bbox,
            max_grid_subdivisions=self.max_grid_subdivisions,
            max_sdf_cell_evaluations=1e6,
            smoothing_radius=self.smoothing_radius,
            compute_gradient=compute_grad,
            aggregation=WeightedMeanSquare(),
        )
        val = float(im.value)

        if not compute_grad:
            return val, None, None

        if im.gradient is None or val == 0.0:
            return val, np.zeros_like(coeffs), 0.0

        # Body 0: OML wall
        row0 = im.gradient.getrow(0)
        if row0.data.size > 0:
            pts0 = im.points[row0.indices]
            w0 = row0.data
            _, forward_state = self.projection_model.project(coeffs, pts0)
            _, d_val_dcoeffs = self.projection_model.compute_vjp(coeffs, pts0, -w0, forward_state)
        else:
            d_val_dcoeffs = np.zeros_like(coeffs)

        # Body 1: Pallet stack
        row1 = im.gradient.getrow(1)
        if row1.data.size > 0:
            pts1 = im.points[row1.indices]
            w1 = row1.data
            _, min_grad = pallet_stack_sdf_and_grad(pts1, cx)
            d_val_dcx = -np.sum(w1 * min_grad[:, 0])
        else:
            d_val_dcx = 0.0

        return val, d_val_dcoeffs, d_val_dcx

    def compute(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]) -> None:
        val, _, _ = self._eval(inputs, compute_grad=False)
        outputs["interference"] = np.array([val])

    def compute_derivatives(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        derivatives: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        _, d_val_dcoeffs, d_val_dcx = self._eval(inputs, compute_grad=True)
        coeffs = inputs["coefficients"]
        derivatives["interference", "coefficients"] = d_val_dcoeffs.reshape((1,) + coeffs.shape)
        derivatives["interference", "pallet_stack_center_x"] = np.array([[d_val_dcx]])


def evaluate_sdfpack_geonic(
    projection_model,
    coefficients: csdl.Variable,
    payload_mode: str,
    payload_center_x: Optional[csdl.Variable] = None,
    pallet_stack_center_x: Optional[csdl.Variable] = None,
    clearance: float = GEONIC_CLEARANCE_M,
    max_grid_subdivisions: Optional[int] = None,
    smoothing_radius: Optional[float] = None,
    bbox_oversized: Optional[Tuple[np.ndarray, np.ndarray] | List[np.ndarray]] = None,
    bbox_pallets: Optional[Tuple[np.ndarray, np.ndarray] | List[np.ndarray]] = None,
) -> csdl.Variable | Tuple[csdl.Variable, csdl.Variable]:
    """
    Evaluate geometric non-interference constraints using sdfpack.

    Parameters
    ----------
    projection_model : FunctionSetSignedDistanceOperation | Geometry
        LSDO_GEO signed distance operation or Geometry for the aircraft OML.
    coefficients : csdl.Variable
        Stacked B-spline coefficients of the OML geometry.
    payload_mode : str
        One of 'oversized', 'pallets', or 'both'.
    payload_center_x : csdl.Variable | None
        Oversized payload center x coordinate in metres.
    pallet_stack_center_x : csdl.Variable | None
        Pallet stack CG x coordinate in metres.
    clearance : float
        Minimum interior buffer clearance in metres (default 0.2 m).
    max_grid_subdivisions : int | None
        Octree depth for sdfpack (default from env or 5).
    smoothing_radius : float | None
        Smoothing radius for sdfpack (default from env or 0.01).
    bbox_oversized : tuple or list of np.ndarray | None
        Custom bounding box for oversized payload domain.
    bbox_pallets : tuple or list of np.ndarray | None
        Custom bounding box for pallet stack domain.

    Returns
    -------
    csdl.Variable or tuple[csdl.Variable, csdl.Variable]
        Interference constraint variable(s) (<= 0.0 when satisfied).
    """
    if max_grid_subdivisions is None:
        max_grid_subdivisions = int(os.environ.get("SDFPACK_MAX_SUBDIVISIONS", 5))
    if smoothing_radius is None:
        smoothing_radius = float(os.environ.get("SDFPACK_SMOOTHING_RADIUS", 0.01))

    if payload_mode == "oversized":
        if payload_center_x is None:
            raise ValueError("payload_center_x must be provided for oversized mode")
        op = SDFPackOversizedInterferenceOperation(
            projection_model=projection_model,
            clearance=clearance,
            max_grid_subdivisions=max_grid_subdivisions,
            smoothing_radius=smoothing_radius,
            bbox=bbox_oversized,
        )
        return op.evaluate(coefficients, payload_center_x)

    elif payload_mode == "pallets":
        if pallet_stack_center_x is None:
            raise ValueError("pallet_stack_center_x must be provided for pallets mode")
        op = SDFPackPalletsInterferenceOperation(
            projection_model=projection_model,
            clearance=clearance,
            max_grid_subdivisions=max_grid_subdivisions,
            smoothing_radius=smoothing_radius,
            bbox=bbox_pallets,
        )
        return op.evaluate(coefficients, pallet_stack_center_x)

    elif payload_mode == "both":
        if payload_center_x is None or pallet_stack_center_x is None:
            raise ValueError("Both payload_center_x and pallet_stack_center_x required for 'both' mode")
        op_oversized = SDFPackOversizedInterferenceOperation(
            projection_model=projection_model,
            clearance=clearance,
            max_grid_subdivisions=max_grid_subdivisions,
            smoothing_radius=smoothing_radius,
            bbox=bbox_oversized,
        )
        op_pallets = SDFPackPalletsInterferenceOperation(
            projection_model=projection_model,
            clearance=clearance,
            max_grid_subdivisions=max_grid_subdivisions,
            smoothing_radius=smoothing_radius,
            bbox=bbox_pallets,
        )
        interf_pay = op_oversized.evaluate(coefficients, payload_center_x)
        interf_pal = op_pallets.evaluate(coefficients, pallet_stack_center_x)
        return interf_pay, interf_pal

    else:
        raise ValueError(f"Unknown payload mode for sdfpack: '{payload_mode}'")
