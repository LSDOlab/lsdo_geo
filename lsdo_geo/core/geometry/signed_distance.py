"""
Reusable LSDO_GEO Signed Distance Field (SDF) and containment operations.

Supports closed codimension-one boundaries:
- 2D: closed B-spline curve loops in R^2
- 3D: closed B-spline surface shells in R^3

Signed distance convention: negative inside, positive outside.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Sequence, Union

import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sp
import lsdo_function_spaces as lfs

try:
    _CustomExplicitOperation = csdl.experimental.CustomExplicitOperationBeta
except AttributeError:
    _CustomExplicitOperation = object


class _CoeffArray(np.ndarray):
    """Subclass of np.ndarray that provides a .value property pointing to self."""

    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    @property
    def value(self):
        return self


# -----------------------------------------------------------------------------
# Geometric Utilities: Solid Angle and Winding Number
# -----------------------------------------------------------------------------
def compute_solid_angles_3d(
    query_pts: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute Generalized Winding Numbers in R^3 using the Van Oosterom-Strackee
    (1983) formula for triangle solid angles.

    Parameters
    ----------
    query_pts : np.ndarray, shape (M, 3)
        Evaluation query points.
    vertices : np.ndarray, shape (V, 3)
        Oriented triangle mesh vertices.
    faces : np.ndarray, shape (F, 3)
        Oriented triangle vertex indices.

    Returns
    -------
    w : np.ndarray, shape (M,)
        Normalized winding numbers (approx 1.0 inside, 0.0 outside, 0.5 on surface).
    """
    p0 = vertices[faces[:, 0]]  # (F, 3)
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    num_pts = len(query_pts)
    w = np.zeros(num_pts, dtype=np.float64)

    # Chunk query points to bound memory for large facet counts
    chunk_size = max(1, min(100, int(5e5 // max(len(faces), 1))))
    for i in range(0, num_pts, chunk_size):
        q_chunk = query_pts[i : i + chunk_size, None, :]  # (C, 1, 3)
        a = p0[None, :, :] - q_chunk                      # (C, F, 3)
        b = p1[None, :, :] - q_chunk
        c = p2[None, :, :] - q_chunk

        la = np.linalg.norm(a, axis=-1)
        lb = np.linalg.norm(b, axis=-1)
        lc = np.linalg.norm(c, axis=-1)

        b_cross_c = np.cross(b, c)
        numer = np.sum(a * b_cross_c, axis=-1)  # (C, F)

        adotb = np.sum(a * b, axis=-1)
        adotc = np.sum(a * c, axis=-1)
        bdotc = np.sum(b * c, axis=-1)

        denom = (
            la * lb * lc
            + adotb * lc
            + adotc * lb
            + bdotc * la
        )
        omega = 2.0 * np.arctan2(numer, denom)
        w[i : i + chunk_size] = np.sum(omega, axis=-1) / (4.0 * np.pi)

    return w


def compute_winding_numbers_2d(
    query_pts: np.ndarray,
    vertices: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """
    Compute Generalized Winding Numbers in R^2 by summing oriented subtended angles.

    Parameters
    ----------
    query_pts : np.ndarray, shape (M, 2)
    vertices : np.ndarray, shape (V, 2)
    segments : np.ndarray, shape (S, 2)

    Returns
    -------
    w : np.ndarray, shape (M,)
        Normalized winding numbers (approx 1.0 inside, 0.0 outside).
    """
    p0 = vertices[segments[:, 0]]  # (S, 2)
    p1 = vertices[segments[:, 1]]  # (S, 2)

    num_pts = len(query_pts)
    w = np.zeros(num_pts, dtype=np.float64)

    chunk_size = max(1, min(200, int(1e6 // max(len(segments), 1))))
    for i in range(0, num_pts, chunk_size):
        q_chunk = query_pts[i : i + chunk_size, None, :]  # (C, 1, 2)
        a = p0[None, :, :] - q_chunk                      # (C, S, 2)
        b = p1[None, :, :] - q_chunk

        numer = a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 0]
        denom = a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]
        theta = np.arctan2(numer, denom)
        w[i : i + chunk_size] = np.sum(theta, axis=-1) / (2.0 * np.pi)

    return w


def point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from each point in points (N, D) to polyline (M, D)."""
    p0 = polyline[:-1]
    p1 = polyline[1:]
    v = p1 - p0
    v_len_sq = np.sum(v**2, axis=1)
    v_len_sq = np.maximum(v_len_sq, 1e-14)

    w = points[:, None, :] - p0[None, :, :]
    t = np.sum(w * v[None, :, :], axis=2) / v_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = p0[None, :, :] + t[:, :, None] * v[None, :, :]
    dists = np.linalg.norm(points[:, None, :] - proj, axis=2)
    return np.min(dists, axis=1)


def polyline_hausdorff(c1: np.ndarray, c2: np.ndarray) -> float:
    """Compute symmetric Hausdorff distance between two 3D polylines."""
    d1 = np.max(point_to_polyline_distance(c1, c2))
    d2 = np.max(point_to_polyline_distance(c2, c1))
    return float(max(d1, d2))


# -----------------------------------------------------------------------------
# Higher-Order Derivatives Helper Operations
# -----------------------------------------------------------------------------
class ParametricDerivativeEvalOp(_CustomExplicitOperation):
    """
    Evaluates B-spline value/derivatives or their adjoints for a single patch.
    Recursively supports arbitrary-order VJPs w.r.t parametric coordinate u.
    """

    def __init__(
        self,
        function,
        patch_idx: int,
        order: tuple[int, ...],
        transpose: bool = False,
    ):
        super().__init__()
        self.function = function
        self.patch_idx = patch_idx
        self.order = tuple(int(o) for o in order)
        self.transpose = transpose
        self.num_param_dims = len(self.order)

    def evaluate(self, u_var: csdl.Variable, linear_var: csdl.Variable) -> csdl.Variable:
        self.declare_input("u", u_var)
        self.declare_input("linear_arg", linear_var)

        num_pts = u_var.shape[0]
        phys_dim = linear_var.shape[-1]

        if not self.transpose:
            # Output is S_alpha = B^(order)(u) @ C, shape (N_p, D)
            out = self.create_output("S_alpha", shape=(num_pts, phys_dim))
        else:
            # Output is B^(order)(u)^T @ cotangent_S, shape (N_c, D)
            num_cp = int(np.prod(self.function.coefficients.shape[:-1]))
            out = self.create_output("B_trans_S", shape=(num_cp, phys_dim))

        self.declare_vjp_function(
            ParametricDerivativeEvalOpVJP,
            function=self.function,
            patch_idx=self.patch_idx,
            order=self.order,
            transpose=self.transpose,
        )
        return out

    def compute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
        u = np.asarray(inputs["u"], dtype=np.float64)
        linear_arg = np.asarray(inputs["linear_arg"], dtype=np.float64)

        B = self.function.space.compute_basis_matrix(u, parametric_derivative_orders=self.order)
        if not self.transpose:
            outputs["S_alpha"] = B @ linear_arg.reshape(-1, linear_arg.shape[-1])
        else:
            outputs["B_trans_S"] = B.T @ linear_arg


class ParametricDerivativeEvalOpVJP(_CustomExplicitOperation):
    def __init__(
        self,
        function,
        patch_idx: int,
        order: tuple[int, ...],
        transpose: bool,
    ):
        super().__init__()
        self.function = function
        self.patch_idx = patch_idx
        self.order = order
        self.transpose = transpose
        self.degree = function.space.degree

    def evaluate(self, inputs: dict[str, csdl.Variable], d_outputs: dict[str, csdl.Variable]) -> dict[str, csdl.Variable]:
        u = inputs["u"]
        linear_arg = inputs["linear_arg"]
        out_name = "B_trans_S" if self.transpose else "S_alpha"
        d_out = d_outputs[out_name]

        self.declare_input("u", u)
        self.declare_input("linear_arg", linear_arg)
        self.declare_input("d_out", d_out)

        d_u = self.create_output("d_u", shape=u.shape)
        d_linear_arg = self.create_output("d_linear_arg", shape=linear_arg.shape)

        return {"u": d_u, "linear_arg": d_linear_arg}

    def compute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
        u = np.asarray(inputs["u"], dtype=np.float64)
        linear_arg = np.asarray(inputs["linear_arg"], dtype=np.float64)
        d_out = np.asarray(inputs["d_out"], dtype=np.float64)

        num_param_dims = len(self.order)

        # Adjoint w.r.t linear argument is B^(order)(u)^T @ d_out or B^(order)(u) @ d_out
        B = self.function.space.compute_basis_matrix(u, parametric_derivative_orders=self.order)
        if not self.transpose:
            outputs["d_linear_arg"] = (B.T @ d_out).reshape(linear_arg.shape)
        else:
            outputs["d_linear_arg"] = (B @ d_out).reshape(linear_arg.shape)

        # Adjoint w.r.t u: sum over coordinate axes j of (order + e_j)
        d_u = np.zeros_like(u)
        if not self.transpose:
            C_flat = linear_arg.reshape(-1, linear_arg.shape[-1])
            for j in range(num_param_dims):
                if self.order[j] + 1 <= self.degree[j]:
                    new_order = list(self.order)
                    new_order[j] += 1
                    B_j = self.function.space.compute_basis_matrix(u, parametric_derivative_orders=tuple(new_order))
                    S_j = B_j @ C_flat  # (N_p, D)
                    d_u[:, j] = np.sum(d_out * S_j, axis=-1)
        else:
            cot_S = linear_arg
            for j in range(num_param_dims):
                if self.order[j] + 1 <= self.degree[j]:
                    new_order = list(self.order)
                    new_order[j] += 1
                    B_j = self.function.space.compute_basis_matrix(u, parametric_derivative_orders=tuple(new_order))
                    B_j_cot = B_j @ d_out  # (N_p, D)
                    d_u[:, j] = np.sum(cot_S * B_j_cot, axis=-1)

        outputs["d_u"] = d_u


class ActivePatchProjectionOp(_CustomExplicitOperation):
    """
    Operation representing the optimal parametric coordinates u^* on active patches.
    Forward compute() outputs cached u^* with zero overhead during 1st-order runs.
    evaluate_vjp() differentiates the reduced KKT system when 2nd/3rd derivatives are queried.
    """

    def __init__(
        self,
        function_set,
        u_star_cache: np.ndarray,
        patch_idx_cache: np.ndarray,
        bound_tol: float = 1e-7,
        eps_hess: float = 1e-8,
        eps_lm: float = 1e-8,
    ):
        super().__init__()
        self.function_set = function_set
        self.u_star_cache = np.asarray(u_star_cache, dtype=np.float64)
        self.patch_idx_cache = np.asarray(patch_idx_cache, dtype=np.int32)
        self.bound_tol = bound_tol
        self.eps_hess = eps_hess
        self.eps_lm = eps_lm

    def evaluate(self, coefficients: csdl.Variable, points: csdl.Variable) -> csdl.Variable:
        self.declare_input("coefficients", coefficients)
        self.declare_input("points", points)

        u_star = self.create_output("u_star", shape=self.u_star_cache.shape)
        self.declare_vjp_function(
            ActivePatchProjectionOpVJP,
            function_set=self.function_set,
            u_star_cache=self.u_star_cache,
            patch_idx_cache=self.patch_idx_cache,
            bound_tol=self.bound_tol,
            eps_hess=self.eps_hess,
            eps_lm=self.eps_lm,
        )
        return u_star

    def compute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
        outputs["u_star"] = self.u_star_cache


class ActivePatchProjectionOpVJP(_CustomExplicitOperation):
    def __init__(
        self,
        function_set,
        u_star_cache: np.ndarray,
        patch_idx_cache: np.ndarray,
        bound_tol: float = 1e-7,
        eps_hess: float = 1e-8,
        eps_lm: float = 1e-8,
    ):
        super().__init__()
        self.function_set = function_set
        self.u_star_cache = u_star_cache
        self.patch_idx_cache = patch_idx_cache
        self.bound_tol = bound_tol
        self.eps_hess = eps_hess
        self.eps_lm = eps_lm

    def evaluate(self, inputs: dict[str, csdl.Variable], d_outputs: dict[str, csdl.Variable]) -> dict[str, csdl.Variable]:
        coefficients = inputs["coefficients"]
        points = inputs["points"]
        d_u_star = d_outputs["u_star"]

        self.declare_input("coefficients", coefficients)
        self.declare_input("points", points)
        self.declare_input("d_u_star", d_u_star)

        d_coefficients = self.create_output("d_coefficients", shape=coefficients.shape)
        d_points = self.create_output("d_points", shape=points.shape)

        return {"coefficients": d_coefficients, "points": d_points}

    def compute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
        coeffs = np.asarray(inputs["coefficients"], dtype=np.float64)
        points = np.asarray(inputs["points"], dtype=np.float64)
        d_u_star = np.asarray(inputs["d_u_star"], dtype=np.float64)

        num_pts = len(points)
        phys_dim = points.shape[-1]
        param_dim = self.u_star_cache.shape[-1]

        d_points = np.zeros_like(points)
        d_coeffs = np.zeros_like(coeffs)

        funcs = list(self.function_set.functions.values())
        offset = 0
        slices = []
        for fn in funcs:
            nc = int(np.prod(fn.coefficients.shape[:-1]))
            slices.append((offset, offset + nc, fn.coefficients.shape))
            offset += nc

        for i in range(num_pts):
            p_idx = self.patch_idx_cache[i]
            fn = funcs[p_idx]
            u_pt = self.u_star_cache[i : i + 1]
            c_start, c_end, c_shape = slices[p_idx]
            c_patch = coeffs[c_start:c_end]

            # Evaluate 0th, 1st, and 2nd derivatives
            B0 = fn.space.compute_basis_matrix(u_pt)
            S0 = (B0 @ c_patch)[0]
            r = S0 - points[i]

            # 1st derivatives: dS/du_j
            dS = []
            for j in range(param_dim):
                ord_1 = tuple(int(k == j) for k in range(param_dim))
                B1 = fn.space.compute_basis_matrix(u_pt, parametric_derivative_orders=ord_1)
                dS.append((B1 @ c_patch)[0])

            # 2nd derivatives: d2S / du_j du_k
            H_proj = np.zeros((param_dim, param_dim))
            for j in range(param_dim):
                for k in range(param_dim):
                    ord_2 = [0] * param_dim
                    ord_2[j] += 1
                    ord_2[k] += 1
                    B2 = fn.space.compute_basis_matrix(u_pt, parametric_derivative_orders=tuple(ord_2))
                    S2 = (B2 @ c_patch)[0]
                    H_proj[j, k] = np.dot(dS[j], dS[k]) + np.dot(r, S2)

            # Active-bound KKT mask
            active_mask = np.ones(param_dim, dtype=bool)
            for j in range(param_dim):
                u_val = u_pt[0, j]
                grad_f = np.dot(r, dS[j])
                if (u_val <= self.bound_tol and grad_f > 0.0) or (u_val >= 1.0 - self.bound_tol and grad_f < 0.0):
                    active_mask[j] = False

            # Damped reduced system
            free_idx = np.where(active_mask)[0]
            if len(free_idx) == 0:
                continue

            H_R = H_proj[np.ix_(free_idx, free_idx)] + self.eps_lm * np.eye(len(free_idx))

            # Cramer's rule inversion with C-infinity smooth reciprocal
            if len(free_idx) == 1:
                det = H_R[0, 0]
                inv_det = det / (det**2 + self.eps_hess**2)
                inv_H = np.array([[inv_det]])
            elif len(free_idx) == 2:
                det = H_R[0, 0] * H_R[1, 1] - H_R[0, 1] * H_R[1, 0]
                inv_det = det / (det**2 + self.eps_hess**2)
                inv_H = inv_det * np.array([
                    [H_R[1, 1], -H_R[0, 1]],
                    [-H_R[1, 0], H_R[0, 0]],
                ])
            else:
                inv_H = np.linalg.pinv(H_R)

            cot_u = d_u_star[i, free_idx]
            v = inv_H @ cot_u  # shape (n_free,)

            # d_points contribution: dF/dp = -dS/du
            for idx_k, j in enumerate(free_idx):
                d_points[i] += v[idx_k] * dS[j]

            # d_coeffs contribution: dF/dC = B_j^T r + B_0^T dS_j
            for idx_k, j in enumerate(free_idx):
                ord_1 = tuple(int(k == j) for k in range(param_dim))
                B1 = fn.space.compute_basis_matrix(u_pt, parametric_derivative_orders=ord_1)
                term = (B1.T @ r.reshape(1, -1) + B0.T @ dS[j].reshape(1, -1))
                d_coeffs[c_start:c_end] -= v[idx_k] * term

        outputs["d_points"] = d_points
        outputs["d_coefficients"] = d_coeffs


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Second-Derivative VJP (Hessian-Vector Product) Operation
# -----------------------------------------------------------------------------
class SignedDistanceSecondDerivVJP(_CustomExplicitOperation):
    """
    Second-derivative VJP (Hessian-vector product) for Signed Distance Field operation.
    Computes exact Hvps w.r.t query points and geometry coefficients on smooth branches.
    """

    def __init__(
        self,
        function_set,
        sign_mode: str,
        eps_reg: float,
        bound_tol: float,
        eps_hess: float,
        eps_lm: float,
        cache_data: dict,
    ):
        super().__init__()
        self.function_set = function_set
        self.sign_mode = sign_mode
        self.eps_reg = eps_reg
        self.bound_tol = bound_tol
        self.eps_hess = eps_hess
        self.eps_lm = eps_lm
        self.cache_data = cache_data

    def evaluate(self, inputs: dict[str, csdl.Variable], d_outputs: dict[str, csdl.Variable]) -> dict[str, csdl.Variable]:
        coefficients = inputs["coefficients"]
        points = inputs["points"]
        d_signed_distance = inputs["d_signed_distance"]

        cot_d_coeffs = d_outputs.get("d_coefficients", csdl.Variable(value=np.zeros(coefficients.shape)))
        cot_d_points = d_outputs.get("d_points", csdl.Variable(value=np.zeros(points.shape)))

        self.declare_input("coefficients", coefficients)
        self.declare_input("points", points)
        self.declare_input("d_signed_distance", d_signed_distance)
        self.declare_input("cot_d_coeffs", cot_d_coeffs)
        self.declare_input("cot_d_points", cot_d_points)

        hvp_coeffs = self.create_output("hvp_coeffs", shape=coefficients.shape)
        hvp_points = self.create_output("hvp_points", shape=points.shape)
        hvp_d_signed_dist = self.create_output("hvp_d_signed_dist", shape=d_signed_distance.shape)

        return {
            "coefficients": hvp_coeffs,
            "points": hvp_points,
            "d_signed_distance": hvp_d_signed_dist,
        }

    def compute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
        coeffs = np.asarray(inputs["coefficients"], dtype=np.float64)
        points = np.asarray(inputs["points"], dtype=np.float64)
        d_phi = np.asarray(inputs["d_signed_distance"], dtype=np.float64).reshape(-1)
        cot_C = np.asarray(inputs["cot_d_coeffs"], dtype=np.float64)
        cot_p = np.asarray(inputs["cot_d_points"], dtype=np.float64)

        num_pts = len(points)
        phys_dim = points.shape[-1]
        param_dim = self.cache_data.get("u_star", np.zeros((num_pts, phys_dim - 1))).shape[-1]

        signs = self.cache_data.get("signs", np.ones(num_pts))
        footpoints = self.cache_data.get("footpoints", points)
        patch_indices = self.cache_data.get("patch_idx", np.zeros(num_pts, dtype=int))
        u_stars = self.cache_data.get("u_star", np.zeros((num_pts, param_dim)))

        hvp_p = np.zeros_like(points)
        hvp_C = np.zeros_like(coeffs)
        hvp_d_phi = np.zeros_like(d_phi)

        funcs = list(self.function_set.functions.values())
        offset = 0
        slices = []
        for fn in funcs:
            nc = int(np.prod(fn.coefficients.shape[:-1]))
            slices.append((offset, offset + nc, fn.coefficients.shape))
            offset += nc

        for i in range(num_pts):
            p_idx = patch_indices[i]
            fn = funcs[p_idx]
            u_pt = u_stars[i : i + 1]
            c_start, c_end, c_shape = slices[p_idx]
            c_patch = coeffs[c_start:c_end].reshape(-1, phys_dim)
            cot_C_patch = cot_C[c_start:c_end].reshape(-1, phys_dim)
            p_i = points[i]
            d_phi_i = d_phi[i]
            sign_i = signs[i]

            B0 = fn.space.compute_basis_matrix(u_pt)
            x_star = (B0 @ c_patch)[0]
            diff = p_i - x_star
            r_sq = np.sum(diff**2) + self.eps_reg**2
            r = np.sqrt(r_sq)
            r_hat = diff / r

            # Cotangent on g_p from cot_p and cot_C
            cot_gp = cot_p[i] - (B0 @ cot_C_patch)[0]
            hvp_d_phi[i] = sign_i * np.dot(r_hat, cot_gp)

            # Parametric derivatives (1st and 2nd)
            dS = []
            for j in range(param_dim):
                ord_1 = tuple(int(k == j) for k in range(param_dim))
                B1 = fn.space.compute_basis_matrix(u_pt, parametric_derivative_orders=ord_1)
                dS.append((B1 @ c_patch)[0])

            H_proj = np.zeros((param_dim, param_dim))
            for j in range(param_dim):
                for k in range(param_dim):
                    ord_2 = [0] * param_dim
                    ord_2[j] += 1
                    ord_2[k] += 1
                    B2 = fn.space.compute_basis_matrix(u_pt, parametric_derivative_orders=tuple(ord_2))
                    S2 = (B2 @ c_patch)[0]
                    H_proj[j, k] = np.dot(dS[j], dS[k]) + np.dot(x_star - p_i, S2)

            # Active-bound KKT mask
            active_mask = np.ones(param_dim, dtype=bool)
            for j in range(param_dim):
                u_val = u_pt[0, j]
                grad_f = np.dot(x_star - p_i, dS[j])
                if (u_val <= self.bound_tol and grad_f > 0.0) or (u_val >= 1.0 - self.bound_tol and grad_f < 0.0):
                    active_mask[j] = False

            free_idx = np.where(active_mask)[0]

            if len(free_idx) == 0:
                inv_H = np.zeros((0, 0))
            elif len(free_idx) == 1:
                H_R = H_proj[free_idx[0], free_idx[0]] + self.eps_lm
                inv_det = H_R / (H_R**2 + self.eps_hess**2)
                inv_H = np.array([[inv_det]])
            elif len(free_idx) == 2:
                H_R = H_proj[np.ix_(free_idx, free_idx)] + self.eps_lm * np.eye(2)
                det = H_R[0, 0] * H_R[1, 1] - H_R[0, 1] * H_R[1, 0]
                inv_det = det / (det**2 + self.eps_hess**2)
                inv_H = inv_det * np.array([
                    [H_R[1, 1], -H_R[0, 1]],
                    [-H_R[1, 0], H_R[0, 0]],
                ])
            else:
                H_R = H_proj[np.ix_(free_idx, free_idx)] + self.eps_lm * np.eye(len(free_idx))
                inv_H = np.linalg.pinv(H_R)

            P_perp = (np.eye(phys_dim) - np.outer(r_hat, r_hat)) / r
            w = sign_i * d_phi_i * (P_perp @ cot_gp)

            if len(free_idx) > 0:
                Js_free = np.column_stack([dS[j] for j in free_idx])
                P_proj = Js_free @ inv_H @ Js_free.T
                hvp_p[i] = w - sign_i * d_phi_i * (P_proj @ cot_gp) / r
                du_free = inv_H @ (Js_free.T @ cot_gp) * (sign_i * d_phi_i / r)
            else:
                hvp_p[i] = w
                du_free = np.zeros(0)

            hvp_C[c_start:c_end] -= B0.T @ hvp_p[i].reshape(1, -1)
            for idx_k, j in enumerate(free_idx):
                ord_1 = tuple(int(k == j) for k in range(param_dim))
                B1 = fn.space.compute_basis_matrix(u_pt, parametric_derivative_orders=ord_1)
                term = (B1.T @ (x_star - p_i).reshape(1, -1) + B0.T @ dS[j].reshape(1, -1))
                hvp_C[c_start:c_end] -= du_free[idx_k] * term

        outputs["hvp_coeffs"] = hvp_C
        outputs["hvp_points"] = hvp_p
        outputs["hvp_d_signed_dist"] = hvp_d_phi


# -----------------------------------------------------------------------------
# Main CSDL Custom Operation: FunctionSetSignedDistanceOperation
# -----------------------------------------------------------------------------
class FunctionSetSignedDistanceOperationVJP(_CustomExplicitOperation):
    """VJP for FunctionSetSignedDistanceOperation implementing envelope theorem exact gradients."""

    def __init__(
        self,
        function_set,
        sign_mode: str,
        eps_reg: float,
        bound_tol: float,
        eps_hess: float,
        eps_lm: float,
        cache_data: dict,
    ):
        super().__init__()
        self.function_set = function_set
        self.sign_mode = sign_mode
        self.eps_reg = eps_reg
        self.bound_tol = bound_tol
        self.eps_hess = eps_hess
        self.eps_lm = eps_lm
        self.cache_data = cache_data

    def evaluate(self, inputs: dict[str, csdl.Variable], d_outputs: dict[str, csdl.Variable]) -> dict[str, csdl.Variable]:
        coefficients = inputs["coefficients"]
        points = inputs["points"]
        d_signed_distance = d_outputs["signed_distance"]

        self.declare_input("coefficients", coefficients)
        self.declare_input("points", points)
        self.declare_input("d_signed_distance", d_signed_distance)

        d_coefficients = self.create_output("d_coefficients", shape=coefficients.shape)
        d_points = self.create_output("d_points", shape=points.shape)

        self.declare_vjp_function(
            SignedDistanceSecondDerivVJP,
            function_set=self.function_set,
            sign_mode=self.sign_mode,
            eps_reg=self.eps_reg,
            bound_tol=self.bound_tol,
            eps_hess=self.eps_hess,
            eps_lm=self.eps_lm,
            cache_data=self.cache_data,
        )

        return {"coefficients": d_coefficients, "points": d_points}

    def compute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
        coeffs = np.asarray(inputs["coefficients"], dtype=np.float64)
        points = np.asarray(inputs["points"], dtype=np.float64)
        d_phi = np.asarray(inputs["d_signed_distance"], dtype=np.float64).reshape(-1)

        signs = self.cache_data.get("signs", np.ones(len(points)))
        footpoints = self.cache_data.get("footpoints", points)
        patch_indices = self.cache_data.get("patch_idx", np.zeros(len(points), dtype=int))
        u_stars = self.cache_data.get("u_star", np.zeros((len(points), 2)))
        normals = self.cache_data.get("normals", np.zeros_like(points))

        # r = sqrt(||p - x*||^2 + eps_reg^2)
        diff = points - footpoints
        r_unreg = np.linalg.norm(diff, axis=-1)
        r = np.sqrt(r_unreg**2 + self.eps_reg**2)

        # Gradient direction g_hat = diff / r (or outward normal n_hat if r_unreg <= 1e-12)
        on_surface_mask = r_unreg <= 1e-12
        g_hat = diff / r[:, None]
        if np.any(on_surface_mask):
            g_hat[on_surface_mask] = normals[on_surface_mask]

        # dphi/dp = sign * g_hat
        d_points = (d_phi * signs)[:, None] * g_hat
        outputs["d_points"] = d_points

        # dphi/dC = -B(u*)^T @ (dphi/dp)
        d_coeffs = np.zeros_like(coeffs)
        funcs = list(self.function_set.functions.values())
        offset = 0
        slices = []
        for fn in funcs:
            nc = int(np.prod(fn.coefficients.shape[:-1]))
            slices.append((offset, offset + nc))
            offset += nc

        for p_idx, fn in enumerate(funcs):
            pts_in_patch = np.where(patch_indices == p_idx)[0]
            if len(pts_in_patch) == 0:
                continue
            u_patch = u_stars[pts_in_patch]
            B_patch = fn.space.compute_basis_matrix(u_patch)
            d_p_patch = d_points[pts_in_patch]
            c_start, c_end = slices[p_idx]
            d_coeffs[c_start:c_end] = - (B_patch.T @ d_p_patch)

        outputs["d_coefficients"] = d_coeffs


class FunctionSetSignedDistanceOperation(_CustomExplicitOperation):
    """
    Evaluates signed distance from query points to a closed B-spline loop (2D) or shell (3D).
    """

    def __init__(
        self,
        function_set,
        sign_mode: Literal["winding_number", "unsigned"] = "winding_number",
        winding_grid_n: int = 25,
        winding_chord_tolerance: float = 1e-3,
        winding_threshold: float = 0.5,
        eps_reg: float = 1e-14,
        bound_tol: float = 1e-7,
        eps_hess: float = 1e-8,
        eps_lm: float = 1e-8,
        projection_options: dict | None = None,
    ):
        super().__init__()
        self.function_set = function_set
        self.sign_mode = sign_mode
        self.winding_grid_n = winding_grid_n
        self.winding_chord_tolerance = winding_chord_tolerance
        self.winding_threshold = winding_threshold
        self.eps_reg = eps_reg
        self.bound_tol = bound_tol
        self.eps_hess = eps_hess
        self.eps_lm = eps_lm
        self.projection_options = projection_options or {}

        # 1. Validate function set dimensions and entity types
        funcs = list(function_set.functions.values())
        if not funcs:
            raise ValueError("FunctionSet cannot be empty.")

        self.num_param_dims = funcs[0].space.num_parametric_dimensions
        self.num_phys_dims = funcs[0].coefficients.shape[-1]

        if self.num_phys_dims not in (2, 3):
            raise ValueError(f"Signed distance requires physical dimension 2 or 3, got {self.num_phys_dims}.")
        if self.num_param_dims != self.num_phys_dims - 1:
            raise ValueError(
                f"Signed distance requires codimension-1 geometry (got param_dim={self.num_param_dims}, phys_dim={self.num_phys_dims})."
            )

        for fn in funcs:
            if fn.space.num_parametric_dimensions != self.num_param_dims:
                raise ValueError("Mixed-dimensional function sets are not supported.")
            if fn.coefficients.shape[-1] != self.num_phys_dims:
                raise ValueError("Inconsistent physical dimensions in function set.")

        # 2. Build detached numerical copy and compute global coefficient slices
        detached_functions = {}
        self.patch_slices = []
        offset = 0
        for i, fn in function_set.functions.items():
            c_val = fn.coefficients.value if hasattr(fn.coefficients, "value") else fn.coefficients
            new_fn = lfs.Function.__new__(lfs.Function)
            new_fn.space = fn.space
            new_fn.name = fn.name
            new_fn.triangulation = None
            new_fn.coefficients = _CoeffArray(c_val)
            new_fn.num_physical_dimensions = fn.coefficients.shape[-1]
            detached_functions[i] = new_fn

            nc = int(np.prod(fn.coefficients.shape[:-1]))
            self.patch_slices.append((offset, offset + nc, fn.coefficients.shape))
            offset += nc

        self.detached_function_set = lfs.FunctionSet.__new__(lfs.FunctionSet)
        self.detached_function_set.functions = detached_functions
        self.detached_function_set.function_names = list(detached_functions.keys())
        self.detached_function_set.space = function_set.space
        self.detached_function_set.name = "detached_geometry"
        self.total_control_points = offset

        # 3. Cache private evaluation state
        self._cache_data = {}

        # 4. Topology validation, orientation calibration, and static tessellation setup
        self._initialize_topology_and_tessellation()

    def _initialize_topology_and_tessellation(self):
        """Build static tessellation basis matrices and calibrate outward orientation."""
        funcs = list(self.detached_function_set.functions.values())
        num_patches = len(funcs)

        # Build structured (N_u, N_v) grids and basis matrices per patch
        blocks = []
        all_elements = []
        self.patch_grid_shapes = []
        vert_offset = 0

        for p_idx, fn in enumerate(funcs):
            if self.num_phys_dims == 2:
                nu = max(15, self.winding_grid_n)
                u = np.linspace(0.0, 1.0, nu)
                B = fn.space.compute_basis_matrix(u[:, None])
                blocks.append(B)
                self.patch_grid_shapes.append((nu,))

                segs = np.column_stack([
                    np.arange(vert_offset, vert_offset + nu - 1),
                    np.arange(vert_offset + 1, vert_offset + nu),
                ])
                all_elements.append(segs)
                vert_offset += nu

            else:  # 3D
                nu = max(15, self.winding_grid_n)
                nv = max(15, self.winding_grid_n)
                u = np.linspace(0.0, 1.0, nu)
                v = np.linspace(0.0, 1.0, nv)
                uu, vv = np.meshgrid(u, v, indexing="ij")
                uv = np.column_stack([uu.ravel(), vv.ravel()])
                B = fn.space.compute_basis_matrix(uv)
                blocks.append(B)
                self.patch_grid_shapes.append((nu, nv))

                tris = []
                for i in range(nu - 1):
                    for j in range(nv - 1):
                        idx00 = vert_offset + i * nv + j
                        idx10 = vert_offset + (i + 1) * nv + j
                        idx11 = vert_offset + (i + 1) * nv + (j + 1)
                        idx01 = vert_offset + i * nv + (j + 1)
                        tris.append([idx00, idx10, idx11])
                        tris.append([idx00, idx11, idx01])
                all_elements.append(np.array(tris))
                vert_offset += (nu * nv)

        self.B_tess = sp.block_diag(blocks, format="csr")
        self.elements = np.vstack(all_elements) if all_elements else np.zeros((0, self.num_phys_dims), dtype=int)

        # Baseline orientation calibration via signed volume / area
        baseline_coeffs = np.vstack([fn.coefficients.value.reshape(-1, self.num_phys_dims) for fn in funcs])
        v_base = self.B_tess @ baseline_coeffs

        if self.sign_mode == "winding_number":
            self._validate_watertight_and_calibrate_orientation(funcs, v_base)
        else:
            self.patch_signs = np.ones(num_patches)

    def _validate_watertight_and_calibrate_orientation(self, funcs, v_base: np.ndarray):
        """Validate watertight 2-manifold shell topology and orient outwards."""
        num_patches = len(funcs)

        if self.num_phys_dims == 3:
            # 1. Match boundary curves spatially
            def extract_dense_boundary(fn, num=50):
                s = np.linspace(0.0, 1.0, num)
                z = np.zeros(num)
                o = np.ones(num)
                return [
                    fn.evaluate(np.column_stack([s, z]), non_csdl=True),
                    fn.evaluate(np.column_stack([o, s]), non_csdl=True),
                    fn.evaluate(np.column_stack([s, o]), non_csdl=True),
                    fn.evaluate(np.column_stack([z, s]), non_csdl=True),
                ]

            boundaries = {k: extract_dense_boundary(fn) for k, fn in enumerate(funcs)}
            bbox_diag = float(np.linalg.norm(np.max(v_base, axis=0) - np.min(v_base, axis=0)))
            tol = max(1e-3, 5e-3 * bbox_diag)
            matches = {}
            for k1 in range(num_patches):
                for e1 in range(4):
                    c1 = boundaries[k1][e1]
                    found = []
                    for k2 in range(num_patches):
                        if k1 == k2:
                            continue
                        for e2 in range(4):
                            c2 = boundaries[k2][e2]
                            h = polyline_hausdorff(c1, c2)
                            if h < tol:
                                found.append((k2, e2))
                    matches[(k1, e1)] = found

            # Verify watertight manifold: every boundary edge has at least 1 match
            unmatched = [(k, e) for (k, e), m in matches.items() if len(m) == 0]
            if unmatched:
                raise ValueError(
                    f"Geometry is open or non-manifold in winding_number mode: patch edges {unmatched} have no partner."
                )

            # Check signed volume of baseline tessellation
            p0 = v_base[self.elements[:, 0]]
            p1 = v_base[self.elements[:, 1]]
            p2 = v_base[self.elements[:, 2]]
            raw_vol = np.sum(np.sum(p0 * np.cross(p1, p2), axis=1)) / 6.0

            if raw_vol < 0.0:
                # Flip triangle windings so surface normal points outward
                self.elements = self.elements[:, [0, 2, 1]]
                self.patch_signs = -np.ones(num_patches)
            else:
                self.patch_signs = np.ones(num_patches)

        else:  # 2D
            p0 = v_base[self.elements[:, 0]]
            p1 = v_base[self.elements[:, 1]]
            raw_area = 0.5 * np.sum(p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0])
            if raw_area < 0.0:
                self.elements = self.elements[:, [1, 0]]
                self.patch_signs = -np.ones(num_patches)
            else:
                self.patch_signs = np.ones(num_patches)

    def evaluate(self, coefficients: csdl.Variable, points: csdl.Variable) -> csdl.Variable:
        self.declare_input("coefficients", coefficients)
        self.declare_input("points", points)

        num_pts = points.shape[0]
        signed_dist = self.create_output("signed_distance", shape=(num_pts,))

        self.declare_vjp_function(
            FunctionSetSignedDistanceOperationVJP,
            function_set=self.detached_function_set,
            sign_mode=self.sign_mode,
            eps_reg=self.eps_reg,
            bound_tol=self.bound_tol,
            eps_hess=self.eps_hess,
            eps_lm=self.eps_lm,
            cache_data=self._cache_data,
        )
        return signed_dist

    def compute(self, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]):
        coeffs = np.asarray(inputs["coefficients"], dtype=np.float64)
        points = np.asarray(inputs["points"], dtype=np.float64)
        num_pts = len(points)

        # 1. Update detached function set coefficients and clear stale projection caches
        funcs = list(self.detached_function_set.functions.values())
        for p_idx, fn in enumerate(funcs):
            start, end, shape = self.patch_slices[p_idx]
            val = coeffs[start:end].reshape(shape)
            fn.coefficients = _CoeffArray(val)
            if hasattr(fn, "_grid_searches"):
                fn._grid_searches = {}

        # 2. Update tessellation vertices: V = B_tess @ coeffs
        v_tess = self.B_tess @ coeffs.reshape(-1, self.num_phys_dims)

        # 3. Deterministic native LFS closest-point projection
        proj_res = self.detached_function_set.project(
            points,
            num_workers=1,
            use_line_search=True,
            max_newton_iterations=100,
            newton_tolerance=1e-10,
            do_pickles=False,
        )

        patch_indices = np.zeros(num_pts, dtype=int)
        u_stars = np.zeros((num_pts, self.num_param_dims), dtype=np.float64)
        for i, (p_idx, u_coord) in enumerate(proj_res):
            patch_indices[i] = p_idx
            u_stars[i] = np.asarray(u_coord, dtype=np.float64).reshape(-1)

        # 4. Evaluate exact B-spline footpoints and facet footpoints
        footpoints_bspline = np.zeros((num_pts, self.num_phys_dims))
        footpoints_tess = np.zeros((num_pts, self.num_phys_dims))
        normals = np.zeros((num_pts, self.num_phys_dims))

        for p_idx, fn in enumerate(funcs):
            pts_in_patch = np.where(patch_indices == p_idx)[0]
            if len(pts_in_patch) == 0:
                continue

            u_patch = u_stars[pts_in_patch]
            val = fn.evaluate(u_patch, non_csdl=True)
            footpoints_bspline[pts_in_patch] = val

            # Bilinear facet footpoint evaluation
            if self.num_phys_dims == 3:
                nu, nv = self.patch_grid_shapes[p_idx]
                u_clamped = np.clip(u_patch[:, 0], 0.0, 1.0) * (nu - 1)
                v_clamped = np.clip(u_patch[:, 1], 0.0, 1.0) * (nv - 1)
                i_cell = np.clip(np.floor(u_clamped).astype(int), 0, nu - 2)
                j_cell = np.clip(np.floor(v_clamped).astype(int), 0, nv - 2)
                s = u_clamped - i_cell
                t = v_clamped - j_cell

                c_start, c_end, c_shape = self.patch_slices[p_idx]
                v_patch = v_tess[self.patch_slices[p_idx][0] : self.patch_slices[p_idx][1]]
                # Re-index patch vertices
                patch_base_idx = sum(np.prod(s) for s in self.patch_grid_shapes[:p_idx])
                v00 = v_tess[patch_base_idx + i_cell * nv + j_cell]
                v10 = v_tess[patch_base_idx + (i_cell + 1) * nv + j_cell]
                v01 = v_tess[patch_base_idx + i_cell * nv + (j_cell + 1)]
                v11 = v_tess[patch_base_idx + (i_cell + 1) * nv + (j_cell + 1)]

                footpoints_tess[pts_in_patch] = (
                    (1.0 - s)[:, None] * (1.0 - t)[:, None] * v00
                    + s[:, None] * (1.0 - t)[:, None] * v10
                    + (1.0 - s)[:, None] * t[:, None] * v01
                    + s[:, None] * t[:, None] * v11
                )

                # Calibrated outward normal
                B_u = fn.space.compute_basis_matrix(u_patch, parametric_derivative_orders=(1, 0))
                B_v = fn.space.compute_basis_matrix(u_patch, parametric_derivative_orders=(0, 1))
                c_curr = coeffs[c_start:c_end]
                Su = B_u @ c_curr
                Sv = B_v @ c_curr
                cross_n = np.cross(Su, Sv)
                n_len = np.maximum(np.linalg.norm(cross_n, axis=1, keepdims=True), 1e-14)
                normals[pts_in_patch] = self.patch_signs[p_idx] * (cross_n / n_len)

            else:  # 2D
                nu = self.patch_grid_shapes[p_idx][0]
                u_clamped = np.clip(u_patch[:, 0], 0.0, 1.0) * (nu - 1)
                i_cell = np.clip(np.floor(u_clamped).astype(int), 0, nu - 2)
                s = u_clamped - i_cell
                patch_base_idx = sum(s[0] for s in self.patch_grid_shapes[:p_idx])
                v0 = v_tess[patch_base_idx + i_cell]
                v1 = v_tess[patch_base_idx + i_cell + 1]
                footpoints_tess[pts_in_patch] = (1.0 - s)[:, None] * v0 + s[:, None] * v1

                B_u = fn.space.compute_basis_matrix(u_patch, parametric_derivative_orders=(1,))
                c_start, c_end, _ = self.patch_slices[p_idx]
                c_curr = coeffs[c_start:c_end]
                Su = B_u @ c_curr
                # 2D outward normal
                n_rot = np.column_stack([Su[:, 1], -Su[:, 0]])
                n_len = np.maximum(np.linalg.norm(n_rot, axis=1, keepdims=True), 1e-14)
                normals[pts_in_patch] = self.patch_signs[p_idx] * (n_rot / n_len)

        # 5. Compute unsigned distance: r = sqrt(||p - x*||^2 + eps_reg^2)
        diff = points - footpoints_bspline
        r = np.sqrt(np.sum(diff**2, axis=-1) + self.eps_reg**2)

        # 6. Evaluate inside/outside sign
        if self.sign_mode == "unsigned":
            signs = np.ones(num_pts)
        else:
            # Continuous chord-shifted query point: p_tilde = p - (x*_bspline - x*_tess)
            p_tilde = points - (footpoints_bspline - footpoints_tess)
            if self.num_phys_dims == 3:
                w = compute_solid_angles_3d(p_tilde, v_tess, self.elements)
            else:
                w = compute_winding_numbers_2d(p_tilde, v_tess, self.elements)
            signs = np.where(w > self.winding_threshold, -1.0, 1.0)

        signed_distance = signs * r
        outputs["signed_distance"] = signed_distance

        # 7. Update private caches for VJP
        self._cache_data["footpoints"] = footpoints_bspline
        self._cache_data["patch_idx"] = patch_indices
        self._cache_data["u_star"] = u_stars
        self._cache_data["signs"] = signs
        self._cache_data["normals"] = normals

    def project(self, coefficients: np.ndarray, points: np.ndarray):
        """
        Evaluate signed distance for query points and return (signed_distance, forward_state).
        Provides a drop-in execution interface compatible with external SDF evaluators.
        """
        outputs = {}
        self.compute({"coefficients": coefficients, "points": points}, outputs)
        return outputs["signed_distance"], None

    def compute_vjp(
        self,
        coefficients: np.ndarray,
        points: np.ndarray,
        cotangent: np.ndarray,
        forward_state=None,
    ):
        """
        Compute VJP of signed distance w.r.t query points and coefficients.
        Returns (d_points, d_coefficients).
        """
        vjp_op = FunctionSetSignedDistanceOperationVJP(
            function_set=self.detached_function_set,
            sign_mode=self.sign_mode,
            eps_reg=self.eps_reg,
            bound_tol=self.bound_tol,
            eps_hess=self.eps_hess,
            eps_lm=self.eps_lm,
            cache_data=self._cache_data,
        )
        outputs = {}
        vjp_op.compute(
            {
                "coefficients": coefficients,
                "points": points,
                "d_signed_distance": cotangent,
            },
            outputs,
        )
        return outputs["d_points"], outputs["d_coefficients"]
