"""
Cubic B-spline Target Regularization for the ar_area formulation.

This module provides static NumPy precomputations and CSDL-native weak residual assembly
for enforcing taper, thickness-to-chord ratio, and quarter-chord sweep targets
via an exact Galerkin weak form in the LSDO_GEO ParameterizationSolver.

The spline basis is symmetric across the root (y = 0) with zero spanwise derivative
(f'(0) = 0), matching the full-span FFD volume parameterization.
The geometry trial functions phi_j are cardinal symmetric cubic B-splines
satisfying phi_j(eta_i) = delta_ij at station locations, exactly matching the
cubic B-spline space of the FFD volume parameterization and target distributions.
Taper enforces T_h(0) = 1 as an essential boundary condition, removing the root DOF.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

def _to_vec(x):
    if isinstance(x, list):
        return csdl.concatenate([csdl.reshape(item, (1,)) for item in x])
    elif hasattr(x, "shape") and len(x.shape) == 0:
        return csdl.reshape(x, (1,))
    return x


class BsplineTargetRegularization:
    """
    Constructs the symmetric cubic B-spline spaces, static quadrature/test data,
    and CSDL-native weak residual equations for the ar_area formulation.
    """
    def __init__(
        self,
        num_chord_stations: int,
        scale_factor: float = 7.5,
        dense_points: int = 100,
        eta_stations: Optional[np.ndarray] = None,
    ):
        self.num_chord_stations = num_chord_stations
        self.n = num_chord_stations
        self.scale_factor = scale_factor
        self.dense_points = dense_points
        self.taper_control_points_count = self.n - 1
        self.tc_control_points_count = self.n
        self.sweep_control_points_count = self.n - 1

        # Normalized station coordinates in [0, 1]
        if eta_stations is not None:
            self.eta_stations = np.asarray(eta_stations, dtype=float)
        else:
            self.eta_stations = np.linspace(0.0, 1.0, self.n)

        # 1. Full-span FFD chord/thickness B-spline space (2n - 1 coefficients)
        num_ffd_chord = 2 * self.n - 1
        self.space_chord_full = lfs.BSplineSpace(
            num_parametric_dimensions=1, degree=3, coefficients_shape=(num_ffd_chord,)
        )
        knots_chord_full = self.space_chord_full.knots[0]
        knots_chord_eta = 2.0 * (knots_chord_full[knots_chord_full >= 0.5] - 0.5)

        # 2. Full-span FFD sweep B-spline space (2(n-1) - 1 coefficients)
        num_ffd_sweep = 2 * (self.n - 1) - 1
        self.space_sweep_full = lfs.BSplineSpace(
            num_parametric_dimensions=1, degree=3, coefficients_shape=(num_ffd_sweep,)
        )
        knots_sweep_full = self.space_sweep_full.knots[0]
        knots_sweep_eta = 2.0 * (knots_sweep_full[knots_sweep_full >= 0.5] - 0.5)

        # 3. Static subinterval partition at the union of station boundaries and knot spans
        self.partition = np.unique(np.concatenate([self.eta_stations, knots_chord_eta, knots_sweep_eta]))

        # 4. Static 4-point Gauss-Legendre quadrature setup
        z_leg, w_leg = np.polynomial.legendre.leggauss(4)
        quad_eta_list = []
        quad_w_list = []
        for k in range(len(self.partition) - 1):
            a, b = self.partition[k], self.partition[k + 1]
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)
            quad_eta_list.extend(mid + half * z_leg)
            quad_w_list.extend(half * w_leg)

        self.quad_eta = np.array(quad_eta_list)
        self.quad_w = np.array(quad_w_list)

        # 5. Evaluate symmetric bases at quadrature points
        self.B_chord_sym = self._eval_symmetric_chord_basis(self.quad_eta)       # shape (N_q, n)
        self.B_sweep_sym = self._eval_symmetric_sweep_basis(self.quad_eta)       # shape (N_q, n - 1)

        # 6. Evaluate cardinal symmetric cubic B-spline basis functions phi_j(eta_q)
        # B_stations maps B-spline control points to station values; its inverse gives the cardinal basis
        self.B_stations = self._eval_symmetric_chord_basis(self.eta_stations)     # shape (n, n)
        self.B_stations_inv = np.linalg.inv(self.B_stations)                     # shape (n, n)
        self.phi_quad = self.B_chord_sym @ self.B_stations_inv                   # shape (N_q, n)

        # 7. Precompute Galerkin matrices
        self._precompute_taper_matrices()
        self._precompute_thickness_matrices()
        self._precompute_sweep_matrices()
        self.B_bar_sweep = self.W_sweep.T                                        # shape (n - 1, n - 1)

        # 8. Setup dense evaluation matrices for diagnostics and plotting
        self.dense_eta = np.linspace(0.0, 1.0, dense_points)
        self.dense_B_chord = self._eval_symmetric_chord_basis(self.dense_eta)
        self.dense_B_sweep = self._eval_symmetric_sweep_basis(self.dense_eta)
        self.dense_phi = self.dense_B_chord @ self.B_stations_inv

    def _eval_cardinal_chord_basis(self, eta_vals: np.ndarray) -> np.ndarray:
        """
        Evaluate cardinal symmetric cubic B-spline basis functions phi_j on eta in [0, 1]
        satisfying phi_j(eta_i) = delta_ij at station locations.
        """
        return self._eval_symmetric_chord_basis(eta_vals) @ self.B_stations_inv

    def _eval_symmetric_chord_basis(self, eta_vals: np.ndarray) -> np.ndarray:
        """
        Evaluate the n symmetric cubic basis functions on normalized right-half span eta in [0, 1].
        """
        u = 0.5 + 0.5 * eta_vals.reshape(-1, 1)
        B_full = self.space_chord_full.compute_basis_matrix(u).toarray()
        n = self.n
        center_idx = n - 1

        B_sym = np.zeros((len(eta_vals), n))
        B_sym[:, 0] = B_full[:, center_idx]
        for i in range(1, n):
            B_sym[:, i] = B_full[:, center_idx + i] + B_full[:, center_idx - i]
        return B_sym

    def _eval_symmetric_sweep_basis(self, eta_vals: np.ndarray) -> np.ndarray:
        """
        Evaluate the n - 1 symmetric cubic sweep basis functions on eta in [0, 1].
        """
        u = 0.5 + 0.5 * eta_vals.reshape(-1, 1)
        B_full = self.space_sweep_full.compute_basis_matrix(u).toarray()
        n_sweep = self.n - 1
        center_idx = n_sweep - 1

        B_sym = np.zeros((len(eta_vals), n_sweep))
        B_sym[:, 0] = B_full[:, center_idx]
        for i in range(1, n_sweep):
            B_sym[:, i] = B_full[:, center_idx + i] + B_full[:, center_idx - i]
        return B_sym

    def _eval_hat_functions(self, eta_vals: np.ndarray) -> np.ndarray:
        """
        Evaluate piecewise-linear hat functions phi_j at points eta_vals.
        """
        n = self.n
        phi = np.zeros((len(eta_vals), n))
        for j in range(n - 1):
            eta_j = self.eta_stations[j]
            eta_next = self.eta_stations[j + 1]
            d_eta = eta_next - eta_j
            idx = (eta_vals >= eta_j) & (eta_vals <= eta_next)
            phi[idx, j] = (eta_next - eta_vals[idx]) / d_eta
            phi[idx, j + 1] = (eta_vals[idx] - eta_j) / d_eta
        return phi

    def _precompute_taper_matrices(self):
        """
        Precompute taper matrices K_taper (n-1, n), M_taper (n-1, n-1), and v0_taper (n-1).
        Test functions w_a for a = 1, ..., n-1 satisfy w_a(0) = 0:
            w_1 = B_1 - (B_1(0)/B_0(0)) * B_0 = B_1 - 0.5 * B_0
            w_a = B_a for a >= 2.
        Strong BC T_h(0) = 1 gives P_0 = 1.5 - 0.5 * P_1.
        Then T_h = 1.5 * B_0 + P_1 * (B_1 - 0.5 * B_0) + sum_{b=2}^{n-1} P_b * B_b.
        """
        n = self.n
        # Test functions: shape (N_q, n - 1)
        w_taper = np.zeros((len(self.quad_eta), n - 1))
        w_taper[:, 0] = self.B_chord_sym[:, 1] - 0.5 * self.B_chord_sym[:, 0]
        for a in range(2, n):
            w_taper[:, a - 1] = self.B_chord_sym[:, a]

        # Geometry matrix K_taper: (n-1, n)
        # int_0^1 w_a * phi_j d_eta
        self.K_taper = np.zeros((n - 1, n))
        for a in range(n - 1):
            for j in range(n):
                self.K_taper[a, j] = np.sum(self.quad_w * w_taper[:, a] * self.phi_quad[:, j])

        # Target trial functions Psi_b: shape (N_q, n - 1)
        # Psi_1 = B_1 - 0.5 * B_0, Psi_b = B_b for b >= 2
        Psi = np.zeros((len(self.quad_eta), n - 1))
        Psi[:, 0] = self.B_chord_sym[:, 1] - 0.5 * self.B_chord_sym[:, 0]
        for b in range(2, n):
            Psi[:, b - 1] = self.B_chord_sym[:, b]

        # Target mass matrix M_taper: (n-1, n-1)
        self.M_taper = np.zeros((n - 1, n - 1))
        for a in range(n - 1):
            for b in range(n - 1):
                self.M_taper[a, b] = np.sum(self.quad_w * w_taper[:, a] * Psi[:, b])

        # Constant target vector v0_taper: (n-1,) from the 1.5 * B_0 term
        self.v0_taper = np.zeros(n - 1)
        for a in range(n - 1):
            self.v0_taper[a] = 1.5 * np.sum(self.quad_w * w_taper[:, a] * self.B_chord_sym[:, 0])

    def _precompute_thickness_matrices(self):
        """
        Precompute thickness matrices K_thick (n, n) and T_thick (n, n, n).
        Test functions are all n symmetric chord basis functions B_a.
        int_0^1 w_a * t_g d_eta = K_thick @ t
        int_0^1 w_a * c_g * R_h d_eta = sum_{k, b} T_{a, k, b} * c_k * rho_b
        """
        n = self.n
        w_thick = self.B_chord_sym  # (N_q, n)

        self.K_thick = np.zeros((n, n))
        for a in range(n):
            for j in range(n):
                self.K_thick[a, j] = np.sum(self.quad_w * w_thick[:, a] * self.phi_quad[:, j])

        self.T_thick = np.zeros((n, n, n))
        for a in range(n):
            for k in range(n):
                for b in range(n):
                    self.T_thick[a, k, b] = np.sum(
                        self.quad_w * w_thick[:, a] * self.phi_quad[:, k] * self.B_chord_sym[:, b]
                    )

        # 2D flattened version for fast matvec
        self.T_thick_2d = self.T_thick.reshape(n * n, n)

    def _precompute_sweep_matrices(self):
        """
        Precompute sweep matrices W_sweep (n-1, n-1) and V_sweep (n-1, n-1, n-1).
        Test functions are all n - 1 symmetric sweep basis functions B_a^sweep.
        int_0^1 w_a * x'_{qc,g} d_eta = W_sweep @ dx_qc
        int_0^1 w_a * y'_{qc,g} * S_h d_eta = sum_{i, b} V_{a, i, b} * dy_i * tan(alpha_b)
        """
        n_sweep = self.n - 1
        w_sweep = self.B_sweep_sym  # (N_q, n - 1)

        self.W_sweep = np.zeros((n_sweep, n_sweep))
        self.V_sweep = np.zeros((n_sweep, n_sweep, n_sweep))

        for i in range(n_sweep):
            eta_i = self.eta_stations[i]
            eta_next = self.eta_stations[i + 1]
            d_eta = eta_next - eta_i
            idx = (self.quad_eta >= eta_i) & (self.quad_eta <= eta_next)

            for a in range(n_sweep):
                self.W_sweep[a, i] = np.sum(self.quad_w[idx] * w_sweep[idx, a]) / d_eta
                for b in range(n_sweep):
                    self.V_sweep[a, i, b] = (
                        np.sum(self.quad_w[idx] * w_sweep[idx, a] * self.B_sweep_sym[idx, b]) / d_eta
                    )

        # 2D flattened version for fast matvec
        self.V_sweep_2d = self.V_sweep.reshape(n_sweep * n_sweep, n_sweep)

    def compute_taper_residual(
        self, local_chords: csdl.Variable, taper_control_points: csdl.Variable, c_ref: float
    ) -> csdl.Variable:
        """
        Compute normalized taper weak residual vector of shape (n - 1,):
            R_taper = (K_taper @ local_chords - local_chords[0] * (M_taper @ taper_control_points + v0_taper)) / c_ref
        """
        local_chords = _to_vec(local_chords)
        taper_control_points = _to_vec(taper_control_points)
        c_root = local_chords[0]
        geom_int = csdl.matvec(self.K_taper, local_chords)
        target_int = c_root * (csdl.matvec(self.M_taper, taper_control_points) + self.v0_taper)
        res_taper = (geom_int - target_int) / c_ref
        res_taper.add_name("taper_weak_residual")
        return res_taper

    def compute_thickness_residual(
        self,
        local_thicknesses: csdl.Variable,
        local_chords: csdl.Variable,
        tc_target_control_points: csdl.Variable,
        t_ref: float,
    ) -> csdl.Variable:
        """
        Compute normalized thickness weak residual vector of shape (n,):
            R_thick = K_thick @ (local_thicknesses - local_chords * (B_stations @ tc_target_control_points)) / t_ref
        """
        local_thicknesses = _to_vec(local_thicknesses)
        local_chords = _to_vec(local_chords)
        tc_target_control_points = _to_vec(tc_target_control_points)
        geom_int = csdl.matvec(self.K_thick, local_thicknesses)
        tc_target_stations = csdl.matvec(self.B_stations, tc_target_control_points)
        target_thicknesses = local_chords * tc_target_stations
        target_int = csdl.matvec(self.K_thick, target_thicknesses)
        res_thick = (geom_int - target_int) / t_ref
        res_thick.add_name("thickness_weak_residual")
        return res_thick

    def compute_sweep_residual(
        self,
        dx_qc: csdl.Variable,
        dy_qc: csdl.Variable,
        sweep_angle_control_points: csdl.Variable,
        y_scale: float,
    ) -> csdl.Variable:
        """
        Compute normalized quarter-chord sweep weak residual vector of shape (n - 1,):
            R_sweep = (dx_qc - dy_qc * (B_bar_sweep @ tan(sweep_angle_control_points))) / y_scale
        where B_bar_sweep is the segment-averaged symmetric sweep B-spline matrix.
        """
        dx_qc = _to_vec(dx_qc)
        dy_qc = _to_vec(dy_qc)
        sweep_angle_control_points = _to_vec(sweep_angle_control_points)
        tan_sweep = csdl.tan(sweep_angle_control_points)
        tan_sweep_avg = csdl.matvec(self.B_bar_sweep, tan_sweep)
        target_dx = dy_qc * tan_sweep_avg
        res_sweep = (dx_qc - target_dx) / y_scale
        res_sweep.add_name("sweep_weak_residual")
        return res_sweep

    def compute_diagnostics(
        self,
        local_chords: csdl.Variable,
        local_thicknesses: csdl.Variable,
        dx_qc: csdl.Variable,
        dy_qc: csdl.Variable,
        taper_control_points: csdl.Variable,
        tc_target_control_points: csdl.Variable,
        sweep_angle_control_points: csdl.Variable,
    ) -> Dict[str, csdl.Variable]:
        """
        Compute dense profiles, mismatch residuals, and discrete second-difference curvature indicators.
        """
        local_chords = _to_vec(local_chords)
        local_thicknesses = _to_vec(local_thicknesses)
        dx_qc = _to_vec(dx_qc)
        dy_qc = _to_vec(dy_qc)
        taper_control_points = _to_vec(taper_control_points)
        tc_target_control_points = _to_vec(tc_target_control_points)
        sweep_angle_control_points = _to_vec(sweep_angle_control_points)

        c_root = local_chords[0]
        n = self.n

        # 1. Full taper control vector: P_0 = 1.5 - 0.5 * P_1, P_1, ..., P_{n-1}
        P_1 = taper_control_points[0]
        P_0 = csdl.reshape(1.5 - 0.5 * P_1, (1,))
        taper_full_cp = csdl.concatenate([P_0, taper_control_points])

        # 2. Dense target distributions
        target_taper_dense = csdl.matvec(self.dense_B_chord, taper_full_cp)
        target_tc_dense = csdl.matvec(self.dense_B_chord, tc_target_control_points)
        tan_sweep = csdl.tan(sweep_angle_control_points)
        target_sweep_tan_dense = csdl.matvec(self.dense_B_sweep, tan_sweep)
        target_sweep_dense = csdl.arctan(target_sweep_tan_dense)

        # 3. Dense reconstructed distributions
        c_dense = csdl.matvec(self.dense_phi, local_chords)
        t_dense = csdl.matvec(self.dense_phi, local_thicknesses)
        reconstructed_taper_dense = c_dense / c_root
        reconstructed_tc_dense = t_dense / c_dense

        # Dense piecewise-constant sweep reconstruction
        sweep_recon_list = []
        for i in range(self.n - 1):
            sw_val = csdl.arctan2(dx_qc[i], dy_qc[i])
            sweep_recon_list.append(csdl.reshape(sw_val, (1,)))
        station_sweep_angles = csdl.concatenate(sweep_recon_list)

        # Map station sweep to dense segments
        dense_sweep_pieces = []
        for k, eta_val in enumerate(self.dense_eta):
            # Find interval
            seg_idx = min(int(eta_val * (self.n - 1)), self.n - 2)
            dense_sweep_pieces.append(csdl.reshape(station_sweep_angles[seg_idx], (1,)))
        reconstructed_sweep_dense = csdl.concatenate(dense_sweep_pieces)

        # 4. Pointwise mismatch diagnostics
        taper_mismatch_dense = reconstructed_taper_dense - target_taper_dense
        tc_mismatch_dense = reconstructed_tc_dense - target_tc_dense
        sweep_mismatch_dense = reconstructed_sweep_dense - target_sweep_dense

        # 5. Second-difference curvature indicators (Delta^2 y_k = y_{k+1} - 2*y_k + y_{k-1})
        target_taper_curvature = (
            target_taper_dense[2:] - 2.0 * target_taper_dense[1:-1] + target_taper_dense[:-2]
        )
        reconstructed_taper_curvature = (
            reconstructed_taper_dense[2:] - 2.0 * reconstructed_taper_dense[1:-1] + reconstructed_taper_dense[:-2]
        )
        target_sweep_curvature = (
            target_sweep_dense[2:] - 2.0 * target_sweep_dense[1:-1] + target_sweep_dense[:-2]
        )

        return {
            "target_taper_dense": target_taper_dense,
            "reconstructed_taper_dense": reconstructed_taper_dense,
            "taper_mismatch_dense": taper_mismatch_dense,
            "target_tc_dense": target_tc_dense,
            "reconstructed_tc_dense": reconstructed_tc_dense,
            "tc_mismatch_dense": tc_mismatch_dense,
            "target_sweep_dense": target_sweep_dense,
            "reconstructed_sweep_dense": reconstructed_sweep_dense,
            "sweep_mismatch_dense": sweep_mismatch_dense,
            "target_taper_curvature": target_taper_curvature,
            "reconstructed_taper_curvature": reconstructed_taper_curvature,
            "target_sweep_curvature": target_sweep_curvature,
            "taper_control_points_full": taper_full_cp,
        }
