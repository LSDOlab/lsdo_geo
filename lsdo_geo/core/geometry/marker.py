from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, Sequence, Tuple, List, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

if TYPE_CHECKING:
    from lsdo_geo.core.geometry.geometry import Geometry


@dataclass(eq=False)
class Marker(lfs.Marker):
    """
    A geometric marker (datum point / evaluation site) in a 3D Geometry.

    Defined either directly in parametric coordinates or by projecting a physical
    3D Cartesian point onto the geometry. Supports evaluating physical position,
    tangent vectors, surface normal, and connection alignment null-spaces.

    Parameters
    ----------
    geometry : Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]], optional
        The geometry entity to which the marker is attached.
    parametric_coordinates : Optional[Union[tuple[int, npt.NDArray[np.float64]], list[tuple[int, npt.NDArray[np.float64]]]]]
        The parametric coordinate(s) on the geometry patches, specified as
        a tuple (function_index, uv_array) or list of such tuples.
    name : Optional[str]
        The name of the marker. If None, an auto-generated unique name is assigned.
    initial_point : Optional[npt.NDArray[np.float64]]
        The initial physical 3D Cartesian coordinates of the marker.
    """
    initial_point: Optional[npt.NDArray[np.float64]] = None

    def __init__(
        self,
        geometry: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        parametric_coordinates: Optional[Any] = None,
        name: Optional[str] = None,
        initial_point: Optional[npt.NDArray[np.float64]] = None,
        domain: Optional[lfs.InterfaceDomain] = None,
    ):
        super().__init__(
            entity=geometry,
            parametric_coordinates=parametric_coordinates,
            name=name,
            initial_point=initial_point,
            domain=domain,
        )
        self.value: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = initial_point

    @property
    def geometry(self) -> Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]]:
        """Returns the geometry associated with this marker."""
        return self.entity

    @geometry.setter
    def geometry(self, val: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]]):
        """Sets the geometry associated with this marker."""
        self.entity = val

    def copy(
        self,
        geometry: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a copy of the marker, optionally rebinding to a new geometry.
        """
        target = geometry if geometry is not None else self.geometry
        return self.__class__(
            geometry=target,
            parametric_coordinates=self.parametric_coordinates,
            name=name if name is not None else self.name,
            initial_point=self.initial_point.copy() if self.initial_point is not None else None,
            domain=self.domain,
        )

    @classmethod
    def from_function(
        cls,
        geometry: Union[Geometry, lfs.Function, lfs.FunctionSet],
        function_map: Union[lfs.Function, Tuple[int, lfs.Function]],
        domain: Optional[lfs.InterfaceDomain] = None,
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a continuous marker where alpha(gamma) is an lsdo_function_spaces.Function
        mapping gamma in Gamma to parametric coordinates in the geometry's domain.
        """
        if domain is None:
            fn = function_map[1] if isinstance(function_map, tuple) else function_map
            bounds = getattr(fn, "parametric_bounds", None)
            if bounds is not None:
                domain = lfs.InterfaceDomain(bounds=np.asarray(bounds, dtype=np.float64), num_copies=1)
            else:
                m_dim = getattr(fn.space, "num_parametric_dimensions", 1)
                domain = lfs.InterfaceDomain(bounds=np.array([[0.0, 1.0]] * m_dim, dtype=np.float64), num_copies=1)

        return cls(
            geometry=geometry,
            parametric_coordinates=function_map,
            domain=domain,
            name=name,
        )

    def __hash__(self):
        return id(self)

    @classmethod
    def from_parametric(
        cls,
        geometry: Geometry,
        parametric_coordinates: Union[tuple[int, npt.NDArray[np.float64]], list[tuple[int, npt.NDArray[np.float64]]]],
        name: Optional[str] = None,
    ) -> Marker:
        """
        Creates a marker directly from patch parametric coordinates on a Geometry.

        Parameters
        ----------
        geometry : Geometry
            The geometry to which this marker is attached.
        parametric_coordinates : Union[tuple[int, npt.NDArray[np.float64]], list[tuple[int, npt.NDArray[np.float64]]]]
            Parametric coordinate(s) on the geometry patches, specified as
            either a single tuple (function_index, uv_array) or a list of such tuples.
        name : Optional[str]
            Optional unique identifier name for the marker.

        Returns
        -------
        Marker
            The newly created geometric marker.
        """
        if isinstance(parametric_coordinates, list) and len(parametric_coordinates) > 0 and isinstance(parametric_coordinates[0], tuple):
            coords = [(int(idx), np.asarray(uv, dtype=np.float64)) for idx, uv in parametric_coordinates]
        elif isinstance(parametric_coordinates, tuple) and len(parametric_coordinates) == 2 and isinstance(parametric_coordinates[0], (int, np.integer)):
            coords = [(int(parametric_coordinates[0]), np.asarray(parametric_coordinates[1], dtype=np.float64))]
        elif hasattr(geometry, 'functions') and len(geometry.functions) == 1 and isinstance(parametric_coordinates, (np.ndarray, list)):
            first_idx = list(geometry.functions.keys())[0]
            coords = [(first_idx, np.asarray(parametric_coordinates, dtype=np.float64))]
        else:
            raise ValueError(
                f"Parametric coordinates for Geometry must be a tuple of (function_index, uv_array) "
                f"or a list of such tuples, but got {type(parametric_coordinates)}."
            )

        marker = cls(
            geometry=geometry,
            parametric_coordinates=coords,
            name=name,
        )
        try:
            val = marker.evaluate(non_csdl=True)
            if hasattr(val, "value"):
                val = val.value
            marker.initial_point = np.asarray(val, dtype=np.float64)
            if marker.initial_point.ndim == 2 and marker.initial_point.shape[0] == 1:
                marker.initial_point = marker.initial_point.flatten()
        except Exception:
            try:
                val = marker.evaluate()
                if hasattr(val, "value"):
                    val = val.value
                marker.initial_point = np.asarray(val, dtype=np.float64)
                if marker.initial_point.ndim == 2 and marker.initial_point.shape[0] == 1:
                    marker.initial_point = marker.initial_point.flatten()
            except Exception:
                pass
        return marker

    @classmethod
    def from_locations(
        cls,
        geometry: Geometry,
        locations: npt.NDArray[np.float64],
        name: Optional[str] = None,
        projection_tolerance: float = 1e-6,
    ) -> Marker:
        """
        Creates a marker by projecting physical 3D Cartesian coordinates onto the geometry.

        Parameters
        ----------
        geometry : Geometry
            The geometry to which this marker is attached.
        locations : npt.NDArray[np.float64]
            3D Cartesian coordinate array of shape (3,) or (N, 3).
        name : Optional[str]
            Optional unique identifier name for the marker.
        projection_tolerance : float
            Convergence tolerance for the geometric surface projection algorithm.

        Returns
        -------
        Marker
            The newly created geometric marker.
        """
        pt = np.asarray(locations, dtype=np.float64)
        if pt.ndim == 1:
            pt_project = pt.reshape((1, 3))
        else:
            pt_project = pt
        proj_coords = geometry.project(pt_project, projection_tolerance=projection_tolerance)
        marker = cls(
            geometry=geometry,
            parametric_coordinates=proj_coords,
            name=name,
            initial_point=pt,
        )
        return marker

    @classmethod
    def from_point(
        cls,
        geometry: Geometry,
        point: npt.NDArray[np.float64],
        name: Optional[str] = None,
        projection_tolerance: float = 1e-6,
    ) -> Marker:
        """
        Convenience alias for from_locations to create a marker from a 3D Cartesian point.

        Parameters
        ----------
        geometry : Geometry
            The geometry to which this marker is attached.
        point : npt.NDArray[np.float64]
            3D Cartesian point coordinates of shape (3,).
        name : Optional[str]
            Optional unique identifier name for the marker.
        projection_tolerance : float
            Convergence tolerance for the surface projection algorithm.

        Returns
        -------
        Marker
            The newly created geometric marker.
        """
        return cls.from_locations(geometry=geometry, locations=point, name=name,
                                  projection_tolerance=projection_tolerance)

    def evaluate(
        self,
        geometry: Optional[Geometry] = None,
        plot: bool = False,
        non_csdl: bool = False,
        sample_parameters: Optional[npt.NDArray[np.float64]] = None,
        parametric_derivative_orders: Optional[Sequence[int]] = None,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Evaluates the physical 3D Cartesian position or parametric derivative of the marker.

        Parameters
        ----------
        geometry : Optional[Geometry], optional
            Geometry instance to evaluate on (overrides self.entity if provided).
        plot : bool, optional
            Whether to trigger visual plotting of the evaluation point. Default is False.
        non_csdl : bool, optional
            If True, evaluates purely in NumPy without building CSDL graph operations. Default is False.
        sample_parameters : Optional[npt.NDArray[np.float64]], optional
            Collocation / quadrature sampling nodes for continuous (m >= 1) markers.
        parametric_derivative_orders : Optional[Sequence[int]], optional
            Orders of derivative with respect to each parametric dimension, e.g. (1, 0) for tu.

        Returns
        -------
        Union[csdl.Variable, npt.NDArray[np.float64]]
            The evaluated 3D coordinates or derivative vector (CSDL Variable in graph mode, or NumPy array).
        """
        val = super().evaluate(
            entity=geometry,
            parametric_derivative_orders=parametric_derivative_orders,
            plot=plot,
            non_csdl=non_csdl,
            sample_parameters=sample_parameters,
        )
        self.value = val
        return val

    def evaluate_derivatives(
        self,
        derivative_orders: Sequence[int],
        geometry: Optional[Geometry] = None,
        non_csdl: bool = False,
        sample_parameters: Optional[npt.NDArray[np.float64]] = None,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Evaluates surface parametric derivatives at the marker location.

        Parameters
        ----------
        derivative_orders : Sequence[int]
            Orders of derivative with respect to each parametric dimension, e.g. (1, 0) for tu.
        geometry : Optional[Geometry], optional
            Geometry instance to evaluate on.
        non_csdl : bool, optional
            If True, evaluates purely in NumPy without building CSDL graph operations. Default is False.
        sample_parameters : Optional[npt.NDArray[np.float64]], optional
            Collocation / quadrature sampling nodes for continuous (m >= 1) markers.

        Returns
        -------
        Union[csdl.Variable, npt.NDArray[np.float64]]
            The evaluated surface derivative vector.
        """
        return self.evaluate(
            geometry=geometry,
            parametric_derivative_orders=derivative_orders,
            non_csdl=non_csdl,
            sample_parameters=sample_parameters,
        )

    def compute_tangents(
        self,
        geometry: Optional[Geometry] = None,
        non_csdl: bool = False,
    ) -> Tuple[Union[csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]:
        """
        Computes the surface tangent vectors (tu, tv) at the marker location.

        Parameters
        ----------
        geometry : Optional[Geometry]
            Geometry instance to evaluate on.
        non_csdl : bool
            If True, evaluates purely in NumPy without building CSDL graph operations. Default is False.

        Returns
        -------
        Tuple[Union[csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]
            Tuple of tangent vectors (tu, tv).
        """
        tu = self.evaluate_derivatives((1, 0), geometry=geometry, non_csdl=non_csdl)
        tv = self.evaluate_derivatives((0, 1), geometry=geometry, non_csdl=non_csdl)
        return tu, tv

    def compute_surface_normal(
        self,
        geometry: Optional[Geometry] = None,
        non_csdl: bool = False,
        flip: bool = False,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Computes the unit surface normal vector at the marker location.

        Parameters
        ----------
        geometry : Optional[Geometry]
            Geometry instance to evaluate on.
        non_csdl : bool
            If True, evaluates purely in NumPy without building CSDL graph operations. Default is False.
        flip : bool
            If True, inverts normal direction (-n). Default is False.

        Returns
        -------
        Union[csdl.Variable, npt.NDArray[np.float64]]
            Unit surface normal 3D vector.
        """
        tu, tv = self.compute_tangents(geometry=geometry, non_csdl=non_csdl)
        if non_csdl:
            normal = np.cross(tu, tv)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
            if flip:
                normal = -normal
            return normal
        else:
            normal = csdl.cross(tu, tv)
            norm = csdl.norm(normal)
            normal = normal / norm
            if flip:
                normal = -normal
            return normal

    def compute_null_space(
        self,
        geometry: Optional[Geometry] = None,
        non_csdl: bool = False,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Computes a 3x2 orthonormal basis spanning the tangent plane orthogonal to the surface normal.

        Parameters
        ----------
        geometry : Optional[Geometry]
            Geometry instance to evaluate on.
        non_csdl : bool
            If True, evaluates purely in NumPy without building CSDL graph operations. Default is False.

        Returns
        -------
        Union[csdl.Variable, npt.NDArray[np.float64]]
            A 3x2 matrix [t1, t2] spanning the null space perpendicular to the normal.
        """
        tu, tv = self.compute_tangents(geometry=geometry, non_csdl=non_csdl)
        if non_csdl:
            normal = np.cross(tu, tv)
            norm_n = np.linalg.norm(normal)
            if norm_n > 1e-12:
                normal = normal / norm_n
            norm_tu = np.linalg.norm(tu)
            if norm_tu > 1e-12:
                t1 = tu / norm_tu
            else:
                ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                t1 = np.cross(normal, ref)
                t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            return np.column_stack((t1, t2))
        else:
            normal = csdl.cross(tu, tv)
            normal = normal / csdl.norm(normal)
            t1 = tu / csdl.norm(tu)
            t2 = csdl.cross(normal, t1)
            null_space = csdl.Variable(value=np.zeros((3, 2)))
            null_space = null_space.set(csdl.slice[:, 0], t1)
            null_space = null_space.set(csdl.slice[:, 1], t2)
            return null_space
