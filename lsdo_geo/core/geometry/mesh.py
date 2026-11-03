from __future__ import annotations

from typing import Union, Optional, Sequence, Tuple, List, Callable, Dict, Any, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

from lsdo_geo.core.geometry.marker import Marker

if TYPE_CHECKING:
    from lsdo_geo.core.geometry.geometry import Geometry


class Mesh(Marker):
    """
    A geometric and disciplinary mesh discretization on a Geometry.

    Inherits all continuous and discrete capabilities from Marker, allowing a Mesh
    to serve directly as a discrete nodal discretization or as a continuous function
    space (e.g., for Isogeometric Analysis, IGA).

    Supports constituent markers and custom subroutines (functions of markers), enabling
    composite disciplinary meshes such as VLM mean camber surfaces, panel wireframes,
    or structural beam grids.

    Parameters
    ----------
    geometry : Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]], optional
        The geometry entity to which the mesh is attached.
    entity : Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]], optional
        Alias for geometry (inherited from Marker).
    parametric_coordinates : Optional[Any], optional
        The parametric coordinate(s) on the geometry patches. Can be discrete coordinates
        or a continuous Function/patch map for IGA.
    markers : Optional[Union[Dict[str, Marker], Sequence[Marker]]], optional
        Constituent marker(s) backing the mesh.
    subroutine : Optional[Callable], optional
        Custom post-processing subroutine mapping evaluated markers/nodes to mesh values.
    shape : Optional[Tuple[int, ...]], optional
        Expected output shape of the mesh variable.
    name : Optional[str], optional
        Descriptive identifier name. If None, an auto-incremented name 'mesh_{counter}' is assigned.
    initial_point : Optional[npt.NDArray[np.float64]], optional
        Initial physical coordinates.
    domain : Optional[lfs.InterfaceDomain], optional
        Parametric domain (deduced automatically if omitted).
    """
    mesh_counter = 0

    def __init__(
        self,
        geometry: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        entity: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        parametric_coordinates: Optional[Any] = None,
        markers: Optional[Union[Dict[str, Marker], Sequence[Marker]]] = None,
        subroutine: Optional[Callable] = None,
        shape: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
        initial_point: Optional[npt.NDArray[np.float64]] = None,
        domain: Optional[lfs.InterfaceDomain] = None,
    ):
        resolved_entity = entity if entity is not None else geometry
        if resolved_entity is None and markers is not None:
            if isinstance(markers, dict) and len(markers) > 0:
                resolved_entity = next(iter(markers.values())).entity
            elif isinstance(markers, (list, tuple)) and len(markers) > 0:
                resolved_entity = markers[0].entity

        if name is None:
            name = f"mesh_{Mesh.mesh_counter}"
            Mesh.mesh_counter += 1

        super().__init__(
            entity=resolved_entity,
            parametric_coordinates=parametric_coordinates,
            name=name,
            initial_point=initial_point,
            domain=domain,
        )

        if isinstance(markers, (list, tuple)):
            self.markers: Optional[Dict[str, Marker]] = {
                (m.name if hasattr(m, "name") and m.name else f"marker_{i}"): m
                for i, m in enumerate(markers)
            }
        elif isinstance(markers, dict):
            self.markers = markers
        else:
            self.markers = None

        self.subroutine = subroutine
        self.shape = shape
        self.value: Optional[csdl.Variable] = None

        # Shape inference if not explicitly provided
        if self.shape is None and self.parametric_coordinates is not None and self.entity is not None:
            if isinstance(self.entity, lfs.FunctionSet):
                num_physical_dimensions = list(self.entity.functions.values())[0].coefficients.shape[-1]
                if isinstance(self.parametric_coordinates, list) and len(self.parametric_coordinates) > 0:
                    uv_shape = getattr(self.parametric_coordinates[0][1], "shape", ())
                    self.shape = (len(self.parametric_coordinates),) + tuple(uv_shape[:-1]) + (num_physical_dimensions,)
            elif isinstance(self.entity, lfs.Function):
                num_physical_dimensions = self.entity.coefficients.shape[-1]
                uv_shape = getattr(self.parametric_coordinates, "shape", ())
                self.shape = tuple(uv_shape[:-1]) + (num_physical_dimensions,)

    @property
    def geometry(self) -> Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]]:
        """Returns the geometry entity associated with this mesh."""
        return self.entity

    @geometry.setter
    def geometry(self, val: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]]):
        """Sets the geometry entity associated with this mesh."""
        self.entity = val

    def evaluate(
        self,
        geometry: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        plot: bool = False,
        non_csdl: bool = False,
        sample_parameters: Optional[npt.NDArray[np.float64]] = None,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Evaluates the mesh on the geometry.

        If a subroutine is defined, it is executed on the evaluated constituent markers
        (or evaluated nodal coordinates). Otherwise, evaluates the underlying marker.

        Parameters
        ----------
        geometry : Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]], optional
            Geometry to evaluate on. If provided, updates self.entity.
        plot : bool, optional
            Whether to plot the evaluated mesh. Default is False.
        non_csdl : bool, optional
            If True, evaluates in NumPy without CSDL graph operations. Default is False.
        sample_parameters : Optional[npt.NDArray[np.float64]], optional
            Collocation / quadrature sampling parameters for continuous (IGA) function meshes.

        Returns
        -------
        Union[csdl.Variable, npt.NDArray[np.float64]]
            The evaluated mesh coordinates or subroutine output.
        """
        if geometry is not None:
            self.entity = geometry

        target_entity = self.entity

        if self.subroutine is not None:
            if self.markers is not None:
                evaluated_markers = {
                    name: m.evaluate(
                        geometry=target_entity,
                        plot=plot,
                        non_csdl=non_csdl,
                        sample_parameters=sample_parameters,
                    )
                    for name, m in self.markers.items()
                }
                res = self.subroutine(evaluated_markers)
            else:
                pts = super().evaluate(
                    geometry=target_entity,
                    plot=plot,
                    non_csdl=non_csdl,
                    sample_parameters=sample_parameters,
                )
                res = self.subroutine(pts)
        else:
            if self.markers is not None:
                evaluated_markers = {
                    name: m.evaluate(
                        geometry=target_entity,
                        plot=plot,
                        non_csdl=non_csdl,
                        sample_parameters=sample_parameters,
                    )
                    for name, m in self.markers.items()
                }
                if len(evaluated_markers) == 1:
                    res = next(iter(evaluated_markers.values()))
                else:
                    res = evaluated_markers
            else:
                res = super().evaluate(
                    geometry=target_entity,
                    plot=plot,
                    non_csdl=non_csdl,
                    sample_parameters=sample_parameters,
                )

        if not non_csdl and isinstance(res, csdl.Variable):
            self.value = res
        return res


class CamberSurfaceMesh(Mesh):
    """
    A mean camber surface mesh formed by averaging upper and lower surface wireframes.

    Parameters
    ----------
    upper_surface_marker : Marker
        Marker evaluating the upper surface wireframe points.
    lower_surface_marker : Marker
        Marker evaluating the lower surface wireframe points.
    shape : Optional[Tuple[int, ...]], optional
        Structured grid shape, e.g. (num_chordwise, num_spanwise, 3).
    name : Optional[str], optional
        Descriptive identifier name.
    """
    def __init__(
        self,
        upper_surface_marker: Marker,
        lower_surface_marker: Marker,
        shape: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            markers={"upper": upper_surface_marker, "lower": lower_surface_marker},
            shape=shape,
            name=name,
        )

    def evaluate(
        self,
        geometry: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        plot: bool = False,
        non_csdl: bool = False,
        sample_parameters: Optional[npt.NDArray[np.float64]] = None,
    ) -> Union[csdl.Variable, npt.NDArray[np.float64]]:
        """
        Evaluates the mean camber surface by averaging upper and lower surface markers.
        """
        upper = self.markers["upper"].evaluate(
            geometry=geometry, plot=plot, non_csdl=non_csdl, sample_parameters=sample_parameters
        )
        lower = self.markers["lower"].evaluate(
            geometry=geometry, plot=plot, non_csdl=non_csdl, sample_parameters=sample_parameters
        )
        camber = (upper + lower) / 2.0
        if self.shape is not None:
            camber = camber.reshape(self.shape)
        if not non_csdl and isinstance(camber, csdl.Variable):
            self.value = camber
        return camber


class UnstructuredMesh(Mesh):
    """
    An unstructured mesh with explicit cell connectivity (e.g., quads, triangles, lines).

    Parameters
    ----------
    geometry : Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]], optional
        The geometry entity to which the mesh is attached.
    entity : Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]], optional
        Alias for geometry.
    parametric_coordinates : Optional[Any], optional
        Parametric coordinate(s) of the mesh nodes.
    markers : Optional[Union[Dict[str, Marker], Sequence[Marker]]], optional
        Constituent marker(s) backing the mesh nodes.
    cells : Optional[Dict[str, npt.NDArray[np.int64]]], optional
        Dictionary mapping cell element type (e.g., 'quad', 'triangle', 'line') to
        integer connectivity arrays of shape (num_elements, nodes_per_element).
    subroutine : Optional[Callable], optional
        Custom post-processing subroutine.
    shape : Optional[Tuple[int, ...]], optional
        Explicit nodal shape tuple.
    name : Optional[str], optional
        Descriptive identifier name.
    initial_point : Optional[npt.NDArray[np.float64]], optional
        Initial physical coordinates.
    domain : Optional[lfs.InterfaceDomain], optional
        Parametric domain.
    """
    def __init__(
        self,
        geometry: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        entity: Optional[Union[Geometry, lfs.FunctionSet, lfs.Function]] = None,
        parametric_coordinates: Optional[Any] = None,
        markers: Optional[Union[Dict[str, Marker], Sequence[Marker]]] = None,
        cells: Optional[Dict[str, npt.NDArray[np.int64]]] = None,
        subroutine: Optional[Callable] = None,
        shape: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
        initial_point: Optional[npt.NDArray[np.float64]] = None,
        domain: Optional[lfs.InterfaceDomain] = None,
    ):
        super().__init__(
            geometry=geometry,
            entity=entity,
            parametric_coordinates=parametric_coordinates,
            markers=markers,
            subroutine=subroutine,
            shape=shape,
            name=name,
            initial_point=initial_point,
            domain=domain,
        )
        self.cells = cells or {}

    def compute_element_centroids(
        self,
        evaluated_nodes: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None,
        non_csdl: bool = False,
    ) -> Dict[str, Union[csdl.Variable, npt.NDArray[np.float64]]]:
        """
        Computes the geometric centroids for each cell type.

        Parameters
        ----------
        evaluated_nodes : Optional[Union[csdl.Variable, npt.NDArray[np.float64]]], optional
            Node coordinates array of shape (num_nodes, 3). If None, evaluates the mesh.
        non_csdl : bool, optional
            If True, computes using NumPy operations. Default is False.

        Returns
        -------
        Dict[str, Union[csdl.Variable, npt.NDArray[np.float64]]]
            Dictionary mapping cell type to centroid coordinate array of shape (num_cells, 3).
        """
        if evaluated_nodes is None:
            evaluated_nodes = self.evaluate(non_csdl=non_csdl)

        centroids = {}
        for cell_type, conn in self.cells.items():
            conn_arr = np.asarray(conn, dtype=int)
            num_cells, nodes_per_elem = conn_arr.shape
            if non_csdl:
                pts = np.asarray(evaluated_nodes, dtype=np.float64)
                if pts.ndim > 2:
                    pts = pts.reshape((-1, pts.shape[-1]))
                centroids[cell_type] = np.mean(pts[conn_arr], axis=1)
            else:
                pts = evaluated_nodes
                if len(pts.shape) > 2:
                    pts = pts.reshape((-1, pts.shape[-1]))
                num_nodes = pts.shape[0]
                M = np.zeros((num_cells, num_nodes), dtype=np.float64)
                for i in range(num_cells):
                    M[i, conn_arr[i]] = 1.0 / nodes_per_elem
                centroids[cell_type] = csdl.matmat(M, pts)
        return centroids

    def compute_element_normals_and_areas(
        self,
        evaluated_nodes: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None,
        non_csdl: bool = False,
    ) -> Dict[str, Tuple[Union[csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]]:
        """
        Computes the unit outward normal vectors and surface areas for 2D surface elements (quads, triangles).

        Parameters
        ----------
        evaluated_nodes : Optional[Union[csdl.Variable, npt.NDArray[np.float64]]], optional
            Node coordinates array of shape (num_nodes, 3). If None, evaluates the mesh.
        non_csdl : bool, optional
            If True, computes using NumPy operations. Default is False.

        Returns
        -------
        Dict[str, Tuple[Union[csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]]
            Dictionary mapping cell type to a tuple of (unit_normals, areas).
        """
        if evaluated_nodes is None:
            evaluated_nodes = self.evaluate(non_csdl=non_csdl)

        def _extract_corner(pts, conn_arr, corner_idx):
            num_cells = conn_arr.shape[0]
            num_nodes = pts.shape[0]
            P = np.zeros((num_cells, num_nodes), dtype=np.float64)
            for i in range(num_cells):
                P[i, conn_arr[i, corner_idx]] = 1.0
            return csdl.matmat(P, pts)

        results = {}
        for cell_type, conn in self.cells.items():
            conn_arr = np.asarray(conn, dtype=int)
            if cell_type in ("triangle", "tri", "tri3"):
                if non_csdl:
                    pts = np.asarray(evaluated_nodes, dtype=np.float64)
                    if pts.ndim > 2:
                        pts = pts.reshape((-1, pts.shape[-1]))
                    v1 = pts[conn_arr[:, 1]] - pts[conn_arr[:, 0]]
                    v2 = pts[conn_arr[:, 2]] - pts[conn_arr[:, 0]]
                    cross = np.cross(v1, v2)
                    norms = np.linalg.norm(cross, axis=1, keepdims=True)
                    norms_clean = np.where(norms > 1e-12, norms, 1.0)
                    unit_normals = cross / norms_clean
                    areas = norms.flatten() / 2.0
                    results[cell_type] = (unit_normals, areas)
                else:
                    pts = evaluated_nodes
                    if len(pts.shape) > 2:
                        pts = pts.reshape((-1, pts.shape[-1]))
                    num_cells = conn_arr.shape[0]
                    n0 = _extract_corner(pts, conn_arr, 0)
                    n1 = _extract_corner(pts, conn_arr, 1)
                    n2 = _extract_corner(pts, conn_arr, 2)
                    v1 = n1 - n0
                    v2 = n2 - n0
                    cross = csdl.cross(v1, v2, axis=1)
                    norms = csdl.norm(cross, axes=(1,))
                    if num_cells == 1:
                        norm_2d = csdl.expand(norms, (1, 3))
                    else:
                        norm_2d = csdl.expand(norms, (num_cells, 3), action='i->ij')
                    unit_normals = cross / norm_2d
                    areas = norms / 2.0
                    results[cell_type] = (unit_normals, areas)
            elif cell_type in ("quad", "quad4"):
                if non_csdl:
                    pts = np.asarray(evaluated_nodes, dtype=np.float64)
                    if pts.ndim > 2:
                        pts = pts.reshape((-1, pts.shape[-1]))
                    d1 = pts[conn_arr[:, 2]] - pts[conn_arr[:, 0]]
                    d2 = pts[conn_arr[:, 3]] - pts[conn_arr[:, 1]]
                    cross = np.cross(d1, d2)
                    norms = np.linalg.norm(cross, axis=1, keepdims=True)
                    norms_clean = np.where(norms > 1e-12, norms, 1.0)
                    unit_normals = cross / norms_clean
                    areas = norms.flatten() / 2.0
                    results[cell_type] = (unit_normals, areas)
                else:
                    pts = evaluated_nodes
                    if len(pts.shape) > 2:
                        pts = pts.reshape((-1, pts.shape[-1]))
                    num_cells = conn_arr.shape[0]
                    n0 = _extract_corner(pts, conn_arr, 0)
                    n1 = _extract_corner(pts, conn_arr, 1)
                    n2 = _extract_corner(pts, conn_arr, 2)
                    n3 = _extract_corner(pts, conn_arr, 3)
                    d1 = n2 - n0
                    d2 = n3 - n1
                    cross = csdl.cross(d1, d2, axis=1)
                    norms = csdl.norm(cross, axes=(1,))
                    if num_cells == 1:
                        norm_2d = csdl.expand(norms, (1, 3))
                    else:
                        norm_2d = csdl.expand(norms, (num_cells, 3), action='i->ij')
                    unit_normals = cross / norm_2d
                    areas = norms / 2.0
                    results[cell_type] = (unit_normals, areas)
        return results

    def to_meshio(
        self,
        evaluated_nodes: Optional[npt.NDArray[np.float64]] = None,
    ) -> Any:
        """
        Converts this mesh to a meshio.Mesh object for exporting to .vtk, .msh, .bdf, etc.

        Parameters
        ----------
        evaluated_nodes : Optional[npt.NDArray[np.float64]], optional
            Nodal coordinates array. If None, evaluates the mesh in non_csdl mode.

        Returns
        -------
        meshio.Mesh
            The converted meshio data structure.
        """
        try:
            import meshio
        except ImportError:
            raise ImportError("meshio is required for to_meshio(); please install via pip install meshio")

        if evaluated_nodes is None:
            if self.value is not None and hasattr(self.value, "value"):
                nodes = np.asarray(self.value.value, dtype=np.float64)
            elif self.initial_point is not None:
                nodes = np.asarray(self.initial_point, dtype=np.float64)
            else:
                nodes = np.asarray(self.evaluate(non_csdl=True), dtype=np.float64)
        else:
            nodes = np.asarray(evaluated_nodes, dtype=np.float64)

        if nodes.ndim == 1:
            nodes = nodes.reshape((1, -1))
        elif nodes.ndim > 2:
            nodes = nodes.reshape((-1, nodes.shape[-1]))

        cell_blocks = [(k, np.asarray(v, dtype=int)) for k, v in self.cells.items()]
        return meshio.Mesh(points=nodes, cells=cell_blocks)
