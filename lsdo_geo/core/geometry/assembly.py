from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Optional, Sequence, Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
from lsdo_function_spaces.core.function import Function
from lsdo_function_spaces.core.function_set import FunctionSet

from lsdo_geo.core.geometry.geometry import Geometry
from lsdo_geo.core.geometry.marker import Marker


class Connection(lfs.CollocationCondition):
    """
    Geometric kinematic connection connecting two components at their respective markers.
    Specializes lsdo_function_spaces.CollocationCondition for 3D Euclidean kinematics.

    Parameters
    ----------
    component_1 : Union[Geometry, FunctionSet, Function]
        The primary moving component (e.g. rotor / articulated link).
    marker_1 : Marker
        The attachment marker on component_1.
    component_2 : Optional[Union[Geometry, FunctionSet, Function]]
        The secondary mounting component (e.g. base / mount). If None, component_1 is connected to world/inertial frame.
    marker_2 : Optional[Marker]
        The attachment marker on component_2.
    connection_type : str
        The type of connection: 'revolute', 'prismatic', 'spherical', 'fixed'. Default is 'revolute'.
    alignment_axis_1 : Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]]
        Optional custom axis provider. Defaults to marker_1.compute_surface_normal().
    alignment_axis_2 : Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]]
        Optional custom axis/null-space provider. Defaults to marker_2.compute_surface_normal().
    name : Optional[str]
        The name of the connection. If None, an auto-generated unique name is assigned.
    """
    _connection_counter: int = 0

    def __init__(
        self,
        component_1: Union[Geometry, FunctionSet, Function],
        marker_1: Marker,
        component_2: Optional[Union[Geometry, FunctionSet, Function]] = None,
        marker_2: Optional[Marker] = None,
        connection_type: str = "revolute",
        alignment_axis_1: Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]] = None,
        alignment_axis_2: Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]] = None,
        offset: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None,
        selectors: Optional[Tuple[slice, ...]] = None,
        derivative_orders: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        if name is None:
            Connection._connection_counter += 1
            name = f"connection_{Connection._connection_counter}"

        self.component_1 = component_1
        self.marker_1 = marker_1
        self.component_2 = component_2
        self.marker_2 = marker_2
        self.connection_type = connection_type
        self.alignment_axis_1 = alignment_axis_1
        self.alignment_axis_2 = alignment_axis_2

        # Wrap markers in an Interface
        markers = [marker_1] if marker_2 is None else [marker_1, marker_2]
        interface = lfs.Interface(markers=markers, name=f"{name}_interface")
        super().__init__(
            interface=interface,
            selectors=selectors,
            offset=offset,
            derivative_orders=derivative_orders,
            name=name,
        )

    def evaluate_translational_offset(
        self,
        geometry_1: Optional[Geometry] = None,
        geometry_2: Optional[Geometry] = None,
    ) -> csdl.Variable:
        """
        Computes the translational connection error: r_1 - r_2 - offset.
        For a physical connection constraint, this error must evaluate to zero.
        """
        p1 = self.marker_1.evaluate(geometry=geometry_1)
        if self.component_2 is not None and self.marker_2 is not None:
            p2 = self.marker_2.evaluate(geometry=geometry_2)
        elif self.marker_2 is not None:
            p2 = self.marker_2.evaluate()
        elif self.offset is not None:
            p2 = csdl.Variable(value=np.zeros(3))
        else:
            # Fixed world point from marker_1's initial coordinate
            p2 = self.marker_1.initial_point
            if p2 is None:
                raise ValueError(f"Connection {self.name} has no component_2 or marker_2, and marker_1 has no initial_point.")
            p2 = csdl.Variable(value=p2)
        diff = p1 - p2
        offset_val = None
        if self.offset is not None:
            if isinstance(self.offset, csdl.Variable):
                offset_val = self.offset
            else:
                offset_arr = np.atleast_1d(np.asarray(self.offset, dtype=np.float64))
                offset_val = csdl.Variable(value=offset_arr)

        if offset_val is not None and offset_val.shape == diff.shape:
            diff = diff - offset_val

        if self.selectors is not None:
            if isinstance(self.selectors, (list, tuple)):
                if len(self.selectors) == 1:
                    diff = diff[self.selectors[0]]
                else:
                    diff = csdl.concatenate([diff[i] for i in self.selectors])
            elif isinstance(self.selectors, slice):
                diff = diff[self.selectors]

        if offset_val is not None and self.selectors is not None and offset_val.shape != (3,) and offset_val.size == 1:
            diff = diff - offset_val
        return diff

    def evaluate_alignment_constraint(
        self,
        geometry_1: Optional[Geometry] = None,
        geometry_2: Optional[Geometry] = None,
    ) -> csdl.Variable:
        """
        Computes the 2D rotational alignment constraint:
        null_space_2^T @ axis_1 = [0, 0].
        
        Constrains axis_1 (on component_1) to remain parallel to axis_2 (on component_2).
        """
        # Axis 1 evaluation
        if self.alignment_axis_1 is not None:
            if hasattr(self.alignment_axis_1, "compute_surface_normal") and getattr(self.alignment_axis_1, "compute_surface_normal") is not None and getattr(self.alignment_axis_1, "parametric_coordinates", None) is not None:
                geom_1 = geometry_1 if geometry_1 is not None else getattr(self.component_1, "geometry", self.component_1)
                axis_1 = self.alignment_axis_1.compute_surface_normal(geometry=geom_1)
            elif hasattr(self.alignment_axis_1, "evaluate"):
                geom_1 = geometry_1 if geometry_1 is not None else getattr(self.component_1, "geometry", self.component_1)
                axis_1 = self.alignment_axis_1.evaluate(geom_1)
            else:
                axis_1 = self.alignment_axis_1
        else:
            axis_1 = self.marker_1.compute_surface_normal(geometry=geometry_1)

        # Axis 2 / Null space evaluation
        if self.alignment_axis_2 is not None:
            if hasattr(self.alignment_axis_2, "compute_null_space") and getattr(self.alignment_axis_2, "compute_null_space") is not None:
                geom_2 = geometry_2 if geometry_2 is not None else getattr(self.component_2, "geometry", self.component_2)
                null_space_2 = self.alignment_axis_2.compute_null_space(geometry=geom_2)
            elif hasattr(self.alignment_axis_2, "evaluate"):
                geom_2 = geometry_2 if geometry_2 is not None else getattr(self.component_2, "geometry", self.component_2)
                eval_res = self.alignment_axis_2.evaluate(geom_2)
                if isinstance(eval_res, tuple) and len(eval_res) == 2:
                    _, null_space_2 = eval_res
                else:
                    null_space_2 = eval_res
            else:
                null_space_2 = self.alignment_axis_2
        else:
            if self.marker_2 is not None:
                null_space_2 = self.marker_2.compute_null_space(geometry=geometry_2)
            else:
                # Fixed null space in world frame
                null_space_2 = self.marker_1.compute_null_space()

        # If null_space_2 is a 1D vector (3,), compute its 3x2 orthogonal basis
        if hasattr(null_space_2, "shape") and len(null_space_2.shape) == 1 and null_space_2.shape[0] == 3:
            has_val = hasattr(null_space_2, "value") and null_space_2.value is not None
            if has_val or not isinstance(null_space_2, csdl.Variable):
                axis_arr = np.asarray(null_space_2.value if has_val else null_space_2, dtype=np.float64)
                ref = np.array([1., 0., 0.]) if abs(axis_arr[0]) < 0.9 else np.array([0., 0., 1.])
                t1 = np.cross(axis_arr, ref)
                norm1 = np.linalg.norm(t1)
                t1 = t1 / norm1 if norm1 > 1e-12 else np.array([1., 0., 0.])
                t2 = np.cross(axis_arr, t1)
                norm2 = np.linalg.norm(t2)
                t2 = t2 / norm2 if norm2 > 1e-12 else np.array([0., 1., 0.])
                null_space_2 = csdl.Variable(value=np.column_stack([t1, t2]))
            else:
                ref = csdl.Variable(value=np.array([1., 0., 0.]))
                t1 = csdl.cross(null_space_2, ref)
                t1 = t1 / (csdl.norm(t1) + 1e-12)
                t2 = csdl.cross(null_space_2, t1)
                t2 = t2 / (csdl.norm(t2) + 1e-12)
                t1_col = csdl.reshape(t1, (3, 1))
                t2_col = csdl.reshape(t2, (3, 1))
                null_space_2 = csdl.concatenate([t1_col, t2_col], axis=1)

        # Alignment projection (must be [0, 0] for revolute alignment)
        alignment_constraint = csdl.matvec(csdl.transpose(null_space_2), axis_1)
        return alignment_constraint

    def evaluate_residual(self) -> csdl.Variable:
        """
        Evaluates the combined kinematic residual vector based on connection_type.
        """
        if self.connection_type in ("spherical", "translational", "fixed"):
            return self.evaluate_translational_offset()
        elif self.connection_type == "prismatic":
            return self.evaluate_alignment_constraint()
        elif self.connection_type == "revolute":
            trans = self.evaluate_translational_offset()
            align = self.evaluate_alignment_constraint()
            return csdl.concatenate([trans, align])
        elif self.connection_type == "fixed":
            return self.evaluate_translational_offset()
        else:
            return self.evaluate_translational_offset()

    def rebind(self, target_map: Dict[Any, Any]) -> Connection:
        """
        Rebinds the connection to new components and markers according to target_map.

        Parameters
        ----------
        target_map : Dict[Any, Any]
            Mapping from old components/functions/markers to new components/functions/markers.

        Returns
        -------
        Connection
            Rebound connection instance.
        """
        # Resolve component_1
        c1 = self.component_1
        new_c1 = target_map.get(c1, c1)

        # Resolve component_2
        c2 = self.component_2
        new_c2 = target_map.get(c2, c2) if c2 is not None else None

        # Resolve marker_1
        m1 = self.marker_1
        if m1 in target_map:
            new_m1 = target_map[m1]
        elif hasattr(new_c1, "markers") and m1.name in new_c1.markers:
            new_m1 = new_c1.markers[m1.name]
        elif hasattr(m1, "copy") and new_c1 is not c1:
            new_m1 = m1.copy(geometry=new_c1)
        else:
            new_m1 = m1

        # Resolve marker_2
        m2 = self.marker_2
        if m2 is not None:
            if m2 in target_map:
                new_m2 = target_map[m2]
            elif new_c2 is not None and hasattr(new_c2, "markers") and m2.name in new_c2.markers:
                new_m2 = new_c2.markers[m2.name]
            elif hasattr(m2, "copy") and new_c2 is not None and new_c2 is not c2:
                new_m2 = m2.copy(geometry=new_c2)
            else:
                new_m2 = m2
        else:
            new_m2 = None

        # Resolve alignment axes
        axis_1 = self.alignment_axis_1
        if axis_1 is not None:
            if axis_1 in target_map:
                new_axis_1 = target_map[axis_1]
            elif hasattr(new_c1, "markers") and hasattr(axis_1, "name") and axis_1.name in new_c1.markers:
                new_axis_1 = new_c1.markers[axis_1.name]
            elif hasattr(axis_1, "copy") and new_c1 is not c1:
                new_axis_1 = axis_1.copy(geometry=new_c1)
            else:
                new_axis_1 = axis_1
        else:
            new_axis_1 = None

        axis_2 = self.alignment_axis_2
        if axis_2 is not None:
            if axis_2 in target_map:
                new_axis_2 = target_map[axis_2]
            elif new_c2 is not None and hasattr(new_c2, "markers") and hasattr(axis_2, "name") and axis_2.name in new_c2.markers:
                new_axis_2 = new_c2.markers[axis_2.name]
            elif hasattr(axis_2, "copy") and new_c2 is not None and new_c2 is not c2:
                new_axis_2 = axis_2.copy(geometry=new_c2)
            else:
                new_axis_2 = axis_2
        else:
            new_axis_2 = None

        return Connection(
            component_1=new_c1,
            marker_1=new_m1,
            component_2=new_c2,
            marker_2=new_m2,
            connection_type=self.connection_type,
            alignment_axis_1=new_axis_1,
            alignment_axis_2=new_axis_2,
            offset=self.offset,
            selectors=self.selectors,
            derivative_orders=self.derivative_orders,
            name=self.name,
        )

    def copy(self, target_map: Optional[Dict[Any, Any]] = None) -> Connection:
        """
        Creates a copy of the connection, optionally rebinding to new components/markers.
        """
        return self.rebind(target_map if target_map is not None else {})


class FixedConnection(Connection):
    """
    Fixed kinematic connection locking relative translation (and orientation).
    """
    def __init__(
        self,
        component_1: Union[Geometry, FunctionSet, Function],
        marker_1: Marker,
        component_2: Optional[Union[Geometry, FunctionSet, Function]] = None,
        marker_2: Optional[Marker] = None,
        offset: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None,
        selectors: Optional[Tuple[slice, ...]] = None,
        derivative_orders: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            component_1=component_1,
            marker_1=marker_1,
            component_2=component_2,
            marker_2=marker_2,
            connection_type="fixed",
            offset=offset,
            selectors=selectors,
            derivative_orders=derivative_orders,
            name=name,
        )


class SphericalConnection(Connection):
    """
    Spherical / ball joint connection constraining 3 relative translations while allowing 3D rotation.
    """
    def __init__(
        self,
        component_1: Union[Geometry, FunctionSet, Function],
        marker_1: Marker,
        component_2: Optional[Union[Geometry, FunctionSet, Function]] = None,
        marker_2: Optional[Marker] = None,
        offset: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None,
        selectors: Optional[Tuple[slice, ...]] = None,
        derivative_orders: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            component_1=component_1,
            marker_1=marker_1,
            component_2=component_2,
            marker_2=marker_2,
            connection_type="spherical",
            offset=offset,
            selectors=selectors,
            derivative_orders=derivative_orders,
            name=name,
        )


class RevoluteConnection(Connection):
    """
    Revolute / pin joint connection constraining 3 relative translations and 2 relative rotations,
    allowing 1 degree of freedom rotation around a specified joint axis.
    """
    def __init__(
        self,
        component_1: Union[Geometry, FunctionSet, Function],
        marker_1: Marker,
        component_2: Optional[Union[Geometry, FunctionSet, Function]] = None,
        marker_2: Optional[Marker] = None,
        axis: Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]] = None,
        alignment_axis_1: Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]] = None,
        alignment_axis_2: Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]] = None,
        offset: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None,
        selectors: Optional[Tuple[slice, ...]] = None,
        derivative_orders: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        if axis is not None:
            if alignment_axis_1 is None:
                alignment_axis_1 = axis
            if alignment_axis_2 is None:
                alignment_axis_2 = axis

        super().__init__(
            component_1=component_1,
            marker_1=marker_1,
            component_2=component_2,
            marker_2=marker_2,
            connection_type="revolute",
            alignment_axis_1=alignment_axis_1,
            alignment_axis_2=alignment_axis_2,
            offset=offset,
            selectors=selectors,
            derivative_orders=derivative_orders,
            name=name,
        )


PinConnection = RevoluteConnection


class PrismaticConnection(Connection):
    """
    Prismatic / slider connection allowing 1 relative translation along a guide axis
    while constraining transverse translations and relative rotations.
    """
    def __init__(
        self,
        component_1: Union[Geometry, FunctionSet, Function],
        marker_1: Marker,
        component_2: Optional[Union[Geometry, FunctionSet, Function]] = None,
        marker_2: Optional[Marker] = None,
        guide_axis: Optional[Union[Marker, csdl.Variable, npt.NDArray[np.float64]]] = None,
        offset: Optional[Union[csdl.Variable, npt.NDArray[np.float64]]] = None,
        selectors: Optional[Tuple[slice, ...]] = None,
        derivative_orders: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            component_1=component_1,
            marker_1=marker_1,
            component_2=component_2,
            marker_2=marker_2,
            connection_type="prismatic",
            alignment_axis_1=guide_axis,
            alignment_axis_2=guide_axis,
            offset=offset,
            selectors=selectors,
            derivative_orders=derivative_orders,
            name=name,
        )


class Assembly(lfs.FunctionNetwork):
    """
    Geometric Assembly representing a graph of components and kinematic connections.
    Inherits from lsdo_function_spaces.FunctionNetwork.
    """
    def __init__(
        self,
        name: str = "assembly",
        components: Optional[Dict[str, Union[Geometry, FunctionSet, Function]]] = None,
        connections: Optional[Dict[str, Connection]] = None,
    ):
        super().__init__(name=name)
        self.components: Dict[str, Union[Geometry, FunctionSet, Function]] = components if components is not None else {}
        self.connections: Dict[str, Connection] = connections if connections is not None else {}

    def add_component(
        self,
        component: Union[Geometry, FunctionSet, Function],
        name: Optional[str] = None,
    ) -> Union[Geometry, FunctionSet, Function]:
        """
        Adds a geometry component to the assembly.

        Parameters
        ----------
        component : Union[Geometry, FunctionSet, Function]
            The geometry component to register.
        name : Optional[str]
            Optional identifier name for the component. Defaults to component.name.

        Returns
        -------
        Union[Geometry, FunctionSet, Function]
            The registered component.
        """
        # Support legacy order (name, component) if passed
        if isinstance(component, str) and not isinstance(name, str) and name is not None:
            component, name = name, component

        if name is None:
            name = getattr(component, "name", None)
            if name is None:
                name = f"component_{len(self.components) + 1}"
        self.components[name] = component
        if hasattr(component, "geometry") and component.geometry is not None:
            self.add_function(component.geometry)
        elif hasattr(component, "functions") or hasattr(component, "space"):
            self.add_function(component)
        return component

    def add_connection(self, connection: Union[Connection, lfs.InterfaceCondition]) -> Connection:
        """
        Adds a kinematic Connection or InterfaceCondition to the assembly.

        Parameters
        ----------
        connection : Union[Connection, lfs.InterfaceCondition]
            The kinematic connection or interface condition to register.

        Returns
        -------
        Connection
            The registered connection.
        """
        if hasattr(connection, "connection") and isinstance(connection.connection, Connection):
            actual_conn = connection.connection
        elif isinstance(connection, Connection):
            actual_conn = connection
        elif hasattr(connection, "interface"):
            actual_conn = connection
        else:
            raise TypeError(f"Expected Connection or InterfaceCondition, got {type(connection)}")
        self.connections[actual_conn.name] = actual_conn
        self.add_condition(actual_conn)
        return actual_conn

    def get_connection(self, name: str) -> Optional[Connection]:
        return self.connections.get(name)

    def get_component(self, name: str) -> Optional[Union[Geometry, FunctionSet, Function]]:
        return self.components.get(name)

    def rebind(
        self,
        target_map: Optional[Dict[Any, Any]] = None,
        name: Optional[str] = None,
    ) -> Assembly:
        """
        Rebinds assembly components, connections, and underlying network topology.

        Parameters
        ----------
        target_map : Optional[Dict[Any, Any]], optional
            Mapping from old components, geometries, functions, or markers to new entities.
            May also map component names (str) to new components.
        name : Optional[str], optional
            Optional new name for the rebound assembly. Defaults to self.name.

        Returns
        -------
        Assembly
            New Assembly instance bound to the target entities.
        """
        target_map = dict(target_map) if target_map is not None else {}
        new_assembly = Assembly(name=name if name is not None else self.name)

        # 1. Rebind components
        for cname, comp in self.components.items():
            if comp in target_map:
                new_comp = target_map[comp]
            elif cname in target_map:
                new_comp = target_map[cname]
                target_map[comp] = new_comp
            else:
                new_comp = comp
                target_map[comp] = new_comp

            new_assembly.add_component(new_comp, name=cname)

            # If component has underlying geometry, ensure it's in target_map
            if hasattr(comp, "geometry") and hasattr(new_comp, "geometry"):
                target_map[comp.geometry] = new_comp.geometry

            # Map markers between old and new components if both have markers
            if hasattr(comp, "markers") and hasattr(new_comp, "markers"):
                for mname, old_m in comp.markers.items():
                    if mname in new_comp.markers:
                        target_map[old_m] = new_comp.markers[mname]

        # 2. Rebind connections
        for conn_name, conn in self.connections.items():
            new_conn = conn.rebind(target_map)
            new_assembly.add_connection(new_conn)

        # 3. Rebind any additional conditions that are not in self.connections
        conn_conditions = set(self.connections.values())
        for cond in self.conditions:
            if cond not in conn_conditions:
                new_cond = cond.rebind(target_map)
                new_assembly.add_condition(new_cond)

        # 4. Rebind component groupings
        new_groupings = {}
        for gname, patches in self.component_groupings.items():
            new_patches = []
            for target, patch_id in patches:
                new_target = target_map.get(target, target)
                new_patches.append((new_target, patch_id))
            new_groupings[gname] = new_patches
        new_assembly.component_groupings = new_groupings

        return new_assembly

    def copy(self, name: Optional[str] = None) -> Assembly:
        """
        Creates a deep copy of the assembly, cloning all registered geometry components,
        rebinding attached markers, and recreating all connections.

        Parameters
        ----------
        name : Optional[str], optional
            Optional new name for the copied assembly. Defaults to self.name.

        Returns
        -------
        Assembly
            Deep copy of the assembly.
        """
        target_map = {}
        for cname, comp in self.components.items():
            if hasattr(comp, "copy"):
                target_map[comp] = comp.copy()
            else:
                target_map[comp] = comp
        return self.rebind(target_map, name=name)

    def discover_internal_seams(
        self,
        geometry: Optional[Union[Geometry, FunctionSet, str]] = None,
        tolerance: float = 1e-5,
        search_n: int = 7,
        create_conditions: bool = True,
    ) -> List[lfs.Interface]:
        """
        Automatically discovers internal patch seams and registers them as interfaces and conditions.

        Parameters
        ----------
        geometry : Optional[Union[Geometry, FunctionSet, str]], optional
            The component or geometry to search for internal seams. If a string is provided,
            it resolves from registered components. If None, discovers seams across all registered
            multi-patch components in the assembly.
        tolerance : float, optional
            Maximum physical distance between sampling points to consider patches adjacent.
            Default is 1e-5.
        search_n : int, optional
            Number of sampling points per boundary edge during discovery. Default is 7.
        create_conditions : bool, optional
            If True, automatically creates and registers CollocationCondition instances
            for each discovered seam. Default is True.

        Returns
        -------
        List[Interface]
            List of discovered Interface objects.
        """
        if isinstance(geometry, str):
            comp = self.components.get(geometry)
            if comp is None:
                raise KeyError(f"Component '{geometry}' not found in assembly.")
            geometry = comp

        if geometry is not None:
            target_entity = getattr(geometry, "geometry", geometry)
            return super().discover_internal_seams(
                entity=target_entity,
                tolerance=tolerance,
                search_n=search_n,
                create_conditions=create_conditions,
            )

        all_interfaces = []
        for comp in self.components.values():
            target_entity = getattr(comp, "geometry", comp)
            if hasattr(target_entity, "functions") and len(target_entity.functions) > 1:
                discovered = super().discover_internal_seams(
                    entity=target_entity,
                    tolerance=tolerance,
                    search_n=search_n,
                    create_conditions=create_conditions,
                )
                all_interfaces.extend(discovered)
        return all_interfaces

    def update_representations(self, plot: bool = False) -> None:
        """
        Updates markers and representations across all registered components in the assembly.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the representations during update, by default False.
        """
        for comp in self.components.values():
            if hasattr(comp, "update_representations"):
                comp.update_representations(plot=plot)
            elif hasattr(comp, "markers") and isinstance(comp.markers, dict):
                for m in comp.markers.values():
                    m.evaluate(comp, plot=plot)
