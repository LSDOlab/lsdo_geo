import numpy as np
import numpy.typing as npt
from typing import Union, Optional, Sequence
import scipy.sparse as sps
import lsdo_function_spaces as lfs
import csdl_alpha as csdl

from dataclasses import dataclass

from lsdo_geo.core.geometry.geometry_functions import rotate

# class NumpyAxis:
#     """
#     Class for representing an axis as a numpy array. This is used for specifying the axis of rotation or translation for sectional parameters.
#     """

#     def __init__(self, axis: npt.NDArray[np.float64]):
#         self.axis = axis
#     # axis: npt.NDArray[np.float64]

@dataclass
class SectionalParameters:
    """
    Dataclass of inputs for the VolumeSectionalParameterization class. This is passed into the evaluate method.
    The desired parameters should be appended to the appropiate dictionary.

    Parameters
    ----------
    stretches : list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]] = None
        The stretches for each section. 
        The first tuple element is the axis of the stretch. An integer axis of 0,1,2,... corresponds to the parametric axes, u,v,w,...
        Alternatively, a csdl variable or numpy array can be passed in to specify the axis.
        The second tuple element is the variable that contains the stretch values.
    translations : list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]] = None
        The translations for each section.
        The first tuple element is the axis of the translation. An integer axis of 0,1,2,... corresponds to the parametric axes, u,v,w,...
        Alternatively, a csdl variable or numpy array can be passed in to specify the axis.
        The second tuple element is the variable that contains the translation values.
    rotations : list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]] = None
        The rotations for each section. The first tuple element is the axis of the rotation.
        The second tuple element is the variable that contains the rotation values.
    """

    translations : Optional[list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]]] = None
    stretches : Optional[list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]]] = None
    rotations : Optional[list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]]] = None

    def __post_init__(self):
        if self.translations is None:
            self.translations = []
        if self.stretches is None:
            self.stretches = []
        if self.rotations is None:
            self.rotations = []

    def add_translation(self, axis: Union[int, csdl.Variable], translation: Union[csdl.Variable, npt.NDArray[np.float64]]):
        """
        Adds a translation to the translations dictionary.

        Parameters
        ----------
        axis : Union[int, csdl.Variable]
            The axis of the translation. integer axes of 0,1,2,... correspond to the parametric axes, u,v,w,...
            Alternatively, a csdl variable can be passed in to specify the axis.
        translation : Union[csdl.Variable, npt.NDArray[np.float64]]
            The translation values.
        """
        if self.translations is None:
            self.translations = []
        self.translations.append((axis, translation))

    def add_stretch(self, axis: int, stretch: Union[csdl.Variable, npt.NDArray[np.float64]], parametric_coordinate:Optional[npt.NDArray[np.float64]]=None):
        """
        Adds a stretch to the stretches dictionary.

        Parameters
        ----------
        axis : int
            The axis of the stretch.
        stretch : Union[csdl.Variable, npt.NDArray[np.float64]]
            The stretch values.
        parametric_coordinate : Optional[npt.NDArray[np.float64]]
            The parametric coordinate to stretch relative to.
        """
        if self.stretches is None:
            self.stretches = []
        self.stretches.append((axis, stretch, parametric_coordinate))

    def add_rotation(self, axis: Union[int, csdl.Variable, npt.NDArray[np.float64]], rotation: Union[csdl.Variable, npt.NDArray[np.float64]], parametric_coordinate:Optional[npt.NDArray[np.float64]]=None):
        """
        Adds a rotation to the rotations dictionary.

        Parameters
        ----------
        axis : Union[int, csdl.Variable, npt.NDArray[np.float64]]
            The axis of the rotation. integer axes of 0,1,2,... correspond to the parametric axes, u,v,w,...
            Alternatively, a csdl variable can be passed in to specify the axis.
        rotation : Union[csdl.Variable, npt.NDArray[np.float64]]
            The rotation values.
        parametric_coordinate : Optional[npt.NDArray[np.float64]]
            The parametric coordinate to rotate about.
        """
        if self.rotations is None:
            self.rotations = []
        self.rotations.append((axis, rotation, parametric_coordinate))


@dataclass
class SectionalParameterization:
    """
    Class for parameterizing volumes by specifying a principal parametric dimension and perceiving the volume
    as a series of sections along the axis.

    Parameters
    ----------
    parameterized_points : csdl.Variable
        The points to parameterize. The points should be in a structured shape.
    principal_parametric_dimension : int = 0
        The principal parametric dimension of the parameterized points. The sections are along this axis (axis is normal to sections).
    linear_parameter_maps : dict[str,sps.csc_matrix] = None
        The linear maps from the sectional parameters to the parameterized points.
    rotational_axes : dict[str,int] = None
        The axes to rotate about.
    name : str = 'volume_sectional_parameterization'
    """

    parameterized_points: csdl.Variable
    principal_parametric_dimension: int = 0
    parameterized_points_shape: Optional[tuple[int,...]] = None
    # linear_parameter_maps: Optional[dict[tuple[str, Union[int, csdl.Variable]], sps.csc_matrix]] = None
    # rotational_axes: Optional[dict[Union[str, csdl.Variable], Union[int, csdl.Variable]]] = None
    translations : Optional[list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]]] = None
    stretches : Optional[list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]]] = None
    rotations : Optional[list[tuple[Union[int, csdl.Variable, npt.NDArray[np.float64]], Union[csdl.Variable, npt.NDArray[np.float64]]]]] = None
    name : str = 'volume_sectional_parameterization'

    def __post_init__(self):
        if self.parameterized_points_shape is None:
            self.parameterized_points_shape = self.parameterized_points.shape
        elif np.prod(self.parameterized_points_shape) != np.prod(
            self.parameterized_points.shape
        ):
            raise Exception("Specified shape and points shape are not the same size.")

        if len(self.parameterized_points_shape) == 1:
            raise Exception(
                "Please pass in parameterized_points with structured shape or pass in parameterized_points_shape."
            )
        elif len(self.parameterized_points_shape) == 2:
            raise Exception(
                "Can't make a sectional parameterization for a 1D set of points."
                + "Check shape and make sure physical dimensions are long the last axis."
            )

        if (
            self.principal_parametric_dimension
            >= len(self.parameterized_points_shape) - 1
        ):
            raise Exception(
                "Principal parametric dimension is greater than the number of parametric dimensions in the parameterized points."
            )

        self.num_sections = self.parameterized_points_shape[
            self.principal_parametric_dimension
        ]
        self.num_points_per_section = (
            np.prod(self.parameterized_points_shape[:-1]) // self.num_sections
        )
        self.num_physical_dimensions = self.parameterized_points_shape[-1]

        fitting_points = self.parameterized_points.value.reshape(
            self.parameterized_points_shape
        )

        # Use points to create a B-spline to help with getting axes
        
        helpful_b_spline_space = lfs.BSplineSpaceNew(num_parametric_dimensions=len(self.parameterized_points_shape[:-1]),
                                                  degree=1, coefficients_shape=self.parameterized_points_shape[:-1])
        fitting_parametric_values = helpful_b_spline_space.generate_parametric_grid(grid_resolution=self.parameterized_points_shape[:-1])
        self.helpful_b_spline = helpful_b_spline_space.fit_function(values=fitting_points, parametric_coordinates=fitting_parametric_values)
        # self.helpful_b_spline = lfs.fit_b_spline(
        #     fitting_points=fitting_points,
        #     order=(2,),
        #     num_coefficients=self.parameterized_points_shape[:-1],
        #     name="helpful_b_spline",
        # )

        self.parameterized_points_shape_without_principal_dimension = tuple(
            dim for i, dim in enumerate(self.parameterized_points_shape) if i != self.principal_parametric_dimension
        )
        self.helpful_section_b_spline_space = lfs.BSplineSpaceNew(
            num_parametric_dimensions=len(self.parameterized_points_shape_without_principal_dimension)-1,
            degree=1,
            coefficients_shape=self.parameterized_points_shape_without_principal_dimension[:-1])

        self.sectional_principal_parametric_coordinate = np.linspace(
            0.0, 1.0, self.num_sections
        ).reshape((-1, 1))

        # self.linear_parameter_maps = {}
        # self.rotational_axes = {}

        # self.updated_points = self.parameterized_points # NOTE: Removing .copy() here because csdl doesn't have one.

    # def add_parameter(self, parameter_type: str, axis: Union[int, csdl.Variable], map: sps.csc_matrix):
    #     """
    #     Adds a sectional parameter to the parameterization. The map should map from the parameter vector to deltas in the parameterized points.

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the sectional parameter.
    #     map : sps.csc_matrix
    #         The map from the parameter vector to deltas in the parameterized points.
    #     """
    #     self.linear_parameter_maps[(parameter_type, axis)] = map

    # def add_translation(self, axis: Union[int, csdl.Variable]):
    #     """
    #     Adds a sectional translation parameter to the parameterization.

    #     Parameters
    #     ----------
    #     axis : int
    #         The axis to translate along.
    #     """
    #     if isinstance(axis, int):
    #         valid_axes = np.arange(len(self.parameterized_points_shape) - 1)
    #         if axis not in valid_axes:
    #             raise Exception(f"Please pass in a valid axis. valid axes:{valid_axes}")
    #     elif isinstance(axis, csdl.Variable):
    #         # Check to make sure the axis is valid
    #         if axis.shape != (self.parameterized_points_shape[-1],):
    #             raise Exception(f"Invalid axis shape. Expected: {(self.parameterized_points_shape[-1],)}, Got: {axis.shape}")

    #     num_outputs = np.prod(self.parameterized_points_shape)

    #     parameter_map_list = []
    #     for i in range(self.num_sections):
    #         parameter_section_map = sps.lil_matrix((num_outputs, 1))

    #         parametric_coordinate = (
    #             np.ones((len(self.parameterized_points_shape[:-1]))) * 0.5
    #         )
    #         parametric_coordinate[self.principal_parametric_dimension] = (
    #             self.sectional_principal_parametric_coordinate[i].reshape((1, -1))
    #         )

    #         if isinstance(axis, int):
    #             parametric_derivative_order = np.zeros(
    #                 (len(self.parameterized_points_shape[:-1]))
    #             )
    #             parametric_derivative_order[axis] = 1
    #             parametric_derivative_order = tuple(parametric_derivative_order)
    #             translation_axis = self.helpful_b_spline.evaluate(
    #                 parametric_coordinates=parametric_coordinate,
    #                 parametric_derivative_orders=parametric_derivative_order,
    #                 non_csdl=True
    #             )
    #             translation_axis /= np.linalg.norm(translation_axis)
    #         else:
    #             translation_axis = axis.value

    #         indices = np.arange(np.prod(self.parameterized_points_shape, dtype=int))
    #         indices = indices.reshape(self.parameterized_points_shape)
    #         indices = np.swapaxes(indices, 0, self.principal_parametric_dimension)
    #         indices = indices[i].reshape((-1,))

    #         parameter_section_map[indices] = np.tile(
    #             translation_axis, self.num_points_per_section
    #         ).reshape((-1, 1))

    #         parameter_map_list.append(parameter_section_map)

    #     parameter_map = sps.hstack(parameter_map_list).tocsc()

    #     self.add_parameter(parameter_type='sectional_translation', axis=axis, map=parameter_map)

    # def add_sectional_stretch(self, axis: int):
    #     """
    #     Adds a sectional stretch parameter to the parameterization.

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the sectional stretch parameter.
    #     axis : int
    #         The axis to stretch along.
    #     """
    #     valid_axes = np.delete(
    #         np.arange(len(self.parameterized_points_shape) - 1),
    #         self.principal_parametric_dimension,
    #     )
    #     if axis not in valid_axes:
    #         raise Exception(f"Please pass in a valid axis. valid axes:{valid_axes}. Warning: Can't stretch along principal axis."
    #                         + "You probably either want to fix your principal axis or instead translate along this axis.")

    #     num_outputs = np.prod(self.parameterized_points_shape)

    #     parameter_map_list = []
    #     for i in range(self.num_sections):
    #         parameter_section_map = sps.lil_matrix((num_outputs, 1))

    #         if axis == self.principal_parametric_dimension:
    #             raise Exception(
    #                 "Can't stretch along the principal parametric dimension because sections have no thickness to stretch."
    #                 + "Use a linear distributions of translations instead"
    #             )

    #         parametric_coordinate = (
    #             np.ones((len(self.parameterized_points_shape[:-1]))) * 0.5
    #         )
    #         parametric_coordinate[self.principal_parametric_dimension] = (
    #             self.sectional_principal_parametric_coordinate[i].reshape((1, -1))
    #         )
    #         parametric_derivative_order = np.zeros(
    #             (len(self.parameterized_points_shape[:-1]))
    #         )
    #         parametric_derivative_order[axis] = 1
    #         parametric_derivative_order = tuple(parametric_derivative_order)
    #         stretch_axis = self.helpful_b_spline.evaluate(
    #             parametric_coordinates=parametric_coordinate,
    #             parametric_derivative_orders=parametric_derivative_order,
    #             non_csdl=True
    #         )
    #         stretch_axis /= np.linalg.norm(stretch_axis)
    #         section_middle = self.helpful_b_spline.evaluate(
    #             parametric_coordinates=parametric_coordinate,
    #             parametric_derivative_orders=(0,),
    #             non_csdl=True
    #         )

    #         section_axis_end_parametric_coordinate = parametric_coordinate.copy()
    #         section_axis_end_parametric_coordinate[axis] = 1.0
    #         section_axis_beginning_parametric_coordinate = parametric_coordinate.copy()
    #         section_axis_beginning_parametric_coordinate[axis] = 0.0

    #         section_axis_end = self.helpful_b_spline.evaluate(
    #             parametric_coordinates=section_axis_end_parametric_coordinate,
    #             parametric_derivative_orders=(0,),
    #             non_csdl=True
    #         )
    #         section_axis_beginning = self.helpful_b_spline.evaluate(
    #             parametric_coordinates=section_axis_beginning_parametric_coordinate,
    #             parametric_derivative_orders=(0,),
    #             non_csdl=True
    #         )
    #         section_length = (section_axis_end - section_axis_beginning).dot(
    #             stretch_axis
    #         )

    #         indices = np.arange(np.prod(self.parameterized_points_shape, dtype=int))
    #         indices = indices.reshape(self.parameterized_points_shape)
    #         indices = np.moveaxis(indices, self.principal_parametric_dimension, 0)
    #         indices = indices.reshape(
    #             (
    #                 self.num_sections,
    #                 self.num_points_per_section,
    #                 self.num_physical_dimensions,
    #             )
    #         )

    #         for j in range(self.num_points_per_section):
    #             point_indices_full_shape = _get_indices_in_shape(
    #                 shape=self.parameterized_points_shape[:-1],
    #                 total_index=j,
    #                 section_axis=self.principal_parametric_dimension,
    #                 section_axis_index=i,
    #             )
    #             displacement = (
    #                 self.parameterized_points.value.reshape(
    #                     self.parameterized_points_shape
    #                 )[point_indices_full_shape]
    #                 - section_middle
    #             )
    #             distance_along_axis = np.dot(displacement, stretch_axis)

    #             point_indices = indices[i, j, :].reshape((-1,))
    #             parameter_section_map[point_indices] = (
    #                 distance_along_axis
    #                 / section_length
    #                 * stretch_axis.reshape((self.num_physical_dimensions, 1))
    #             )

    #         parameter_map_list.append(parameter_section_map)

    #     parameter_map = sps.hstack(parameter_map_list).tocsc()

    #     self.add_parameter(parameter_type='sectional_stretch', axis=axis, map=parameter_map)

    # def add_sectional_rotation(self, axis: Union[int, csdl.Variable, npt.NDArray[np.float64]]):
    #     """
    #     Adds a sectional rotation parameter to the parameterization.

    #     Parameters
    #     ----------
    #     name : str
    #         The name of the sectional rotation parameter.
    #     axis : int
    #         The axis to rotate about.
    #     """
    #     if isinstance(axis, int):
    #         valid_axes = np.arange(len(self.parameterized_points_shape) - 1)
    #         if axis not in valid_axes:
    #             raise Exception(f"Please pass in a valid axis. valid axes:{valid_axes}")

    #     if isinstance(axis, np.ndarray):
    #         axis_label = NumpyAxis(axis)
    #     else:
    #         axis_label = axis
    #     self.rotational_axes[axis_label] = axis

    # def _compute_axis_from_b_spline(self, axis: int, parametric_coordinate: npt.NDArray[np.float64]):
    #     parametric_derivative_order = np.zeros(
    #         (len(self.parameterized_points_shape[:-1])), dtype=int
    #     )
    #     parametric_derivative_order[axis] = 1
    #     parametric_derivative_order = tuple(parametric_derivative_order)
    #     axis_vector = self.helpful_b_spline.evaluate(
    #         parametric_coordinates=parametric_coordinate,
    #         parametric_derivative_orders=parametric_derivative_order,
    #         non_csdl=True
    #     )
    #     axis_vector /= np.linalg.norm(axis_vector)
    #     return axis_vector

    def _compute_section_axis(self, section_points: csdl.Variable, parametric_dimension: int, parametric_coordinate: npt.NDArray[np.float64]) -> csdl.Variable:
        section_b_spline = lfs.Function(space=self.helpful_section_b_spline_space, coefficients=section_points)
        if parametric_dimension == self.principal_parametric_dimension:
            # Compute normal vector by taking the cross product of the two non-principal parametric derivative directions.
            non_principal_parametric_dimensions = [i for i in range(len(self.parameterized_points_shape[:-1])) if i != self.principal_parametric_dimension]
            parametric_derivative_orders_1 = np.zeros((len(non_principal_parametric_dimensions)), dtype=int)
            parametric_derivative_orders_1[non_principal_parametric_dimensions[0]] = 1
            parametric_derivative_orders_2 = np.zeros((len(non_principal_parametric_dimensions)), dtype=int)
            parametric_derivative_orders_2[non_principal_parametric_dimensions[1]] = 1
            parametric_derivative_orders_1 = tuple(parametric_derivative_orders_1)
            parametric_derivative_orders_2 = tuple(parametric_derivative_orders_2)
            derivative_1 = section_b_spline.evaluate(parametric_coordinates=parametric_coordinate, parametric_derivative_orders=parametric_derivative_orders_1)
            derivative_2 = section_b_spline.evaluate(parametric_coordinates=parametric_coordinate, parametric_derivative_orders=parametric_derivative_orders_2)
            axis = csdl.cross(derivative_1, derivative_2).flatten()
        else:
            parametric_derivative_orders = np.zeros((len(self.parameterized_points_shape_without_principal_dimension[:-1]),), dtype=int)
            if parametric_dimension < self.principal_parametric_dimension:
                parametric_derivative_orders[parametric_dimension] = 1
            else:
                parametric_derivative_orders[parametric_dimension-1] = 1
            parametric_derivative_orders = tuple(parametric_derivative_orders)
            axis = section_b_spline.evaluate(parametric_coordinates=parametric_coordinate, parametric_derivative_orders=tuple(parametric_derivative_orders)).flatten()

        axis = axis/csdl.norm(axis)
        return axis
    
    def _compute_section_origin(self, section_points: csdl.Variable, parametric_dimension: int, parametric_coordinate: npt.NDArray[np.float64], non_csdl: bool = False) -> csdl.Variable:
        section_b_spline = lfs.Function(space=self.helpful_section_b_spline_space, coefficients=section_points)
        origin = section_b_spline.evaluate(parametric_coordinates=parametric_coordinate, non_csdl=non_csdl).flatten()
        return origin

    # def evaluate(self, sectional_parameters:dict[str,csdl.Variable], plot:bool=False) -> csdl.Variable:
    def evaluate(
        self,
        sectional_parameters: SectionalParameters,
        plot: bool = False,
    ) -> csdl.Variable:
        """
        Applies the parameters to each section.

        Parameters
        ----------
        section_parameters : SectionalParameters
            An object (imported from this file) that stores all of the sectional parameters to apply.
            The parameters should be added to the object using the add_stretch, add_translation, and add_rotation methods.
        plot : bool = False
            Whether or not to plot the parameterized points after evaluation.

        Returns
        -------
        updated_points : csdl.Variable
            The updated points.
        """
        # Swap principal dimension to the front for easier indexing.
        axis_symbols = "abcdefghijklmnopqrstuvwxyz"
        num_axes = len(self.parameterized_points.shape)
        if num_axes > len(axis_symbols):
            raise Exception(
                f"Can't build axis-reorder action for {num_axes} axes."
            )

        input_axes = axis_symbols[:num_axes]
        principal_axis = input_axes[self.principal_parametric_dimension]
        non_principal_parametric_axes = "".join(
            axis
            for i, axis in enumerate(input_axes[:-1])
            if i != self.principal_parametric_dimension
        )
        physical_dimensions_axis = input_axes[-1]
        output_axes = (
            principal_axis
            + non_principal_parametric_axes
            + physical_dimensions_axis
        )
        points_principal_first = csdl.reorder_axes(
            self.parameterized_points,
            action=f"{input_axes}->{output_axes}",
        )
        section_num_axes = len(points_principal_first.shape) - 1
        expand_source_axis = axis_symbols[0]
        expand_target_axes = axis_symbols[1:section_num_axes] + expand_source_axis
        translation_expand_action = f"{expand_source_axis}->{expand_target_axes}"

        parameter_vectorization_target_axes = axis_symbols[:len(points_principal_first.shape)]
        # stretch_expand_action = f"{expand_source_axis}->{stretch_expand_target_axes}"
        parameter_vectorization_action = f"{expand_source_axis}->{parameter_vectorization_target_axes}"

        # Apply stretches
        for axis_input, parameter, parametric_coordinate in sectional_parameters.stretches:
            # for i in csdl.frange(self.num_sections):
            stretch_basis_vectors = []
            for i in range(self.num_sections):
                if isinstance(axis_input, int):
                    if parametric_coordinate is None:
                        parametric_coordinate = np.ones((len(self.parameterized_points_shape_without_principal_dimension[:-1]),)) * 0.5
                    axis = self._compute_section_axis(section_points=points_principal_first[i], parametric_dimension=axis_input, parametric_coordinate=parametric_coordinate)
                elif isinstance(axis_input, csdl.Variable):
                    axis = axis_input/csdl.norm(axis_input)
                elif isinstance(axis_input, np.ndarray):
                    axis = axis_input/np.linalg.norm(axis_input)
                else:
                    raise Exception(f"Invalid axis type: {type(axis)}. Axis should be either an int, a csdl.Variable, or a numpy array.")
            
                if parametric_coordinate is None:
                    parametric_coordinate = np.ones((len(self.parameterized_points_shape_without_principal_dimension[:-1]),)) * 0.5
                # section_origin = self._compute_section_origin(section_points=points_principal_first[i], parametric_dimension=axis, parametric_coordinate=parametric_coordinate, non_csdl=False)
                # section_origin_expanded = csdl.expand(
                #     section_origin,
                #     out_shape=points_principal_first[i].shape,
                #     action=translation_expand_action,
                # )
                # displacement_from_section_origin = points_principal_first[i] - section_origin_expanded
                # structured_shape = points_principal_first.shape[1:-1]
                # displacement_from_section_origin_flattened = displacement_from_section_origin.reshape((-1, self.num_physical_dimensions))
                # distance_along_axis = csdl.matvec(displacement_from_section_origin_flattened, axis)
                # distance_along_axis = distance_along_axis.reshape(structured_shape)
                # normalization_distance = csdl.maximum(distance_along_axis) - csdl.minimum(distance_along_axis)
                # distance_along_axis_normalized = distance_along_axis / normalization_distance
                # stretch_basis_vectors = csdl.outer(distance_along_axis_normalized, axis)
                section_origin = self._compute_section_origin(section_points=points_principal_first[i], parametric_dimension=axis, parametric_coordinate=parametric_coordinate, non_csdl=True)
                displacement_from_section_origin = points_principal_first.value[i] - section_origin
                structured_shape = points_principal_first.shape[1:-1]
                displacement_from_section_origin_flattened = displacement_from_section_origin.reshape((-1, self.num_physical_dimensions))
                distance_along_axis = np.dot(displacement_from_section_origin_flattened, axis)
                distance_along_axis = distance_along_axis.reshape(structured_shape)
                normalization_distance = np.max(distance_along_axis) - np.min(distance_along_axis)
                distance_along_axis_normalized = distance_along_axis / normalization_distance
                stretch_basis_vector = np.outer(distance_along_axis_normalized, axis).reshape(structured_shape + (self.num_physical_dimensions,))
                stretch_basis_vectors.append(stretch_basis_vector)

            stretch_basis_vectors = np.stack(stretch_basis_vectors, axis=0)
            points_principal_first = points_principal_first + stretch_basis_vectors * csdl.expand(parameter, out_shape=points_principal_first.shape, action=f"{parameter_vectorization_action}")

            # stretch = parameter[i]
            # points_principal_first = points_principal_first.set(csdl.slice[i], points_principal_first[i] + stretch * stretch_basis_vectors)


        # Apply rotations
        for axis_input, parameter, parametric_coordinate in sectional_parameters.rotations:
            for i in csdl.frange(self.num_sections):
                if isinstance(axis_input, int):
                    if parametric_coordinate is None:
                        parametric_coordinate = np.ones((len(self.parameterized_points_shape_without_principal_dimension[:-1]),)) * 0.5
                    axis = self._compute_section_axis(section_points=points_principal_first[i], parametric_dimension=axis_input, parametric_coordinate=parametric_coordinate)
                elif isinstance(axis_input, csdl.Variable):
                    axis = axis_input/csdl.norm(axis_input)
                elif isinstance(axis_input, np.ndarray):
                    axis = axis_input/np.linalg.norm(axis_input)
                else:
                    raise Exception(f"Invalid axis type: {type(axis)}. Axis should be either an int, a csdl.Variable, or a numpy array.")
            
                if parametric_coordinate is None:
                    parametric_coordinate = np.ones((len(self.parameterized_points_shape_without_principal_dimension[:-1]),)) * 0.5
                section_origin = self._compute_section_origin(section_points=points_principal_first[i], parametric_dimension=axis, parametric_coordinate=parametric_coordinate, non_csdl=False)

                angle = parameter[i]
                rotated_section = rotate(points=points_principal_first[i], rotation_origin=section_origin, axis_vector=axis, angles=angle)
                points_principal_first = points_principal_first.set(csdl.slice[i], rotated_section)


        # Apply translations
        for axis_input, parameter in sectional_parameters.translations:
            # for i in csdl.frange(self.num_sections):
            axes = []
            for i in range(self.num_sections):
                if isinstance(axis_input, int):
                    if parametric_coordinate is None:
                        parametric_coordinate = np.ones((len(self.parameterized_points_shape_without_principal_dimension[:-1]),)) * 0.5
                    axis = self._compute_section_axis(section_points=points_principal_first[i], parametric_dimension=axis_input, parametric_coordinate=parametric_coordinate)
                elif isinstance(axis_input, csdl.Variable):
                    axis = axis_input/csdl.norm(axis_input)
                elif isinstance(axis_input, np.ndarray):
                    axis = axis_input/np.linalg.norm(axis_input)
                else:
                    raise Exception(f"Invalid axis type: {type(axis)}. Axis should be either an int, a csdl.Variable, or a numpy array.")
            
                axes.append(axis)
                # translation = parameter[i]
                # expanded_translation = csdl.expand(
                #     translation * axis,
                #     out_shape=points_principal_first[i].shape,
                #     action=translation_expand_action,
                # )
                # points_principal_first = points_principal_first.set(csdl.slice[i], points_principal_first[i] + expanded_translation)
            axes = np.stack(axes, axis=0)
            axes = np.reshape(
                axes,
                (self.num_sections,) + (1,) * (len(points_principal_first.shape) - 2) + (self.num_physical_dimensions,),
            )
            axes = np.broadcast_to(axes, points_principal_first.shape)

            points_principal_first = points_principal_first + axes * csdl.expand(parameter, out_shape=points_principal_first.shape, action=f"{parameter_vectorization_action}")

        # Swap axes back to original order.
        updated_points = csdl.reorder_axes(
            points_principal_first,
            action=f"{output_axes}->{input_axes}",
        )

        # # Assemble linear maps
        # self.assemble()

        # # Add parameters that are found.
        # for axis, parameter in sectional_parameters.stretches.items():
        #     self.add_sectional_stretch(axis=axis)
        # for axis, parameter in sectional_parameters.translations.items():
        #     self.add_sectional_translation(axis=axis)
        # for axis, parameter in sectional_parameters.rotations.items():
        #     self.add_sectional_rotation(axis=axis)

        # Perform update
        # updated_points = self.parameterized_points.reshape((-1,))
        # updated_points = self.parameterized_points
        # for parameter_info, parameter_map in self.linear_parameter_maps.items():
        #     parameter_type, parameter_axis = parameter_info
        #     # parameter_type = parameter_name[: parameter_name.index("_")]
        #     # axis_string = parameter_name[parameter_name.index("_") + 1 :]
        #     # if axis_string.isdigit():
        #     #     parameter_axis = int(axis_string)
        #     # parameter_axis = int(parameter_name[parameter_name.index("_") + 1 :])
        #     if parameter_type == "sectional_stretch":
        #         parameter_variable = sectional_parameters.stretches[parameter_axis]
        #     elif parameter_type == "sectional_translation":
        #         parameter_variable = sectional_parameters.translations[parameter_axis]
        #     else:
        #         raise Exception(
        #             f"Something went wrong. It's storing a linear map for a parameter of type: {parameter_type}"
        #         )

        #     # if parameter_name not in sectional_parameters.keys():
        #     #     raise Exception(f"Please pass in a sectional parameter for {parameter_name}.")
        #     if parameter_variable.shape != (self.num_sections,):
        #         raise Exception(
        #             f"Sectional parameter of type {parameter_type} in axis {parameter_axis} has the wrong shape."
        #             + f"Expected: {self.num_sections}, got: {parameter_variable.shape}"
        #         )

        #     delta_points = csdl.sparse.matvec(parameter_map, parameter_variable.reshape((parameter_variable.size, 1)))
        #     # delta_points = delta_points.reshape((delta_points.size,))
        #     delta_points = delta_points.reshape(updated_points.shape)
        #     updated_points = updated_points + delta_points

        # # updated_points = self.parameterized_points.reshape((-1,))
        # # for parameter_name, parameter in self.parameters.items():
        # #     if parameter_name not in sectional_parameters.keys():
        # #         raise Exception(f"Please pass in a sectional parameter for {parameter_name}.")
        # #     if sectional_parameters[parameter_name].shape != (self.num_sections,):
        # #         raise Exception(f"Sectional parameter {parameter_name} has the wrong shape."+
        # #                         f"Expected: {self.num_sections}, got: {sectional_parameters[parameter_name].shape}")

        # #     updated_points = updated_points + csdl.matvec(map=self.linear_parameter_maps[parameter_name], x=sectional_parameters[parameter_name])

        # # # updated_points = csdl.matvec(map=self.evaluation_map, x=sectional_parameters) + self.parameterized_points
        # # self.parameterized_points = updated_points

        # # Perform rotations
        # updated_points = updated_points.flatten()
        # for parameter_axis, axis in self.rotational_axes.items():
        #     if isinstance(parameter_axis, NumpyAxis):
        #         parameter_axis_label = parameter_axis
        #         axis = parameter_axis.axis
        #     else:
        #         parameter_axis_label = parameter_axis
            
        #     parameter_variable = sectional_parameters.rotations[parameter_axis_label]
        #     # if parameter_type == "rotation":
        #     #     parameter_variable = sectional_parameters.rotations[parameter_axis]
        #     # else:
        #     #     raise Exception(
        #     #         f"Something went wrong. It's storing a rotational map for a parameter of type: {parameter_type}"
        #     #     )

        #     # if parameter_name not in sectional_parameters.keys():
        #     #     raise Exception(f"Please pass in a sectional parameter for {parameter_name}.")
        #     # if parameter_variable.shape != (self.num_sections,):
        #     #     raise Exception(
        #     #         f"Sectional parameter {parameter_type}:{parameter_axis} has the wrong shape."
        #     #         + f"Expected: shape=(num_sections,) ({self.num_sections},), got: {parameter_variable.shape}"
        #     #     )

        #     # # Use points to create a B-spline to help with getting axes
        #     # NOTE: Going to use static axes for now unless if popular demand justifies this.
        #     # fitting_points = self.parameterized_points.value.reshape(self.parameterized_points_shape)
        #     # import lsdo_geo.splines.b_splines.b_spline_functions as bsp
        #     # self.helpful_b_spline = bsp.fit_b_spline(fitting_points=fitting_points, order=(2,),
        #     #                                             num_coefficients=(self.num_sections,),
        #     #                                             name='helpful_b_spline')
        #     for i in range(self.num_sections):
        #         parametric_coordinate = (
        #             np.ones((len(self.parameterized_points_shape[:-1]))) * 0.5
        #         )
        #         parametric_coordinate[self.principal_parametric_dimension] = (
        #             self.sectional_principal_parametric_coordinate[i]
        #         )
        #         if isinstance(axis, int):
        #             parametric_derivative_order = np.zeros(
        #                 (len(self.parameterized_points_shape[:-1])), dtype=int
        #             )
        #             parametric_derivative_order[axis] = 1
        #             parametric_derivative_order = tuple(parametric_derivative_order)
        #             rotation_axis = self.helpful_b_spline.evaluate(
        #                 parametric_coordinates=parametric_coordinate,
        #                 parametric_derivative_orders=parametric_derivative_order,
        #             ).value
        #             rotation_axis /= np.linalg.norm(rotation_axis)
        #         elif isinstance(axis, csdl.Variable):
        #             rotation_axis = axis.value
        #         elif isinstance(axis, np.ndarray):
        #             rotation_axis = axis
        #         else:
        #             raise Exception(f"Invalid axis type: {type(axis)}. Axis should be either an int, a csdl.Variable, or a numpy array.")

        #         angle = parameter_variable[i]
        #         indices = np.arange(np.prod(self.parameterized_points_shape, dtype=int))
        #         indices = indices.reshape(self.parameterized_points_shape)
        #         indices = np.swapaxes(indices, 0, self.principal_parametric_dimension)
        #         indices = indices[i].reshape((-1,))

        #         section_updated_points = updated_points[list(indices)].reshape(
        #             (len(indices)//self.num_physical_dimensions, self.num_physical_dimensions)
        #         )

        #         section_updated_points_sum = csdl.sum(section_updated_points, axes=(0,))
        #         number_of_points = csdl.Variable(
        #             name="number_of_section_points",
        #             shape=(self.num_physical_dimensions,),
        #             value=np.ones((self.num_physical_dimensions,))
        #             * self.num_points_per_section,
        #         )
        #         section_average = section_updated_points_sum / number_of_points

        #         updated_points = updated_points.set(csdl.slice[[indices]], rotate(
        #             points=section_updated_points,
        #             rotation_origin=section_average,
        #             axis_vector=rotation_axis,
        #             angles=angle,
        #         ).reshape((updated_points[list(indices)].size,)))

        # updated_points = updated_points.reshape(self.parameterized_points_shape)

        # self.parameterized_points = updated_points
        self.updated_points = updated_points
        if plot:  # Note: plot the surfaces for each section. (if 3d)
            # plot the updated ffd block in section form with the updated points.
            self.plot()

        return updated_points

    def plot(
        self,
        opacity: float = 0.8,
        color: str = "#182B49",
        surface_texture: str = "",
        additional_plotting_elements: list = [],
        show: bool = True,
    ):
        """
        Plots the updated volume in section form.
        """
        plotting_elements = additional_plotting_elements.copy()

        # plotting_points = self.parameterized_points.value.reshape(self.parameterized_points_shape)
        plotting_points = self.updated_points.value.reshape(
            self.parameterized_points_shape
        )
        plotting_points = np.swapaxes(
            plotting_points, 0, self.principal_parametric_dimension
        )
        for i in range(self.num_sections):
            section_points = plotting_points[i]
            section_parametric_ndim = section_points.ndim - 1

            if section_parametric_ndim == 1:
                plotting_elements = lfs.plot_curve(
                    section_points,
                    opacity=opacity,
                    color=color,
                    additional_plotting_elements=plotting_elements,
                    show=False,
                )
            elif section_parametric_ndim == 2:
                plotting_elements = lfs.plot_surface(
                    section_points,
                    plot_types=['function'],
                    opacity=opacity,
                    color=color,
                    surface_texture=surface_texture,
                    additional_plotting_elements=plotting_elements,
                    show=False,
                )

            plotting_elements = lfs.plot_points(
                section_points,
                opacity=1.0,
                color='#00629B',
                size=10,
                additional_plotting_elements=plotting_elements,
                show=False,
            )

        if show:
            view_up = 'z' if self.num_physical_dimensions >= 3 else 'y'
            lfs.show_plot(
                plotting_elements,
                f"Parameterized Sections: {self.name}",
                axes=True,
                view_up=view_up,
                interactive=True,
            )
            return plotting_elements
        else:
            return plotting_elements


def _get_parametric_coordinate(
    shape: tuple, total_index: int, axis: int, axis_index: int
):
    parametric_coordinate = []
    for i in range(len(shape)):
        if i == axis:
            continue
        axis_index = total_index // np.prod(shape[i + 1 :])
        parametric_coordinate.append(axis_index / shape[i])

    parametric_coordinate = np.array(parametric_coordinate)
    parametric_coordinate = np.insert(
        parametric_coordinate, axis, axis_index / shape[axis]
    )
    return parametric_coordinate.reshape((1, -1))


def _get_indices_in_shape(
    shape: tuple, total_index: int, section_axis: int, section_axis_index: int
):
    indices = []
    remainder = total_index
    for i in range(len(shape)):
        if i == section_axis:
            continue
        # axis_index = int(total_index//np.prod(shape[i+1:]))
        if i < section_axis:
            axis_index, remainder = np.divmod(
                remainder, (np.prod(shape[i + 1 :]) / shape[section_axis])
            )
        else:
            axis_index, remainder = np.divmod(remainder, np.prod(shape[i + 1 :]))
        indices.append(int(axis_index))

    indices = np.array(indices)
    indices = np.insert(indices, section_axis, section_axis_index)
    return tuple(indices)


if __name__ == "__main__":
    pass
