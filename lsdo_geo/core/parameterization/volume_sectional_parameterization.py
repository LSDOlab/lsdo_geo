import numpy as np
import scipy.sparse as sps
import m3l
import vedo
import csdl_alpha as csdl

from dataclasses import dataclass

from lsdo_geo.core.geometry.geometry_functions import rotate


@dataclass
class VolumeSectionalParameterizationInputs:
    """
    Dataclass of inputs for the VolumeSectionalParameterization class. This is passed into the evaluate method.
    The desired parameters should be appended to the appropiate dictionary.

    Parameters
    ----------
    stretches : dict[int,csdl.Variable] = None
        The stretches for each section. The key of the dictionary is the integer corresponding to the axis of the stretch.
        The value is the CSDL variable that contains the stretch values.
    translations : dict[int,csdl.Variable] = None
        The translations for each section. The key of the dictionary is the integer corresponding to the axis of the translation.
        The value is the CSDL variable that contains the translation values.
    rotations : dict[int,csdl.Variable] = None
        The rotations for each section. The key of the dictionary is the integer corresponding to the axis of the rotation.
        The value is the CSDL variable that contains the rotation values.
    """

    stretches: dict[int, csdl.Variable] = None
    translations: dict[int, csdl.Variable] = None
    rotations: dict[int, csdl.Variable] = None

    def __post_init__(self):
        if self.stretches is None:
            self.stretches = {}
        if self.translations is None:
            self.translations = {}
        if self.rotations is None:
            self.rotations = {}

    def add_sectional_stretch(self, axis: int, stretch: csdl.Variable):
        """
        Adds a stretch to the stretches dictionary.

        Parameters
        ----------
        axis : int
            The axis of the stretch.
        stretch : csdl.Variable
            The stretch values.
        """
        self.stretches[axis] = stretch

    def add_sectional_translation(self, axis: int, translation: csdl.Variable):
        """
        Adds a translation to the translations dictionary.

        Parameters
        ----------
        axis : int
            The axis of the translation.
        translation : csdl.Variable
            The translation values.
        """
        self.translations[axis] = translation

    def add_sectional_rotation(self, axis: int, rotation: csdl.Variable):
        """
        Adds a rotation to the rotations dictionary.

        Parameters
        ----------
        axis : int
            The axis of the rotation.
        rotation : csdl.Variable
            The rotation values.
        """
        self.rotations[axis] = rotation


@dataclass
class VolumeSectionalParameterization:
    """
    Class for parameterizing rectangular volumes by specifying a principal parametric dimension and perceiving the volume
    as a series of sections along the axis.

    Parameters
    ----------
    name : str
        The name of the free form deformation (FFD) block.
    principal_parametric_dimension : int = 0
        The principal parametric dimension of the volume.
    parameterized_points : csdl.Variable
        The points that are being parameterized.
    paremeterized_points_shape : tuple[int] = None
        The shape of the parameterized points.
    parameters : dict[str,SectionalParameter] = None
        The sectional parameters to use for the parameterization.
    parameter_indices : dict[str,int] = None
        The indices of the parameters in the parameter vector (constructed for evaluation).
    """

    name: str
    parameterized_points: csdl.Variable
    principal_parametric_dimension: int = 0
    parameterized_points_shape: tuple[int] = None
    linear_parameter_maps: dict[str, sps.csc_matrix] = None
    rotational_axes: dict[str, int] = None

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
        import lsdo_geo.splines.b_splines.b_spline_functions as bsp

        self.helpful_b_spline = bsp.fit_b_spline(
            fitting_points=fitting_points,
            order=(2,),
            num_coefficients=self.parameterized_points_shape[:-1],
            name="helpful_b_spline",
        )

        self.sectional_principal_parametric_coordinate = np.linspace(
            0.0, 1.0, self.num_sections
        ).reshape((-1, 1))

        self.linear_parameter_maps = {}
        self.rotational_axes = {}

        self.updated_points = self.parameterized_points # NOTE: Removing .copy() here because csdl doesn't have one.

    def add_parameter(self, name: str, map: sps.csc_matrix):
        """
        Adds a sectional parameter to the parameterization. The map should map from the parameter vector to deltas in the parameterized points.

        Parameters
        ----------
        name : str
            The name of the sectional parameter.
        map : sps.csc_matrix
            The map from the parameter vector to deltas in the parameterized points.
        """
        self.linear_parameter_maps[name] = map

    def add_sectional_translation(self, name: str, axis: int):
        """
        Adds a sectional translation parameter to the parameterization.

        Parameters
        ----------
        name : str
            The name of the sectional translation parameter.
        axis : int
            The axis to translate along.
        """
        valid_axes = np.arange(len(self.parameterized_points_shape) - 1)
        if axis not in valid_axes:
            raise Exception(f"Please pass in a valid axis. valid axes:{valid_axes}")

        num_outputs = np.prod(self.parameterized_points_shape)

        parameter_map_list = []
        for i in range(self.num_sections):
            parameter_section_map = sps.lil_matrix((num_outputs, 1))

            parametric_coordinate = (
                np.ones((len(self.parameterized_points_shape[:-1]))) * 0.5
            )
            parametric_coordinate[self.principal_parametric_dimension] = (
                self.sectional_principal_parametric_coordinate[i].reshape((1, -1))
            )
            parametric_derivative_order = np.zeros(
                (len(self.parameterized_points_shape[:-1]))
            )
            parametric_derivative_order[axis] = 1
            parametric_derivative_order = tuple(parametric_derivative_order)
            translation_axis = self.helpful_b_spline.evaluate(
                parametric_coordinates=parametric_coordinate,
                parametric_derivative_order=parametric_derivative_order,
            ).value
            translation_axis /= np.linalg.norm(translation_axis)

            indices = np.arange(np.prod(self.parameterized_points_shape, dtype=int))
            indices = indices.reshape(self.parameterized_points_shape)
            indices = np.swapaxes(indices, 0, self.principal_parametric_dimension)
            indices = indices[i].reshape((-1,))

            parameter_section_map[indices] = np.tile(
                translation_axis, self.num_points_per_section
            ).reshape((-1, 1))

            parameter_map_list.append(parameter_section_map)

        parameter_map = sps.hstack(parameter_map_list).tocsc()

        self.add_parameter(name=name, map=parameter_map)

    def add_sectional_stretch(self, name: str, axis: int):
        """
        Adds a sectional stretch parameter to the parameterization.

        Parameters
        ----------
        name : str
            The name of the sectional stretch parameter.
        axis : int
            The axis to stretch along.
        """
        valid_axes = np.delete(
            np.arange(len(self.parameterized_points_shape) - 1),
            self.principal_parametric_dimension,
        )
        if axis not in valid_axes:
            raise Exception(f"Please pass in a valid axis. valid axes:{valid_axes}")

        num_outputs = np.prod(self.parameterized_points_shape)

        parameter_map_list = []
        for i in range(self.num_sections):
            parameter_section_map = sps.lil_matrix((num_outputs, 1))

            if axis == self.principal_parametric_dimension:
                raise Exception(
                    "Can't stretch along the principal parametric dimension because sections have no thickness to stretch."
                    + "Use a linear distributions of translations instead"
                )

            parametric_coordinate = (
                np.ones((len(self.parameterized_points_shape[:-1]))) * 0.5
            )
            parametric_coordinate[self.principal_parametric_dimension] = (
                self.sectional_principal_parametric_coordinate[i].reshape((1, -1))
            )
            parametric_derivative_order = np.zeros(
                (len(self.parameterized_points_shape[:-1]))
            )
            parametric_derivative_order[axis] = 1
            parametric_derivative_order = tuple(parametric_derivative_order)
            stretch_axis = self.helpful_b_spline.evaluate(
                parametric_coordinates=parametric_coordinate,
                parametric_derivative_order=parametric_derivative_order,
            ).value
            stretch_axis /= np.linalg.norm(stretch_axis)
            section_middle = self.helpful_b_spline.evaluate(
                parametric_coordinates=parametric_coordinate,
                parametric_derivative_order=(0,),
            ).value

            section_axis_end_parametric_coordinate = parametric_coordinate
            section_axis_end_parametric_coordinate[axis] = 1.0
            section_axis_beginning_parametric_coordinate = parametric_coordinate.copy()
            section_axis_beginning_parametric_coordinate[axis] = 0.0

            section_axis_end = self.helpful_b_spline.evaluate(
                parametric_coordinates=section_axis_end_parametric_coordinate,
                parametric_derivative_order=(0,),
            ).value
            section_axis_beginning = self.helpful_b_spline.evaluate(
                parametric_coordinates=section_axis_beginning_parametric_coordinate,
                parametric_derivative_order=(0,),
            ).value
            section_length = (section_axis_end - section_axis_beginning).dot(
                stretch_axis
            )

            indices = np.arange(np.prod(self.parameterized_points_shape, dtype=int))
            indices = indices.reshape(self.parameterized_points_shape)
            indices = np.swapaxes(indices, 0, self.principal_parametric_dimension)
            indices = indices.reshape(
                (
                    self.num_sections,
                    self.num_points_per_section,
                    self.num_physical_dimensions,
                )
            )

            for j in range(self.num_points_per_section):
                point_indices_full_shape = _get_indices_in_shape(
                    shape=self.parameterized_points_shape[:-1],
                    total_index=j,
                    section_axis=self.principal_parametric_dimension,
                    section_axis_index=i,
                )
                displacement = (
                    self.parameterized_points.value.reshape(
                        self.parameterized_points_shape
                    )[point_indices_full_shape]
                    - section_middle
                )
                distance_along_axis = np.dot(displacement, stretch_axis)

                point_indices = indices[i, j, :].reshape((-1,))
                parameter_section_map[point_indices] = (
                    distance_along_axis
                    / section_length
                    * stretch_axis.reshape((self.num_physical_dimensions, 1))
                )

            parameter_map_list.append(parameter_section_map)

        parameter_map = sps.hstack(parameter_map_list).tocsc()

        self.add_parameter(name=name, map=parameter_map)

    def add_sectional_rotation(self, name: str, axis: int):
        """
        Adds a sectional rotation parameter to the parameterization.

        Parameters
        ----------
        name : str
            The name of the sectional rotation parameter.
        axis : int
            The axis to rotate about.
        """
        valid_axes = np.arange(len(self.parameterized_points_shape) - 1)
        if axis not in valid_axes:
            raise Exception(f"Please pass in a valid axis. valid axes:{valid_axes}")

        self.rotational_axes[name] = axis

    # def evaluate(self, sectional_parameters:dict[str,csdl.Variable], plot:bool=False) -> csdl.Variable:
    def evaluate(
        self,
        sectional_parameters: VolumeSectionalParameterizationInputs,
        plot: bool = False,
    ) -> csdl.Variable:
        """
        Takes in a dictionary of declared parameters and their variable.

        Parameters
        ----------
        section_parameters : dict[str, csdl.Variable]
            The dictionary of parameters for each section. the key is the name of the parameter and the value is the variable.
        plot : bool = False
            Whether or not to plot the parameterized points after evaluation.

        Returns
        -------
        updated_points : csdl.Variable
            The updated points.
        """
        # # Assemble linear maps
        # self.assemble()

        # Add parameters that are found.
        for axis, parameter in sectional_parameters.stretches.items():
            auto_generated_name = f"stretch_{axis}"
            self.add_sectional_stretch(name=auto_generated_name, axis=axis)
        for axis, parameter in sectional_parameters.translations.items():
            auto_generated_name = f"translation_{axis}"
            self.add_sectional_translation(name=auto_generated_name, axis=axis)
        for axis, parameter in sectional_parameters.rotations.items():
            auto_generated_name = f"rotation_{axis}"
            self.add_sectional_rotation(name=auto_generated_name, axis=axis)

        # Perform update
        # updated_points = self.parameterized_points.reshape((-1,))
        updated_points = self.parameterized_points
        for parameter_name, parameter_map in self.linear_parameter_maps.items():
            parameter_type = parameter_name[: parameter_name.index("_")]
            parameter_axis = int(parameter_name[parameter_name.index("_") + 1 :])
            if parameter_type == "stretch":
                parameter_variable = sectional_parameters.stretches[parameter_axis]
            elif parameter_type == "translation":
                parameter_variable = sectional_parameters.translations[parameter_axis]
            else:
                raise Exception(
                    f"Something went wrong. It's storing a linear map for a parameter of type: {parameter_type}"
                )

            # if parameter_name not in sectional_parameters.keys():
            #     raise Exception(f"Please pass in a sectional parameter for {parameter_name}.")
            if parameter_variable.shape != (self.num_sections,):
                raise Exception(
                    f"Sectional parameter {parameter_name} has the wrong shape."
                    + f"Expected: {self.num_sections}, got: {parameter_variable.shape}"
                )

            delta_points = csdl.sparse.matvec(parameter_map, parameter_variable.reshape((parameter_variable.size, 1)))
            delta_points = delta_points.reshape((delta_points.size,))
            updated_points = updated_points + delta_points

        # updated_points = self.parameterized_points.reshape((-1,))
        # for parameter_name, parameter in self.parameters.items():
        #     if parameter_name not in sectional_parameters.keys():
        #         raise Exception(f"Please pass in a sectional parameter for {parameter_name}.")
        #     if sectional_parameters[parameter_name].shape != (self.num_sections,):
        #         raise Exception(f"Sectional parameter {parameter_name} has the wrong shape."+
        #                         f"Expected: {self.num_sections}, got: {sectional_parameters[parameter_name].shape}")

        #     updated_points = updated_points + csdl.matvec(map=self.linear_parameter_maps[parameter_name], x=sectional_parameters[parameter_name])

        # # updated_points = csdl.matvec(map=self.evaluation_map, x=sectional_parameters) + self.parameterized_points
        # self.parameterized_points = updated_points

        # Perform rotations
        for parameter_name, axis in self.rotational_axes.items():
            parameter_type = parameter_name[: parameter_name.index("_")]
            parameter_axis = int(parameter_name[parameter_name.index("_") + 1 :])
            if parameter_type == "rotation":
                parameter_variable = sectional_parameters.rotations[parameter_axis]
            else:
                raise Exception(
                    f"Something went wrong. It's storing a rotational map for a parameter of type: {parameter_type}"
                )

            # if parameter_name not in sectional_parameters.keys():
            #     raise Exception(f"Please pass in a sectional parameter for {parameter_name}.")
            if parameter_variable.shape != (self.num_sections,):
                raise Exception(
                    f"Sectional parameter {parameter_name} has the wrong shape."
                    + f"Expected: {self.num_sections}, got: {parameter_variable.shape}"
                )

            # # Use points to create a B-spline to help with getting axes
            # NOTE: Going to use static axes for now unless if popular demand justifies this.
            # fitting_points = self.parameterized_points.value.reshape(self.parameterized_points_shape)
            # import lsdo_geo.splines.b_splines.b_spline_functions as bsp
            # self.helpful_b_spline = bsp.fit_b_spline(fitting_points=fitting_points, order=(2,),
            #                                             num_coefficients=(self.num_sections,),
            #                                             name='helpful_b_spline')
            for i in range(self.num_sections):
                parametric_coordinate = (
                    np.ones((len(self.parameterized_points_shape[:-1]))) * 0.5
                )
                parametric_coordinate[self.principal_parametric_dimension] = (
                    self.sectional_principal_parametric_coordinate[i]
                )
                parametric_derivative_order = np.zeros(
                    (len(self.parameterized_points_shape[:-1])), dtype=int
                )
                parametric_derivative_order[axis] = 1
                parametric_derivative_order = tuple(parametric_derivative_order)
                rotation_axis = self.helpful_b_spline.evaluate(
                    parametric_coordinates=parametric_coordinate,
                    parametric_derivative_order=parametric_derivative_order,
                ).value
                rotation_axis /= np.linalg.norm(rotation_axis)

                angle = parameter_variable[i]
                indices = np.arange(np.prod(self.parameterized_points_shape, dtype=int))
                indices = indices.reshape(self.parameterized_points_shape)
                indices = np.swapaxes(indices, 0, self.principal_parametric_dimension)
                indices = indices[i].reshape((-1,))

                section_updated_points = updated_points[list(indices)].reshape(
                    (len(indices)//self.num_physical_dimensions, self.num_physical_dimensions)
                )

                section_updated_points_sum = csdl.sum(section_updated_points, axes=(0,))
                number_of_points_m3l = csdl.Variable(
                    name="number_of_section_points",
                    shape=(self.num_physical_dimensions,),
                    value=np.ones((self.num_physical_dimensions,))
                    * self.num_points_per_section,
                )
                section_average = section_updated_points_sum / number_of_points_m3l

                # # TODO: convert this when rotate function is moved into geometry_functions
                # raise NotImplementedError(
                #     "Need to m3l.rotate function to geometry_functions in new csdl."
                # )
                updated_points = updated_points.set(csdl.slice[[indices]], rotate(
                    points=section_updated_points,
                    axis_origin=section_average,
                    axis_vector=rotation_axis,
                    angles=angle,
                ).reshape((updated_points[list(indices)].size,)))

        # self.parameterized_points = updated_points
        self.updated_points = updated_points
        if plot:  # Note: plot the surfaces for each section. (if 3d)
            # plot the updated ffd block in section form with the updated points.
            self.plot()

        return updated_points

    def plot(
        self,
        opacity: float = 0.3,
        color: str = "#182B49",
        surface_texture: str = "",
        additional_plotting_elements: list = [],
        show: bool = True,
    ):
        """
        Plots the updated ffd block in section form with the updated points.
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
            section_points = plotting_points[i, :, :, :]

            section_plot_types = ["surface", "point_cloud"]
            plotting_elements = self.helpful_b_spline.plot_section(
                section_points,
                plot_types=section_plot_types,
                opacity=opacity,
                color=color,
                surface_texture=surface_texture,
                additional_plotting_elements=plotting_elements,
                show=False,
            )

        if show:
            plotter = vedo.Plotter()
            plotter.show(
                plotting_elements,
                f"Free Form Deformation Block: {self.name}",
                axes=1,
                viewup="z",
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
