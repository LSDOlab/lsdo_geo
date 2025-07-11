import numpy as np
import numpy.typing as npt
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
from lsdo_geo.core.geometry.geometry import Geometry
from typing import Union, Optional, Sequence, cast
import scipy.sparse as sps

# from dataclasses import dataclass
# @dataclass
class FFDBlock(lfs.Function):
    '''
    Free-form deformation block class.

    Parameters
    ----------
    Function class. This class is used to represent a function in a given function space. The function space is used to evaluate the function at
    given coordinates, refit the function, and project points onto the function.

    Attributes
    ----------
    space : lfs.FunctionSpace
        The function space in which the function resides.
    coefficients : csdl.Variable -- shape=coefficients_shape
        The coefficients of the function.
    name : str = None
        If applicable, the name of the function.
    embedded_entities : list[Union[csdl.Variable,Geometry,lfs.Function,lfs.FunctionSet]]
        The entities to be embedded within (parameterized by) the FFD block.
    embedded_entity_parametric_coordinates : list[np.ndarray] -- list_length=len(embedded_entities), array_shape=(num_points, num_parametric_dimensions)
        The parametric coordinates for each of the embedded entities.
    '''

    EntityType = Union[csdl.Variable,Geometry,lfs.Function,lfs.FunctionSet, npt.NDArray[np.float64]]

    def __init__(self, space:lfs.FunctionSpace, coefficients:csdl.Variable, name:Optional[str]=None,
                    embedded_entities:Optional[Sequence[EntityType]]=None,
                    embedded_entity_parametric_coordinates:Optional[Sequence[npt.NDArray[np.float64]]]=None):

        super().__init__(space=space, coefficients=coefficients, name=name)
        
        if embedded_entities is None:
            embedded_entities = []
        if embedded_entity_parametric_coordinates is None:
            embedded_entity_parametric_coordinates = []

        self.embedded_entities = embedded_entities
        self.embedded_entity_parametric_coordinates = embedded_entity_parametric_coordinates

        # self.basis_matrices = []
        self.embed_entities(entities=self.embedded_entities)

        if len(self.embedded_entities) != len(self.embedded_entity_parametric_coordinates):
            raise ValueError(f'Number of entities ({len(self.embedded_entities)}) and parametric coordinates'+
                                f'({len(self.embedded_entity_parametric_coordinates)}) do not match.')

    def embed_entities(self, entities:Sequence[EntityType]):

        for entity in entities:
            if isinstance(entity, np.ndarray):
                embedded_points = entity
            elif isinstance(entity, csdl.Variable):
                embedded_points = entity.value
            elif isinstance(entity, lfs.Function):
                embedded_points = entity.coefficients.value
            elif isinstance(entity, lfs.FunctionSet):
                embedded_points = []
                for function in entity.functions.values():
                    embedded_points.append(function.coefficients.value)
            else:
                raise ValueError(f'Unsupported entity type: {type(entity)}')

            if not isinstance(embedded_points, list):
                embedded_points_parametric_coordinates = self.project(points=embedded_points, projection_tolerance=1e-4, force_reproject=False)
                self.embedded_entity_parametric_coordinates.append(embedded_points_parametric_coordinates)
                # entity_basis_matrix = self.space.compute_basis_matrix(parametric_coordinates=embedded_points_parametric_coordinates)
                # self.basis_matrices.append(entity_basis_matrix)
            else:
                entity_parametric_coordinates = []
                for points in embedded_points:
                    embedded_points_parametric_coordinates = self.project(points=points, projection_tolerance=1e-4)
                    entity_parametric_coordinates.append(embedded_points_parametric_coordinates)
                self.embedded_entity_parametric_coordinates.append(entity_parametric_coordinates)
                #     entity_basis_matrix = self.space.compute_basis_matrix(parametric_coordinates=embedded_points_parametric_coordinates)
                #     entity_basis_matrices.append(entity_basis_matrix)
                # self.basis_matrices.append(entity_basis_matrices)


    def evaluate_ffd(self, coefficients:csdl.Variable, plot:bool=False, non_csdl:bool=False) -> Union[csdl.Variable, npt.NDArray[np.float64], list[csdl.Variable], list[npt.NDArray[np.float64]]]:
        '''
        Evaluates the function.

        Parameters
        ----------
        coefficients : csdl.Variable = None -- shape=coefficients_shape
            The coefficients of the function.
        parametric_coordinates : np.ndarray = None -- shape=(num_points, num_parametric_dimensions)
            OPTIONAL: The coordinates at which to evaluate the function. 
            If None (which is the intended use case), then the function is evaluated at the parametric coordinates of the embedded entities.
        parametric_derivative_order : tuple = None -- shape=(num_points,num_parametric_dimensions)
            The order of the parametric derivatives to evaluate.
        plot : bool = False
            Whether or not to plot the function with the points from the result of the evaluation.
        

        Returns
        -------
        function_values : csdl.Variable
            The function evaluated at the given coordinates.
        '''
        self.coefficients = coefficients
        if len(self.embedded_entity_parametric_coordinates) == 0:
            raise ValueError('No parametric coordinates provided for evaluation.')
        parametric_coordinates = self.embedded_entity_parametric_coordinates

        outputs : Union[list[csdl.Variable], list[list[csdl.Variable]]] = []
        for entity_parametric_coordinates in parametric_coordinates:
            if not isinstance(entity_parametric_coordinates, list):
                entity_parametric_coordinates = [entity_parametric_coordinates]
            entity_outputs : list[csdl.Variable] = []
            for entity_parametric_coordinate in entity_parametric_coordinates:
                entity_outputs.append(super().evaluate(parametric_coordinates=entity_parametric_coordinate, 
                                                        coefficients=coefficients, plot=plot, non_csdl=non_csdl))
            if len(entity_outputs) == 1:
                outputs.append(entity_outputs[0])
            else:
                outputs.append(entity_outputs)

        if plot:
            outputs_to_plot = []
            for entity_points in outputs:
                if isinstance(entity_points, list):
                    for entity_entity_points in entity_points:
                        if isinstance(entity_entity_points, csdl.Variable):
                            outputs_to_plot.append(entity_entity_points.value)
                        else:
                            outputs_to_plot.append(entity_entity_points)
                else:
                    if isinstance(entity_points, csdl.Variable):
                        outputs_to_plot.append(entity_points.value)
                    else:
                        outputs_to_plot.append(entity_points)
            self.plot(plot_embedded_points=True, embedded_points=outputs_to_plot, opacity=0.3, show=True)

        if len(outputs) == 1:
            # output is a single coefficients variable/array
            if non_csdl:
                output_with_type : npt.NDArray[np.float64] = outputs[0]
                return output_with_type
            else:
                output_with_type : csdl.Variable = outputs[0]
            return output_with_type
        else:
            # output is a list of coefficients
            if non_csdl:
                outputs_with_type : list[npt.NDArray[np.float64]] = outputs
                return outputs_with_type
            else:
                outputs_with_type : list[csdl.Variable] = outputs
            return outputs_with_type

    # def evaluate_ffd(self, coefficients:csdl.Variable, plot:bool=False) -> csdl.Variable:
    #     '''
    #     Takes in the FFD block coefficients and evaluates the embedded points.

    #     Parameters
    #     ----------
    #     coefficients : csdl.Variable
    #         The coefficients of the FFD block.
    #     plot : bool
    #         A boolean on whether or not to plot the FFD block.

    #     Returns
    #     -------
    #     updated_points : csdl.Variable
    #         The embedded points.
    #     '''
    #     return self.evaluate(coefficients=coefficients, plot=plot)
    #     # if len(coefficients.shape) > 2:
    #     #     coefficients = coefficients.reshape((coefficients.size//self.num_physical_dimensions, self.num_physical_dimensions))

    #     # # Perform update
    #     # self.coefficients = coefficients

    #     # outputs = []
    #     # for basis_matrix in self.basis_matrices:
    #     #     if not isinstance(basis_matrix, list):
    #     #         updated_points = basis_matrix @ self.coefficients
    #     #         outputs.append(updated_points)
    #     #     else:
    #     #         updated_points = []
    #     #         for entity_basis_matrix in basis_matrix:
    #     #             if isinstance(coefficients, csdl.Variable) and sps.issparse(entity_basis_matrix):
    #     #                 coefficients_reshaped = coefficients.reshape((entity_basis_matrix.shape[1], coefficients.size//entity_basis_matrix.shape[1]))
    #     #                 # NOTE: TEMPORARY IMPLEMENTATION SINCE CSDL ONLY SUPPORTS SPARSE MATVECS AND NOT MATMATS
    #     #                 values = csdl.Variable(value=np.zeros((entity_basis_matrix.shape[0], coefficients_reshaped.shape[1])))
    #     #                 for i in range(coefficients_reshaped.shape[1]):
    #     #                     coefficients_column = coefficients_reshaped[:,i].reshape((coefficients_reshaped.shape[0],1))
    #     #                     values = values.set(csdl.slice[:,i], csdl.sparse.matvec(entity_basis_matrix, coefficients_column).reshape(
    #     #                         (entity_basis_matrix.shape[0],)))
    #     #             else:
    #     #                 values = entity_basis_matrix @ coefficients.reshape((entity_basis_matrix.shape[1], -1))
    #     #             updated_points.append(values)
    #     #         outputs.append(updated_points)

    #     # if plot:
    #     #     outputs_to_plot = []
    #     #     for entity_points in outputs:
    #     #         if isinstance(entity_points, list):
    #     #             for entity_entity_points in entity_points:
    #     #                 outputs_to_plot.append(entity_entity_points.value)
    #     #         else:
    #     #             outputs_to_plot.append(entity_points.value)
    #     #     self.plot(plot_embedded_points=True, embedded_points=outputs_to_plot, opacity=0.3, show=True)

    #     # if len(outputs) == 1:
    #     #     return outputs[0]
    #     # else:
    #     #     return outputs
        

    # def evaluate_as_function(self, parametric_coordinates:np.ndarray, parametric_derivative_orders:list[tuple]=None, coefficients:csdl.Variable=None,
    #              plot:bool=False) -> csdl.Variable:
    #     '''
    #     Evaluates the function.

    #     Parameters
    #     ----------
    #     parametric_coordinates : np.ndarray -- shape=(num_points, num_parametric_dimensions)
    #         The coordinates at which to evaluate the function.
    #     parametric_derivative_order : tuple = None -- shape=(num_points,num_parametric_dimensions)
    #         The order of the parametric derivatives to evaluate.
    #     coefficients : csdl.Variable = None -- shape=coefficients_shape
    #         The coefficients of the function.
    #     plot : bool = False
    #         Whether or not to plot the function with the points from the result of the evaluation.
        

    #     Returns
    #     -------
    #     function_values : csdl.Variable
    #         The function evaluated at the given coordinates.
    #     '''
    #     return super().evaluate(parametric_coordinates=parametric_coordinates, parametric_derivative_orders=parametric_derivative_orders,
    #                      coefficients=coefficients, plot=plot)
        

    def plot(self, plot_embedded_points:bool=True, embedded_points:list=None,
             opacity:float=0.3, color:str='#00629B', surface_texture:str="",
             additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the FFD block and optionally the points embedded within.
        '''
        plotting_elements = additional_plotting_elements.copy()

        plotting_elements = super().plot(point_types=['evaluated_points'], opacity=opacity, color=color,
                                         surface_texture=surface_texture, additional_plotting_elements=plotting_elements, show=False)
        plotting_elements = super().plot(point_types=['coefficients'], plot_types=['point_cloud'], opacity=opacity, color=color,
                                         surface_texture=surface_texture, additional_plotting_elements=plotting_elements, show=False)

        if plot_embedded_points:
            if embedded_points is None:
                embedded_points = []
                for entity in self.embedded_entities:
                    if isinstance(entity, np.ndarray):
                        embedded_points.append(entity)
                    elif isinstance(entity, csdl.Variable):
                        embedded_points.append(entity.value)
                    elif isinstance(entity, lfs.Function):
                        embedded_points.append(entity.coefficients.value)
                    elif isinstance(entity, lfs.FunctionSet):
                        for function in entity.functions.values():
                            embedded_points.append(function.coefficients.value)
                    else:
                        raise ValueError(f'Unsupported entity type: {type(entity)}')
            
            for i, entity_points in enumerate(embedded_points):
                # if isinstance(entity, np.ndarray) or isinstance(entity, csdl.Variable):
                # NOTE: I am intentionally plotting everything as a point cloud because that is specifically what the FFD block is operating on.
                # if isinstance(entity_points, list):
                #     for entity_entity_points in entity_points:
                #         plotting_elements = lfs.plot_points(points=entity_entity_points, opacity=1., color='#182B49',
                #                                         additional_plotting_elements=plotting_elements, show=False)
                # else:
                plotting_elements = lfs.plot_points(points=entity_points, opacity=1., color='#182B49',
                                                    additional_plotting_elements=plotting_elements, show=False)

        if show:
            # plotter = vedo.Plotter()
            # plotter.show(plotting_elements, f'Free Form Deformation Block: {self.name}', axes=1, viewup="z", interactive=True)
            if self.name is not None:
                lfs.show_plot(plotting_elements, f'Free Form Deformation Block: {self.name}', axes=1, interactive=True)
            else:
                lfs.show_plot(plotting_elements, 'Free Form Deformation Block', axes=1, interactive=True)
            return plotting_elements
        else:
            return plotting_elements
    


if __name__ == "__main__":
    pass

