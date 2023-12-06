import numpy as np
import m3l
from lsdo_geo.splines.b_splines.b_spline import BSpline
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
from lsdo_geo.splines.b_splines.b_spline_sub_set import BSplineSubSet
from lsdo_geo.core.geometry.geometry import Geometry
from typing import Union
import vedo

from dataclasses import dataclass
@dataclass
class FFDBlock(BSpline):
    '''
    Free-form deformation block class.

    Parameters
    ----------
    name : str
        The name of the free form deformation (FFD) block.
    space : BSplineSpace
        The space that the B-spline is in.
    coefficients : m3l.Variable
        The coefficients of the B-spline FFD block.
    num_physical_dimensions : int
        The number of physical dimensions that the FFD block is in.
    embedded_points : m3l.Variable
        The points embedded within the FFD block.
    '''
    embedded_entities : list[Union[m3l.Variable,BSpline,BSplineSet,BSplineSubSet,Geometry]]

    def __post_init__(self):
        super().__post_init__()

        if type(self.embedded_entities) is not list:
            self.embedded_entities = [self.embedded_entities]

        embedded_points = []
        embedded_points_indices = {}
        embedded_points_index = 0
        for embedded_entity in self.embedded_entities:
            if type(embedded_entity) is m3l.Variable:
                embedded_points.append(embedded_entity.value)
                embedded_points_indices[embedded_entity.name] = np.arange(embedded_points_index, embedded_points_index+len(embedded_entity))
                embedded_points_index += len(embedded_entity)
            elif type(embedded_entity) is BSpline or type(embedded_entity) is BSplineSet or type(embedded_entity) is Geometry:
                embedded_points.append(embedded_entity.coefficients.value)
                embedded_points_indices[embedded_entity.name] = \
                    np.arange(embedded_points_index, embedded_points_index+len(embedded_entity.coefficients))
                embedded_points_index += len(embedded_entity.coefficients)
            elif type(embedded_entity) is BSplineSubSet:
                embedded_points.append(embedded_entity.get_coefficients().value)
                embedded_points_indices[embedded_entity.name] = \
                    np.arange(embedded_points_index, embedded_points_index+len(embedded_entity.get_coefficients()))
                embedded_points_index += len(embedded_entity.get_coefficients())
            else:
                raise Exception("Please pass in a valid embedded entity type.")
        embedded_points = np.hstack(embedded_points)
        self.embedded_points = embedded_points
        self.embedded_points_indices = embedded_points_indices

        embedded_points_parametric_coordinates = self.project(points=embedded_points, grid_search_density=15)
        self.evaluation_map = self.compute_evaluation_map(parametric_coordinates=embedded_points_parametric_coordinates, 
                                                          expand_map_for_physical=True)


    def evaluate(self, coefficients:m3l.Variable, plot:bool=False) -> m3l.Variable:
        '''
        Takes in the FFD block coefficients and evaluates the embedded points.

        Parameters
        ----------
        coefficients : m3l.Variable
            The coefficients of the FFD block.
        plot : bool
            A boolean on whether or not to plot the FFD block.

        Returns
        -------
        updated_points : m3l.Variable
            The embedded points.
        '''
        # Perform update
        self.coefficients = coefficients

        updated_points = m3l.matvec(map=self.evaluation_map, x=coefficients)

        outputs = {}

        # For given objects, assign updated points/coefficients accordingly
        for embedded_entity in self.embedded_entities:
            updated_embedded_entity_points = updated_points[self.embedded_points_indices[embedded_entity.name]]

            if type(embedded_entity) is m3l.Variable:
                embedded_entity = updated_embedded_entity_points  # Think about this one. This won't update pointer to object with this pointer.
                outputs[embedded_entity.name] = updated_embedded_entity_points
            elif type(embedded_entity) is BSpline or type(embedded_entity) is BSplineSet or type(embedded_entity) is Geometry:
                embedded_entity.coefficients = updated_embedded_entity_points
                outputs[embedded_entity.name + '_coefficients'] = updated_embedded_entity_points
            elif type(embedded_entity) is BSplineSubSet:
                # embedded_entity.assign_coefficients(updated_embedded_entity_points)   # NOTE: DON'T DO THIS. CAUSES CYCLIC GRAPH. 
                outputs[embedded_entity.name + '_coefficients'] = updated_embedded_entity_points

        self.embedded_points = updated_points.value

        if plot:
            self.plot(plot_embedded_points=True, plot_types=['surface','point_cloud'], opacity=0.3, show=True)

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]
        else:
            return outputs
        

    def plot(self, plot_embedded_points:bool=True, point_types:str=['evaluated_points'],
             plot_types:list=['surface','point_cloud'], opacity:float=0.3, color:str='#00629B', surface_texture:str="",
             additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the FFD block and optionally the points embedded within.
        '''
        plotting_elements = additional_plotting_elements.copy()

        plotting_elements = super().plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color,
                                         surface_texture=surface_texture, additional_plotting_elements=plotting_elements, show=False)

        if plot_embedded_points:
            for embedded_entity in self.embedded_entities:
                if type(embedded_entity) is m3l.Variable:
                    updated_entity_points = self.embedded_points[self.embedded_points_indices[embedded_entity.name]]
                    plotting_elements.append(vedo.Points(updated_entity_points, r=5, c='green'))
                elif type(embedded_entity) is BSpline or type(embedded_entity) is BSplineSet or type(embedded_entity) is Geometry \
                    or type(embedded_entity) is BSplineSubSet:
                    plotting_elements.append(embedded_entity.plot(plot_types=['surface'], opacity=1., 
                                                                  additional_plotting_elements=plotting_elements, show=False))

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, f'Free Form Deformation Block: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements
    


if __name__ == "__main__":
    pass

