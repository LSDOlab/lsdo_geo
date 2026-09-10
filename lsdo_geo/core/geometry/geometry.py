from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pickle
from dataclasses import dataclass
from pathlib import Path
# import pickle
import csdl_alpha as csdl
# from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
# from lsdo_geo.splines.b_splines.b_spline_sub_set import BSplineSubSet
import lsdo_function_spaces as lfs
import lsdo_geo as lg
import pyvista as pv
from typing import Optional, Union, Sequence

@dataclass
class Geometry(lfs.FunctionSet):
    representations:dict[str,lg.Mesh] = None

    def __post_init__(self):
        super().__post_init__()
        if self.representations is None:
            self.representations = {}


    def copy(self):
        '''
        Creates a copy of the geometry
        '''
        function_set = super().copy()
        geometry_copy = Geometry(functions=function_set.functions, function_names=function_set.function_names, name=self.name,
                                    space=function_set.space, representations=self.representations)
        return geometry_copy

    
    def get_function_space(self):
        return self.space
    

    def add_representation(self, representation:lg.Mesh):
        '''
        Adds a representation to the geometry.

        Parameters
        ----------
        name : str
            The name of the representation.
        representation : lg.Mesh
            The representation to add.
        '''
        self.representations[representation.name] = representation

    
    def evaluate_representations(self, representations:list[lg.Mesh], plot:bool=False) -> list[csdl.Variable]:
        '''
        Evaluates a representation or a list of representations.

        Parameters
        ----------
        name : str
            The name of the representation.
        plot : bool, optional
            Whether or not to plot the representation.
        '''
        if isinstance(representations, lg.Mesh):
            representations = [representations]

        evaluated_representations = []
        for representation in representations:
            # representation = self.representations[representation.name]
            evaluated_representations.append(representation.evaluate(self, plot=plot))

        if len(evaluated_representations) == 1:
            evaluated_representation = evaluated_representations[0]
            return evaluated_representation
        
        return evaluated_representations
    

    def declare_component(self, function_indices:Optional[Sequence[int]]=None, 
                          function_search_names:Optional[Sequence[str]]=None, 
                          ignore_names:Optional[Sequence[str]]=None, 
                          name:Optional[str]=None) -> lg.Geometry:
        '''
        Declares a component. This component will point to a sub-set of the entire geometry.

        Parameters
        ----------
        function_names : list[str]
            The names of the functions that make up the component.
        function_search_names : list[str], optional
            The names of the functions to search for. Names of functions will be returned for each B-spline that INCLUDES the search name.
        name : str
            The name of the component.
        '''
        function_set = self.create_subset(function_indices=function_indices, function_search_names=function_search_names, ignore_names=ignore_names, name=name)

        component = lg.Geometry(functions=function_set.functions, function_names=function_set.function_names, name=name, 
                                space=function_set.space)
        return component
    
    def create_component_copy(self, function_indices:list[int]=None, function_search_names:list[str]=None, name:str=None) -> lg.Geometry:
        '''
        Declares a component. This component will point to a sub-set of the entire geometry.

        Parameters
        ----------
        function_names : list[str]
            The names of the functions that make up the component.
        function_search_names : list[str], optional
            The names of the functions to search for. Names of functions will be returned for each B-spline that INCLUDES the search name.
        name : str
            The name of the component.
        '''
        component = self.create_subset(function_indices=function_indices, function_search_names=function_search_names, name=name)
        component_copy = component.copy()
        return component_copy
    
    # def copy(self) -> lg.Geometry:
    #     '''
    #     Copies the function set.

    #     Returns
    #     -------
    #     function_set : lfs.FunctionSet
    #         The copied function set.
    #     '''
    #     functions = {i:function.copy() for i, function in self.functions.items()}
    #     function_set = lg.Geometry(functions=functions, function_names=self.function_names, name=self.name)
    #     return function_set

    # def import_geometry(self, file_name:str):
    #     '''
    #     Imports geometry from a file.

    #     Parameters
    #     ----------
    #     file_name : str
    #         The name of the file (with path) that containts the geometric information.
    #     '''
    #     from lsdo_geo.splines.b_splines.b_spline_functions import import_file, create_b_spline_set
    #     b_splines = import_file(file_name)
    #     b_spline_set = create_b_spline_set(self.name, b_splines)

    #     self.space = b_spline_set.space
    #     self.coefficients = b_spline_set.coefficients
    #     self.num_physical_dimensions = b_spline_set.num_physical_dimensions
    #     self.coefficient_indices = b_spline_set.coefficient_indices
    #     self.connections = b_spline_set.connections


    def rotate(self, rotation_origin:Union[csdl.Variable, npt.NDArray[np.float64]], axis_vector:Union[csdl.Variable, npt.NDArray[np.float64]],
               angles:Union[csdl.Variable, npt.NDArray[np.float64], float], function_indices:Optional[list[int]]=None, 
               units:Optional[str]='radians') -> None:
        '''
        Rotates the geometry about an axis.

        Parameters
        ----------
        rotation_origin : csdl.Variable
            The origin of the axis of rotation.
        axis_vector : csdl.Variable
            The vector of the axis of rotation.
        angles : csdl.Variable
            The angle of rotation.
        function_indices : list[int]
            The indices of the functions to rotate.
        units : str
            The units of the angle of rotation. {degrees, radians}
        '''
        from lsdo_geo.core.geometry.geometry_functions import rotate as rotate_function
        if units == 'degrees':
            angles = angles * np.pi / 180.
            units = 'radians'
        elif units == 'radians':
            pass
        else:
            raise ValueError(f'Invalid units {units}.')
        
        if function_indices is None:
            function_indices = list(self.functions.keys())
        if isinstance(function_indices, int):
            function_indices = [function_indices]
        if not isinstance(function_indices, list):
            raise ValueError(f'The function indices must be a list of int, received {type(function_indices)}')
        
        if isinstance(rotation_origin, np.ndarray):
            rotation_origin = csdl.Variable(shape=rotation_origin.shape, value=rotation_origin)
        # if type(axis_vector) is np.ndarray:
        #     axis_vector = csdl.Variable(shape=axis_vector.shape, value=axis_vector)
        if type(angles) is np.ndarray:
            angles = csdl.Variable(shape=angles.shape, value=angles)

        # # Unvectorized:
        # for function_index in function_indices:
        #     function = self.functions[function_index]
        #     rotated_coefficients = rotate_function(
        #         points=function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])), 
        #         rotation_origin=rotation_origin, axis_vector=axis_vector, angles=angles, units=units
        #     )
        #     function.coefficients = rotated_coefficients.reshape(function.coefficients.shape)

        # Vectorized:
        if len(function_indices) == 1:
            function = self.functions[function_indices[0]]
            rotated_coefficients = rotate_function(
                points=function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])), 
                rotation_origin=rotation_origin, axis_vector=axis_vector, angles=angles, units=units
            )
            function.coefficients = rotated_coefficients.reshape(function.coefficients.shape)
        else:
            stacked_coefficients = []
            for function_index in function_indices:
                function = self.functions[function_index]
                stacked_coefficients.append(function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])))
            stacked_coefficients = csdl.vstack(stacked_coefficients)

            rotated_coefficients = rotate_function(
                points=stacked_coefficients, 
                rotation_origin=rotation_origin, axis_vector=axis_vector, angles=angles, units=units
            )

            counter = 0
            for i, function_index in enumerate(function_indices):
                function = self.functions[function_index]
                num_coefficient_points = function.coefficients.size // function.coefficients.shape[-1]
                function.coefficients = rotated_coefficients[counter:counter+num_coefficient_points,:].reshape(function.coefficients.shape)
                counter += num_coefficient_points


    def translate(self, translation:Union[csdl.Variable, npt.NDArray[np.float64]], function_indices:Optional[list[int]]=None) -> None:
        '''
        Translates the geometry.

        Parameters
        ----------
        translation : csdl.Variable
            The translation vector.
        function_indices : list[int]
            The indices of the functions to translate.
        '''
        if function_indices is None:
            function_indices = list(self.functions.keys())
        if isinstance(function_indices, int):
            function_indices = [function_indices]
        if not isinstance(function_indices, list):
            raise ValueError(f'The function indices must be a list of int, received {type(function_indices)}')

        if isinstance(translation, np.ndarray):
            translation = csdl.Variable(shape=translation.shape, value=translation)

        # # Unvectorized:
        # for function_index in function_indices:
        #     function = self.functions[function_index]
        #     translated_coefficients = translate_function(
        #         points=function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])), 
        #         translation=translation
        #     )
        #     function.coefficients = translated_coefficients.reshape(function.coefficients.shape)

        # Vectorized:
        if len(function_indices) == 1:
            function = self.functions[function_indices[0]]
            translated_coefficients = function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])) + translation
            function.coefficients = translated_coefficients.reshape(function.coefficients.shape)
        else:
            stacked_coefficients = []
            for function_index in function_indices:
                function = self.functions[function_index]
                stacked_coefficients.append(function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])))
            stacked_coefficients = csdl.vstack(stacked_coefficients)

            if len(translation.shape) == 1:
                expanded_translation = csdl.expand(translation, stacked_coefficients.shape, 'i->ji',)
                translated_coefficients = stacked_coefficients + expanded_translation
            elif translation.shape == stacked_coefficients.shape:
                translated_coefficients = stacked_coefficients + translation
            else:
                raise ValueError(f'Translation shape {translation.shape} not compatible with stacked coefficients shape {stacked_coefficients.shape}.')
                

            counter = 0
            for i, function_index in enumerate(function_indices):
                function = self.functions[function_index]
                num_coefficient_points = function.coefficients.size // function.coefficients.shape[-1]
                function.coefficients = translated_coefficients[counter:counter+num_coefficient_points,:].reshape(function.coefficients.shape)
                counter += num_coefficient_points


    def rotate_using_quaternion(self, rotation_origin:Union[csdl.Variable,npt.NDArray[np.float64]], 
                                quaternion:Union[csdl.Variable,npt.NDArray[np.float64]],
                                function_indices:Optional[list[int]]=None):
        '''
        Rotates the geometry using a quaternion.

        parameters
        ----------
        rotation_origin : Union[csdl.Variable,npt.NDArray[np.float64]]
            The origin of the rotation axis.
        quaternion : Union[csdl.Variable,npt.NDArray[np.float64]]
            The quaternion representing the rotation.
        function_indices : Optional[list[int]]
            The indices of the functions to rotate.
        '''
        from lsdo_geo import apply_quaternion_rotation
        if function_indices is None:
            function_indices = list(self.functions.keys())
        if isinstance(function_indices, int):
            function_indices = [function_indices]
        if not isinstance(function_indices, list):
            raise ValueError(f'The function indices must be a list of int, received {type(function_indices)}')

        if len(function_indices) == 1:
            function = self.functions[function_indices[0]]
            rotated_coefficients = apply_quaternion_rotation(
                points=function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])), 
                rotation_origin=rotation_origin, quaternion=quaternion
            )
            function.coefficients = rotated_coefficients.reshape(function.coefficients.shape)

        else:
            stacked_coefficients = []
            for function_index in function_indices:
                function = self.functions[function_index]
                stacked_coefficients.append(function.coefficients.reshape((function.coefficients.size // function.coefficients.shape[-1], function.coefficients.shape[-1])))
            stacked_coefficients = csdl.vstack(stacked_coefficients)

            rotated_coefficients = apply_quaternion_rotation(
                points=stacked_coefficients, 
                rotation_origin=rotation_origin, quaternion=quaternion
            )

            counter = 0
            for i, function_index in enumerate(function_indices):
                function = self.functions[function_index]
                num_coefficient_points = function.coefficients.size // function.coefficients.shape[-1]
                function.coefficients = rotated_coefficients[counter:counter+num_coefficient_points,:].reshape(function.coefficients.shape)
                counter += num_coefficient_points


    def plot_meshes(self, meshes:list[csdl.Variable], mesh_plot_types:list[str]=['wireframe'], mesh_opacity:float=1., mesh_color:str='#F5F0E6',
                mesh_color_map:str='jet', mesh_line_width:float=3.,
                function_indices:Optional[list[str]]=None, function_plot_types:list[str]=['function'], function_opacity:float=0.25, function_color:str='#00629B',
                function_color_map:str='jet', function_surface_texture:str="",
                additional_plotting_elements:list=[], camera:Optional[dict]=None, show:bool=True) -> list:
        '''
        Plots a mesh over the geometry.

        Parameters
        ----------
        meshes : list
            A list of meshes to plot.
        mesh_plot_types : list, optional = ['wireframe']
            A list of plot types for each mesh. Options are 'wireframe', 'function', and 'points'.
        mesh_opacity : float, optional = 1.
            The opacity of the mesh.
        mesh_color : str, optional = '#F5F0E6'
            The color of the mesh.
        mesh_color_map : str, optional = 'jet'
            The color map for the mesh.
        mesh_line_width : float, optional = 3.
            The line width of the mesh.
        function_indices : list, optional = None
            A list of indices for which functions to plot.
        function_plot_types : list, optional = ['function']
            A list of plot types for each primitive. Options are 'wireframe', 'function', and 'points'.
        function_opacity : float, optional = 0.25
            The opacity of the function.
        function_color : str, optional = '#00629B'
            The color of the function.
        function_color_map : str, optional = 'jet'
            The color map for the function.
        function_surface_texture : str, optional
            The surface texture for the primitive surfaces.
        additional_plotting_elements : list, optional
            A list of additional plotting elements to plot.
        camera : dict, optional
            A dictionary of camera parameters for PyVista.
        show : bool, optional
            Whether or not to show the plot.

        Returns
        -------
        plotting_elements : list
            A list of the PyVista plotting elements.
        '''
        import lsdo_function_spaces.utils.plotting_functions as pf
        import pyvista as pv
        plotting_elements = additional_plotting_elements.copy()

        if not isinstance(meshes, list) and not isinstance(meshes, tuple):
            meshes = [meshes]

        # Create plotting meshes for the functions/geometry
        plotting_elements = self.plot(point_types=['evaluated_points'], plot_types=function_plot_types, opacity=function_opacity,
                                      color=function_color, color_map=function_color_map,
                                      additional_plotting_elements=plotting_elements, show=False)

        for mesh in meshes:
            if type(mesh) is csdl.Variable:
                points = mesh.value
            else:
                points = mesh

            if isinstance(mesh, tuple):
                # Is vector, so draw an arrow
                processed_points = ()
                for point in mesh:
                    if type(point) is csdl.Variable:
                        processed_points = processed_points + (point.value,)
                    else:
                        processed_points = processed_points + (point,)
                start = np.asarray(processed_points[0]).reshape((3,))
                direction = np.asarray(processed_points[1]).reshape((3,))
                norm = np.linalg.norm(direction)
                if norm > 0:
                    arrow = pv.Arrow(start=start, direction=direction, scale=norm)
                    plotting_elements.append({"mesh": arrow, "kwargs": dict(color=mesh_color)})
                continue

            if 'point_cloud' in mesh_plot_types:
                plotting_elements = pf.plot_points(points, opacity=mesh_opacity, color=mesh_color, color_map=mesh_color_map, 
                                                   additional_plotting_elements=plotting_elements, show=False)

            if points.shape[0] == 1:
                points = points.reshape((points.shape[1:]))

            if len(points.shape) == 2:  # If it's a curve
                plotting_elements = pf.plot_curve(points, opacity=mesh_opacity, color=mesh_color, 
                                                 color_map=mesh_color_map, line_width=mesh_line_width, 
                                                 additional_plotting_elements=plotting_elements, show=False)
                if 'wireframe' in mesh_plot_types:
                    plotting_elements = pf.plot_points(points, opacity=mesh_opacity, color=mesh_color,
                                                       size=6., additional_plotting_elements=plotting_elements, show=False)
                continue

            if ('surface' in mesh_plot_types or 'wireframe' in mesh_plot_types) and len(points.shape) == 3:  # If it's a surface
                surface_types = [t for t in mesh_plot_types if t in ['surface', 'wireframe']]
                # Map 'surface' to 'function' for lfs.plot_surface
                mapped_types = ['function' if t == 'surface' else t for t in surface_types]
                plotting_elements = pf.plot_surface(points, plot_types=mapped_types, opacity=mesh_opacity,
                                                    color=mesh_color, color_map=mesh_color_map,
                                                    line_width=mesh_line_width,
                                                    additional_plotting_elements=plotting_elements, show=False)

        if show:
            pf.show_plot(plotting_elements, 'Meshes', axes=True, view_up="z", interactive=True, camera=camera if camera else {})

        return plotting_elements
    

    def plot_2d_mesh(self, mesh):
        pass

    def export_iges(self, file_name:str):
        '''
        Exports the geometry to an IGES file.

        Parameters
        ----------
        file_name : str
            The name of the file to export to.
        '''
        """
        Write the surface to IGES format
        Parameters
        ----------
        fileName : str
            File name of iges file. Should have .igs extension.
        """
        f = open(file_name, 'w')
        print('Exporting', file_name)
        #TODO Change to correct information
        f.write('                                                                        S      1\n')
        f.write('1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n')
        f.write('7H128-000,1.,6,1HM,8,0.016,15H19970830.165254, 0.0001,0.,               G      2\n')
        f.write('21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n')
        f.write('13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n')
        Dcount = 1
        Pcount = 1
        for surf in self.functions.values():
            space = surf.space
            paraEntries = 13 + (len(space.knot_indices[0])) + (len(space.knot_indices[1])) + space.coefficients_shape[0] * space.coefficients_shape[1] + 3 * space.coefficients_shape[0] * space.coefficients_shape[1] + 1
            paraLines = (paraEntries - 10) // 3 + 2
            if np.mod(paraEntries - 10, 3) != 0:
                paraLines += 1
            f.write("     128%8d       0       0       1       0       0       000000001D%7d\n" % (Pcount, Dcount))
            f.write(
            "     128       0       2%8d       0                               0D%7d\n" % (paraLines, Dcount + 1)
            )
            Dcount += 2
            Pcount += paraLines
        Pcount  = 1
        counter = 1
        for surf in self.functions.values():
            space = surf.space
            f.write(
                "%10d,%10d,%10d,%10d,%10d,          %7dP%7d\n"
                % (128, space.coefficients_shape[0] - 1, space.coefficients_shape[1] - 1, space.degree[0], space.degree[1], Pcount, counter)
            )
            counter += 1
            f.write("%10d,%10d,%10d,%10d,%10d,          %7dP%7d\n" % (0, 0, 1, 0, 0, Pcount, counter))

            counter += 1
            pos_counter = 0
            if isinstance(space.knots, (tuple, list)):
                knots_u = space.knots[0]
                knots_v = space.knots[1]
            else:
                knots_u = space.knots[space.knot_indices[0]]
                knots_v = space.knots[space.knot_indices[1]]
            for i in range(len(knots_u)):
                pos_counter += 1
                f.write("%20.12g," % (np.real(knots_u[i])))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0

            for i in range(len(knots_v)):
                pos_counter += 1
                f.write("%20.12g," % (np.real(knots_v[i])))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0

            for i in range(space.coefficients_shape[0] * space.coefficients_shape[1]):
                pos_counter += 1
                f.write("%20.12g," % (1.0))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0

            for j in range(space.coefficients_shape[1]):
                for i in range(space.coefficients_shape[0]):
                    for idim in range(3):
                        pos_counter += 1
                        if isinstance(surf.coefficients, csdl.Variable):
                            coefficients = surf.coefficients.value
                        else:
                            coefficients = surf.coefficients
                        cntrl_pts = np.reshape(coefficients, (space.coefficients_shape[0], space.coefficients_shape[1],3))
                        f.write("%20.12g," % (np.real(cntrl_pts[i, j, idim])))
                        if np.mod(pos_counter, 3) == 0:
                            f.write("  %7dP%7d\n" % (Pcount, counter))
                            counter += 1
                            pos_counter = 0

            for i in range(4):
                pos_counter += 1
                if i == 0:
                    f.write("%20.12g," % (np.real(knots_u[0])))
                if i == 1:
                    f.write("%20.12g," % (np.real(knots_u[-1])))
                if i == 2:
                    f.write("%20.12g," % (np.real(knots_v[0])))
                if i == 3:
                    f.write("%20.12g;" % (np.real(knots_v[-1])))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0
                else:  
                    if i == 3:
                        for j in range(3 - pos_counter):
                            f.write("%21s" % (" "))
                        pos_counter = 0
                        f.write("  %7dP%7d\n" % (Pcount, counter))
                        counter += 1

            Pcount += 2 
        f.write('S%7dG%7dD%7dP%7d%40sT%6s1\n'%(1, 4, Dcount-1, counter-1, ' ', ' '))
        f.close()  
        print('Complete export')

    def export_obj(self, file_name:str):
        '''
        Exports the geometry to an OBJ file.

        Parameters
        ----------
        file_name : str
            The name of the file to export to.
        '''
        
        pass
