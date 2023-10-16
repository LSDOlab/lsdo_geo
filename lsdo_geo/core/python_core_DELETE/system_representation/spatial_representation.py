import numpy as np
import scipy.sparse as sps
import array_mapper as am
import os
from pathlib import Path
import vedo
import pickle
from lsdo_geo.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
from lsdo_geo.caddee_core.system_representation.utils.io.step_io import read_openvsp_stp, write_step, read_gmsh_stp
from lsdo_geo.caddee_core.system_representation.utils.io.iges_io import read_iges, write_iges
from caddee import IMPORTS_FILES_FOLDER

class SpatialRepresentation:
    '''
    The description of the physical system.

    Parameters
    ----------


    '''
    def __init__(self, primitives:dict={}, primitive_indices:dict={}) -> None:
        self.primitives = primitives.copy()     # NOTE: This is one of those "I can't believe it" moments.
        self.primitive_indices = primitive_indices.copy()
        self.coefficients = None  # Will be instantiated during assemble()

        self.inputs = {}
        self.outputs = {}

    
    def get_primitives(self, search_names=[]):
        '''
        Returns the primtiive objects that include the search name(s) in the primitive name.
        '''
        primitives = {}
        for primitive_name in self.primitives.keys():
            for search_name in search_names:
                if search_name in primitive_name:
                    primitives[primitive_name] = self.primitives[primitive_name]
                    break

        return primitives

    def get_geometry_primitives(self, search_names=[]):
        '''
        Returns the geometry primitive objects that include the search name(s) in the primitive name.
        '''
        primitives = self.get_primitives(search_names=search_names)
        geometry_primitives = {}
        for primitive in list(primitives.values()):
            geometry_primitive = primitive.geometry_primitive
            geometry_primitives[primitive.name] = geometry_primitive

        return geometry_primitives


    def project(self, points:np.ndarray, targets:list=None, direction:np.ndarray=None,
                grid_search_density:int=25, max_iterations=100, properties:list=['geometry'],
                offset:np.ndarray=None, plot:bool=False):
        '''
        Projects points onto the system.

        Parameters
        -----------
        points : {np.ndarray, am.MappedArray}
            The points to be projected onto the system.
        targets : list, optional
            The list of primitives to project onto.
        direction : {np.ndarray, am.MappedArray}, optional
            An axis for perfoming projection along an axis. The projection will return the closest point to the axis.
        grid_search_density : int, optional
            The resolution of the grid search prior to the Newton iteration for solving the optimization problem.
        max_iterations : int, optional
            The maximum number of iterations for the Newton iteration.
        properties : list
            The list of properties to be returned (in order) {geometry, parametric_coordinates, (material_name, array_of_properties),...}
        offset : np.ndarray
            An offset to apply after the parametric evaluation of the projection. TODO Fix offset!!
        plot : bool
            A boolean on whether or not to plot the projection result.
        '''
        #  TODO Consider parallelizing using Numba, or using the FFD method or in Cython.

        if targets is None:
            targets = list(self.primitives.values())
        elif type(targets) is dict:
            pass    # get objects is list
        elif type(targets) is list:
            for i, target in enumerate(targets):
                if isinstance(target, str):
                    targets[i] = self.primitives[target]

        if type(points) is am.MappedArray:
            points = points.value
        if type(direction) is am.MappedArray:
            direction = direction.value

        if len(points.shape) == 1:
            points = points.reshape((1, -1))    # Last axis is reserved for dimensionality of physical space
        
        num_targets = len(targets)
        projected_points_on_each_target = []
        # Project all points onto each target
        for target in targets:   # TODO Parallelize this for loop
            target_projected_points = target.project(points=points, direction=direction, grid_search_density=grid_search_density,
                    max_iter=max_iterations, properties=['geometry', 'parametric_coordinates'])
                    # properties are not passed in here because we NEED geometry
            projected_points_on_each_target.append(target_projected_points)

        projected_points_on_each_target_numpy = np.zeros(tuple((num_targets,)) + points.shape)
        for i in range(num_targets):
                projected_points_on_each_target_numpy[i] = projected_points_on_each_target[i]['geometry'].value

        # Compare results across targets to keep best result
        distances = np.linalg.norm(projected_points_on_each_target_numpy - points, axis=-1)   # Computes norm across spatial axis
        closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
        if len(points.shape) == 1:
            num_points = 1
        else:
            num_points = np.cumprod(points.shape[:-1])[-1]
        flattened_surface_indices = closest_surfaces_indices.flatten()
        # num_coefficients = np.cumprod(self.coefficients.shape[:-1])[-1]
        # linear_map = sps.lil_array((num_points, num_coefficients))
        projection_receiving_primitives = []
        projection_outputs = {}
        # for i in range(num_points): # for each point, assign the closest the projection result
        #     target_index = flattened_surface_indices[i]
        #     receiving_target = targets[target_index]
        #     if receiving_target not in projection_receiving_primitives:
        #         projection_receiving_primitives.append(receiving_target)
        #     # receiving_target_control_point_indices = self.primitive_indices[receiving_target.name]
        #     # point_map_on_receiving_target = projected_points_on_each_target[target_index].linear_map[i,:]
        #     # linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

        # for i in range(num_points):
        #     target_index = flattened_surface_indices[i]
        #     receiving_target = targets[target_index]
        #     if receiving_target not in projection_receiving_primitives:
        #         projection_receiving_primitives.append(receiving_target)

        for property in properties:
            num_coefficients = np.cumprod(self.coefficients[property].shape[:-1])[-1]
            linear_map = sps.lil_array((num_points, num_coefficients))

            for i in range(num_points):
                target_index = flattened_surface_indices[i]
                receiving_target = targets[target_index]
                receiving_target_control_point_indices = self.primitive_indices[receiving_target.name][property]
                point_parametric_coordinates = projected_points_on_each_target[target_index]['parametric_coordinates']
                if property == 'geometry':
                    point_map_on_receiving_target = receiving_target.geometry_primitive.compute_evaluation_map(u_vec=np.array([point_parametric_coordinates[0][i]]), 
                                                                                        v_vec=np.array([point_parametric_coordinates[1][i]]))
                else:
                    point_map_on_receiving_target = receiving_target.material_primitives[property].compute_evaluation_map(u_vec=np.array([point_parametric_coordinates[0][i]]), 
                                                                                        v_vec=np.array([point_parametric_coordinates[1][i]]))
                linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

            property_shape = points.shape[:-1] + (self.coefficients[property].shape[-1],)
            property_mapped_array = am.array(self.coefficients[property], linear_map=linear_map.tocsc(), offset=offset, shape=property_shape)
            projection_outputs[property] = property_mapped_array

        projection_receiving_primitives = list(targets)

        # linear_map = linear_map.tocsc()
        # projected_points = am.array(self.coefficients, linear_map=linear_map, offset=offset, shape=points.shape)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(primitives=projection_receiving_primitives, opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            # TODO This will break if geometry is not one of the properties. Fix this.
            flattened_projected_points = (projection_outputs['geometry'].value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_primitive_coefficients = vedo.Points(flattened_projected_points, r=12, c='#00C6D7')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_coefficients)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        if len(projection_outputs) == 1:
            return list(projection_outputs.values())[0]
        else:
            return projection_outputs


    def add_input(self, name, quantity, val=None):
        '''
        Adds an input to the mechanical structure. The design geometry optimization will manipulate
        the mechanical structure in order to achieve this input.
        '''
        self.inputs[name] = (quantity, val)
    
    def add_output(self, name, quantity):
        '''
        Adds an output to the system configuration. The system configuration will recalculate this output each iteration.
        '''
        self.outputs[name] = quantity


    def import_file(self, file_name : str):
        '''
        Imports geometry primitives from a file.

        Parameters
        ----------
        file_name : str
            The name of the file (with path) that containts the geometric information.
        
        '''
        fn = os.path.basename(file_name)
        fn_wo_ext = fn[:fn.rindex('.')]
        coefficients = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_coefficients.pickle'
        primitive_indices = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitive_indices.pickle'
        primitives = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitives.pickle'
        my_file = Path(coefficients) 
        if my_file.is_file():
            with open(primitives, 'rb') as f:
                self.primitives = pickle.load(f)

            with open(primitive_indices, 'rb') as f:
                self.primitive_indices = pickle.load(f)

            with open(coefficients, 'rb') as f:
                self.coefficients = pickle.load(f)

        else:
            file_name = str(file_name)
            if (file_name[-4:].lower() == '.stp') or (file_name[-5:].lower() == '.step'):
                with open(file_name, 'r') as f:
                    if 'CASCADE' in f.read():  # Not sure, could be another string to identify
                        self.read_gmsh_stp(file_name)
                    else: 
                        self.read_openvsp_stp(file_name)
            elif (file_name[-4:].lower() == '.igs') or (file_name[-5:].lower() == '.iges'):
                raise NotImplementedError
                # self.read_iges(file_name) #TODO
            else:
                print("Please input an iges file or a stp file from openvsp.")

            self.assemble()
            save_file_name = os.path.basename(file_name)
            filename_without_ext = save_file_name[:save_file_name.rindex('.')]
            with open(f'imports/{filename_without_ext}_coefficients.pickle', 'wb+') as handle:
                pickle.dump(self.coefficients, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # np.save(f, self.coefficients)
            with open(f'imports/{filename_without_ext}_primitive_indices.pickle', 'wb+') as handle:
                pickle.dump(self.primitive_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'imports/{filename_without_ext}_primitives.pickle', 'wb+') as handle:
                pickle.dump(self.primitives, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def refit_geometry(self, num_coefficients:int=25, fit_resolution:int=50, only_non_differentiable:bool=False, file_name=None):
        import lsdo_geo.primitives.b_splines.b_spline_functions as mfd  # lsdo_manifolds

        if file_name is not None:
            fn = os.path.basename(file_name)
            fn_wo_ext = fn[:fn.rindex('.')]
            coefficients = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_coefficients_{num_coefficients}_{fit_resolution}.pickle'
            primitive_indices = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitive_indices_{num_coefficients}_{fit_resolution}.pickle'
            primitives = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitives_{num_coefficients}_{fit_resolution}.pickle'
            my_file = Path(coefficients) 
        if file_name is not None and my_file.is_file():
            with open(primitives, 'rb') as f:
                self.primitives = pickle.load(f)

            with open(primitive_indices, 'rb') as f:
                self.primitive_indices = pickle.load(f)

            with open(coefficients, 'rb') as f:
                self.coefficients = pickle.load(f)

        else:
            for primitive_name, primitive in self.primitives.items():
                i_should_refit = True
                if only_non_differentiable:
                    raise Warning("Refitting all surfaces for now regardless of differentiability.")
                    # if differentiable:
                    #     i_should_refit = False
                if i_should_refit:
                    self.primitives[primitive_name].geometry_primitive = \
                        mfd.refit_b_spline(b_spline=primitive.geometry_primitive, name=primitive_name, \
                                        num_coefficients=(num_coefficients,), fit_resolution=(fit_resolution,))
                    self.primitives[primitive_name].assemble()
            self.assemble()
            if file_name is not None:
                save_file_name = os.path.basename(file_name)
                filename_without_ext = save_file_name[:save_file_name.rindex('.')]
                with open(f'imports/{filename_without_ext}_coefficients_{num_coefficients}_{fit_resolution}.pickle', 'wb+') as handle:
                    pickle.dump(self.coefficients, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # np.save(f, self.coefficients)
                with open(f'imports/{filename_without_ext}_primitive_indices_{num_coefficients}_{fit_resolution}.pickle', 'wb+') as handle:
                    pickle.dump(self.primitive_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f'imports/{filename_without_ext}_primitives_{num_coefficients}_{fit_resolution}.pickle', 'wb+') as handle:
                    pickle.dump(self.primitives, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_openvsp_stp(self, file_name):
        b_splines = read_openvsp_stp(file_name)
        imported_primitives = {}
        for b_spline in list(b_splines.values()):
            primitive = SystemPrimitive(name=b_spline.name, geometry_primitive=b_spline)
            imported_primitives[primitive.name] = primitive
        self.primitives.update(imported_primitives)
        return imported_primitives

    def read_gmsh_stp(self, file_name):
        read_gmsh_stp(self, file_name)

    def read_iges(self,file_name):
        read_iges(self, file_name)

    def write_step(self, file_name, plot=False):
        write_step(self, file_name, plot)

    def write_iges(self, file_name, plot = False):
        write_iges(self, file_name, plot)

    '''
    Collects the primitives into a collectivized format.
    '''
    def assemble(self):
        self.coefficients = {}
        starting_indices = {}

        for primitive in list(self.primitives.values()):
            # Adding indices for properties that don't already have starting indices to avoid KeyError
            for property_type in list(primitive.coefficients.keys()):
                if property_type not in starting_indices:
                    starting_indices[property_type] = 0

            self.primitive_indices[primitive.name] = {}

            # Adding primitive control points to mechanical structure control points
            for property_type in list(primitive.coefficients.keys()):
                primitive_property_coefficients = primitive.coefficients[property_type]
                primitive_num_coefficients = np.cumprod(primitive_property_coefficients.shape[:-1])[-1] 
                #NOTE: control points should always be (np,ndim)
                ending_index = starting_indices[property_type] + primitive_num_coefficients
                self.primitive_indices[primitive.name][property_type] = np.arange(starting_indices[property_type], ending_index)
                starting_indices[property_type] = ending_index
                if property_type in self.coefficients:
                    self.coefficients[property_type] = np.vstack((self.coefficients[property_type], 
                                                                    primitive_property_coefficients.reshape((primitive_num_coefficients,-1))))
                else:
                    self.coefficients[property_type] = primitive_property_coefficients.reshape((primitive_num_coefficients, -1))

        # starting_index = 0
        # for primitive in list(self.primitives.values()):
        #     primitive_coefficients = primitive.coefficients
        #     primitive_num_coefficients = np.cumprod(primitive_coefficients.shape[:-1])[-1]    # control points should always be (np,ndim)
        #     ending_index = starting_index + primitive_num_coefficients
        #     self.primitive_indices[primitive.name] = np.arange(starting_index, ending_index)
        #     starting_index = ending_index
        #     if self.coefficients is None:
        #         self.coefficients = primitive_coefficients.reshape((primitive_num_coefficients, -1))
        #     else:
        #         self.coefficients = np.vstack((self.coefficients, primitive_coefficients.reshape((primitive_num_coefficients, -1))))

    def update(self, updated_coefficients:np.ndarray, primitive_names:list=['all']):
        '''
        Updates the control points of the mechanical structure or a portion of the mechanical structure.

        Parameters
        -----------
        updated_coefficients : {np.ndarray, dict}
            The array or dictionary of new control points for the mechanical structure.
            An array updates geometric control points while a dictionary of the form
            {property_name:array_of_values} can be used to update any general set of properties.
        primitive_names : list=['all'], optional
            The list of primitives to be updated with the specified values.
        '''
        if primitive_names == ['all']:
            primitive_names = list(self.primitives.keys())
        
        if type(updated_coefficients) is np.ndarray or \
            type(updated_coefficients) is am.MappedArray:
            starting_index = 0
            for primitive_name in primitive_names:
                primitive = self.primitives[primitive_name]
                property_name = 'geometry'
                indices = self.primitive_indices[primitive_name][property_name]
                ending_index = starting_index + len(indices)
                self.coefficients[property_name][indices] = updated_coefficients[starting_index:ending_index]

                self.primitives[primitive_name].geometry_primitive.coefficients = \
                    updated_coefficients[starting_index:ending_index]

                starting_index = ending_index
        elif type(updated_coefficients) is dict:
            starting_indices = {}
            for primitive_name in primitive_names:
                primitive = self.primitives[primitive_name]
                for property_name in list(updated_coefficients.keys()):
                    indices = self.primitive_indices[primitive_name][property_name]
                    ending_index = starting_indices[property_name] + len(indices)
                    self.coefficients[property_name][indices] = \
                        updated_coefficients[property_name][starting_indices[property_name]:ending_index]

                    if property_name == 'geometry':
                        self.primitives[primitive_name].geometry_primitive.coefficients = \
                            updated_coefficients[property_name][starting_indices[property_name]:ending_index]
                    else:
                        self.primitives[primitive_name].material_primitives[property_name].coefficients = \
                            updated_coefficients[property_name][starting_indices[property_name]:ending_index]

                    starting_indices[property_name] = ending_index
        else:
            raise Exception("When updating, please pass in an array (for purely geometric update) \
                            or dictionary (for general property update)")


    '''
    Completes all setup steps. This may be deleted later.
    '''
    def setup(self):
        pass

    
    def evaluate(self):
        '''
        Evaluates the object. I had a reason for this but I need to remember what it will evaluate. (Actuations? Geo DV?).
        Actuate should probably be a method called "actuate" or something to that effect if it's a method here.
        After some thought, I think it would make sense for this class to not have an evaluate method since the Geometry
        is a true object that is a container for information and not really a model. Any methods should reflect what the object
        is doing.

        NOTE: I think (or at least I now think) that this evaluate is for meshes and outputs as a whole.
        '''
        raise Exception("Geometry.evaluate() is not implemented.")


    def plot_meshes(self, meshes:list, mesh_plot_types:list=['wireframe'], mesh_opacity:float=1., mesh_color:str='#F5F0E6',
                primitives:list=[], primitives_plot_types:list=['mesh'], primitives_opacity:float=0.25, primitives_color:str='#00629B',
                primitives_surface_texture:str="", additional_plotting_elements:list=[], camera:dict=None, show:bool=True):
        '''
        Plots a mesh over the geometry.
        '''
        plotting_elements = additional_plotting_elements.copy()
        if not isinstance(meshes, list) and not isinstance(meshes, tuple):
            meshes = [meshes]

        # Create plotting meshes for primitives
        plotting_elements = self.plot(primitives=primitives, plot_types=primitives_plot_types, opacity=primitives_opacity,
                                      color=primitives_color, surface_texture=primitives_surface_texture,
                                      additional_plotting_elements=plotting_elements,show=False)

        for mesh in meshes:
            if type(mesh) is am.MappedArray:
                points = mesh.value
            else:
                points = mesh

            if isinstance(mesh, tuple):
                # Is vector, so draw an arrow
                processed_points = ()
                for point in mesh:
                    if type(point) is am.MappedArray:
                        processed_points = processed_points + (point.value,)
                    else:
                        processed_points = processed_points + (point,)
                arrow = vedo.Arrow(tuple(processed_points[0].reshape((-1,))), tuple((processed_points[0] + processed_points[1]).reshape((-1,))), s=0.05)
                plotting_elements.append(arrow)
                continue

            if 'point_cloud' in mesh_plot_types:
                num_points = np.cumprod(points.shape[:-1])[-1]
                plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=4).color('#00C6D7'))

            if points.shape[0] == 1:
                points = points.reshape((points.shape[1:]))

            if len(points.shape) == 2:  # If it's a curve
                from vedo import Line
                plotting_elements.append(Line(points).color(mesh_color).linewidth(3))
                
                if 'wireframe' in mesh_plot_types:
                    num_points = np.cumprod(points.shape[:-1])[-1]
                    plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=12).color(mesh_color))
                continue

            if ('mesh' in mesh_plot_types or 'wireframe' in mesh_plot_types) and len(points.shape) == 3:
                num_points_u = points.shape[0]
                num_points_v = points.shape[1]
                num_points = num_points_u*num_points_v
                vertices = []
                faces = []
                for u_index in range(num_points_u):
                    for v_index in range(num_points_v):
                        vertex = tuple(points[u_index,v_index,:])
                        vertices.append(vertex)
                        if u_index != 0 and v_index != 0:
                            face = tuple((
                                (u_index-1)*num_points_v+(v_index-1),
                                (u_index-1)*num_points_v+(v_index),
                                (u_index)*num_points_v+(v_index),
                                (u_index)*num_points_v+(v_index-1),
                            ))
                            faces.append(face)

                plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color('lightblue')
            if 'mesh' in mesh_plot_types:
                plotting_elements.append(plotting_mesh)
            if 'wireframe' in mesh_plot_types:
                # plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color('blue')
                plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color(mesh_color) # Default is UCSD Sand
                plotting_elements.append(plotting_mesh.wireframe().linewidth(3))
            
        if show:
            

            plotter = vedo.Plotter(size=(3200,2000))
            # plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Meshes', axes=1, viewup="z", interactive=True, camera=camera)
            return plotting_elements
        else:
            return plotting_elements
    
    def plot(self, primitives:list=[], point_types=['evaluated_points'], plot_types:list=['mesh'],
             opacity:float=1., color:str='#00629B', surface_texture:str="",
             additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the geometry or a subset of the geometry.
        
        Parameters
        -----------
        primitives : list
            The list of primitives to be plotted. This can be the primitive names or the objects themselves.
        points_type : list
            The type of points to be plotted. {evaluated_points, coefficients}
        plot_types : list
            The type of plot {mesh, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code for the plotting color of the primitives.
        surface_texture : str {"", "metallic", "glossy", "ambient",... see Vedo for more options}
            The surface texture for the primitive surfaces. (determines how light bounces off)
            More options: https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''

        plotting_elements = additional_plotting_elements.copy()
        if not primitives:  # If empty, plot geometry as a whole
            primitives = list(self.primitives.values())        
        if primitives[0] == 'all':
            primitives = list(self.primitives.values())
        if primitives[0] == 'none':
            return plotting_elements

        for primitive in primitives:
            if isinstance(primitive, str):
                primitive = self.primitives[primitive]

            plotting_elements = primitive.plot(point_types=point_types, plot_types=plot_types,
                                               opacity=opacity, color=color, surface_texture=surface_texture,
                                               additional_plotting_elements=plotting_elements, show=False)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Geometry', axes=0, viewup="z")
            return plotting_elements
        else:
            return plotting_elements


''' Leaning towards deleting this and creating a unified way to handle primitives. '''
# class BSplineGeometry:
#     '''
#     TODO: A B-spline geometry object.
#     The current Geometry object will likely be turned into a BSplineGeometry object
#     and a new general Geometry object will be made.

#     Parameters
#     ----------


#     '''
#     def __init__(self) -> None:
#         pass