import numpy as np
import scipy.sparse as sps
import array_mapper as am

# from vedo import Points, Plotter, Mesh
import vedo

from caddee.caddee_core.system_representation.utils.io.step_io import read_openvsp_stp, write_step, read_gmsh_stp
from caddee.caddee_core.system_representation.utils.io.iges_io import read_iges, write_iges

class Geometry:
    '''
    A geometric object.

    Parameters
    ----------


    '''
    def __init__(self, primitives:dict={}, primitive_indices:dict={}) -> None:
        self.primitives = primitives
        self.primitive_indices = primitive_indices
        self.control_points = None  # Will be instantiated during assemble()

    '''
    Returns the primtiive objects that include the search name(s) in the primitive name.
    '''
    def get_primitives(self, search_names=[]):
        primitives = []
        for primitive_name in self.primitives.keys():
            for search_name in search_names:
                if search_name in primitive_name:
                    primitives.append(self.primitives[primitive_name])
                    break

        return primitives


    def project(self, points:np.ndarray, targets:tuple=None, direction:np.ndarray=None,
                grid_search_n:int=25, max_iterations=100, offset:np.ndarray=None, plot:bool=False):
        '''
        Projects points onto the geometry.
        '''
        #  TODO Consider parallelizing using Numba, or using the FFD method or in Cython.

        if targets is None:
            targets = list(self.primitives.values())

        if len(points.shape) == 1:
            points = points.reshape((1, -1))    # Last axis is reserved for dimensionality of physical space
        
        num_targets = len(targets)
        projected_points_on_each_target = []
        # Project all points onto each target
        for target in targets:
            if isinstance(target, str):
                target = self.primitives[target]

        for target in targets:   # TODO Parallelize this for loop
            target_projected_points = target.project(points=points, direction=direction, grid_search_n=grid_search_n,
                    max_iter=max_iterations, return_parametric_coordinates=False)
            # projected_points_on_each_target[i] = target_projected_points.value
            projected_points_on_each_target.append(target_projected_points)

        projected_points_on_each_target_numpy = np.zeros(tuple((num_targets,)) + points.shape)
        for i in range(num_targets):
            projected_points_on_each_target_numpy[i] = projected_points_on_each_target[i].value
        
        # Compare results across targets to keep best result
        distances = np.linalg.norm(projected_points_on_each_target_numpy - points, axis=-1)   # Computes norm across spatial axis
        closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
        if len(points.shape) == 2:
            num_points = points.shape[0]
        elif len(points.shape) == 1:
            num_points = 1
        else:
            num_points = np.cumprod(points.shape[:-1])[-1]
        flattened_surface_indices = closest_surfaces_indices.flatten()
        num_control_points = np.cumprod(self.control_points.shape[:-1])[-1]
        linear_map = sps.lil_array((num_points, num_control_points))
        projection_receiving_primitives = []
        for i in range(num_points): # for each point, assign the closest the projection result
            target_index = flattened_surface_indices[i]
            receiving_target = targets[target_index]
            if receiving_target not in projection_receiving_primitives:
                projection_receiving_primitives.append(receiving_target)
            receiving_target_control_point_indices = self.primitive_indices[receiving_target.name]
            point_map_on_receiving_target = projected_points_on_each_target[target_index].linear_map[i,:]
            linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

        projection_receiving_primitives = list(targets) # TODO decide if we want this or just the primitives receiving the projection

        linear_map = linear_map.tocsc()
        projected_points = am.array(self.control_points, linear_map=linear_map, offset=offset, shape=points.shape)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(primitives=projection_receiving_primitives, opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            flattened_projected_points = (projected_points.value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_primitive_control_points = vedo.Points(flattened_projected_points, r=12, c='blue')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_control_points)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        return projected_points
            
    def add_input(self, function, connection_name=None, val=None):
        # TODO implement
        # TODO Think about rethinking how GeometricCalcaulations are handled.
        #   -- Same as TC1: Use a library of preset calculation objects
        #   -- Any object is fine as long as it has an evaluate method
        #   -- using ArrayMapper with nonlinear operations
        #   -- Using mini CSDL models or mini OpenMDAO models.
        pass
    
    def add_output(self, output_name):
        # TODO same comments as add_input. Should hopefully be easier.
        pass


    def import_file(self, file_name : str):
        '''
        Imports geometry primitives from a file.

        Parameters
        ----------
        file_name : str
            The name of the file (with path) that containts the geometric information.
        
        '''
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

    def read_openvsp_stp(self, file_name):
        read_openvsp_stp(self, file_name)

    def read_gmsh_stp(self, file_name):
        read_gmsh_stp(self, file_name)

    def read_iges(self,file_name):
        read_iges(self, file_name)

    def write_step(self, file_name, plot=False):
        write_step(self, file_name, plot)

    def write_iges(self, file_name, plot = False):
        write_iges(self, file_name, plot)

    '''
    Collects the primitives into a vectorized format.
    '''
    def assemble(self):
        starting_index = 0
        for primitive in list(self.primitives.values()):
            primitive_control_points = primitive.control_points
            primitive_num_control_points = np.cumprod(primitive.shape[:-1])[-1]
            ending_index = starting_index + primitive_num_control_points
            self.primitive_indices[primitive.name] = np.arange(starting_index, ending_index)
            starting_index = ending_index
            if self.control_points is None:
                self.control_points = primitive_control_points.reshape((primitive_num_control_points, -1))
            else:
                self.control_points = np.vstack((self.control_points, primitive_control_points.reshape((primitive_num_control_points, -1))))

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
        '''
        raise Exception("Geometry.evaluate() is not implemented.")


    def plot_meshes(self, meshes:list, mesh_plot_types:list=['wireframe'], mesh_opacity:float=1., 
                primitives:list=[], primitives_plot_types:list=['mesh'], primitives_opacity:float=0.25,
                additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots a mesh over the geometry.
        '''
        plotting_elements = additional_plotting_elements.copy()
        if not isinstance(meshes, list) and not isinstance(meshes, tuple):
            meshes = [meshes]

        plotter = vedo.Plotter()
        # Create plotting meshes for primitives
        plotting_elements = self.plot(primitives=primitives, plot_types=primitives_plot_types, opacity=primitives_opacity,
                additional_plotting_elements=plotting_elements,show=False)

        for mesh in meshes:
            if type(mesh) is am.MappedArray:
                points = mesh.value
            else:
                points = mesh

            if isinstance(mesh, tuple):
                # Is vector
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
                plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=4).color('blue'))
            if points.shape[0] == 1:  # TODO Handle this more gracefully and generalize beyond surfaces!!
                points = points.reshape((points.shape[1:]))
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
                plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color('blue')
                plotting_elements.append(plotting_mesh.wireframe().linewidth(2))
            
        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Meshes', axes=1, viewup="z", interactive=True)
        else:
            return plotting_elements
    
    def plot(self, primitives:list=[], plot_types:list=['mesh'], opacity:float=1., additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the geometry or a subset of the geometry
        '''

        plotting_elements = additional_plotting_elements.copy()
        if not primitives:  # If empty, plot geometry as a whole
            primitives = list(self.primitives.values())
        if primitives[0] == 'all':
            primitives = list(self.primitives.values())

        for primitive in primitives:
            if isinstance(primitive, str):
                primitive = self.primitives[primitive]

            plotting_elements = primitive.plot(plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements, show=False)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Geometry', axes=1, viewup="z", interactive=True)
        else:
            return plotting_elements


class BSplineGeometry:
    '''
    TODO: A B-spline geometry object.
    The current Geometry object will likely be turned into a BSplineGeometry object
    and a new general Geometry object will be made.

    Parameters
    ----------


    '''
    def __init__(self) -> None:
        pass