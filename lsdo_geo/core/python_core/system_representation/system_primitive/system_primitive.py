import numpy as np
import array_mapper as am

class SystemPrimitive:
    '''
    A primitive for the SystemRepresentation. Contains geometric and material primitives.
    The parametric coordinates of each primitive correspond.
    '''

    def __init__(self, name, geometry_primitive, material_primitives:dict={}) -> None:
        self.name = name
        self.geometry_primitive = geometry_primitive
        self.material_primitives = material_primitives
        self.control_points = {}  # assigned during assemble
        self.shapes = None   # assigned during assemble
        self.assemble()

    def compute_evaluation_map(self, u_vec, v_vec):
        '''
        From the parametric coordinates, create a map that will return both the geometry and material properties at the coordinates.

        TODO implement matrix as described from whiteboard picture.
        '''
        # data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        # row_indices = np.zeros(len(data), np.int32)
        # col_indices = np.zeros(len(data), np.int32)

        # num_control_points = self.shape[0] * self.shape[1]

        # get_basis_surface_matrix(self.order_u, self.shape[0], 0, u_vec, self.knots_u, 
        #     self.order_v, self.shape[1], 0, v_vec, self.knots_v, 
        #     len(u_vec), data, row_indices, col_indices)

        # basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_control_points) )
        
        # return basis0
        pass

    def compute_derivative_evaluation_map(self, u_vec, v_vec):
        '''
        From the parametric coordinates, create a map that will return the 
        derivative of the geometry and material properties w.r.t. parametric coordinates.
        '''
        # data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        # row_indices = np.zeros(len(data), np.int32)
        # col_indices = np.zeros(len(data), np.int32)        

        # num_control_points = self.shape[0] * self.shape[1]

        # get_basis_surface_matrix(self.order_u, self.control_points.shape[0], 1, u_vec, self.knots_u, 
        #     self.order_v, self.control_points.shape[1], 1, v_vec, self.knots_v, 
        #     len(u_vec), data, row_indices, col_indices)

        # basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_control_points) )
        
        # return basis1
        pass

    def compute_second_derivative_evaluation_map(self, u_vec, v_vec):
        '''
        From the parametric coordinates, create a map that will return the 
        second of the derivative geometry and material properties w.r.t. parametric coordinates.
        '''
        # data = np.zeros(len(u_vec) * self.order_u * self.order_v)
        # row_indices = np.zeros(len(data), np.int32)
        # col_indices = np.zeros(len(data), np.int32)      

        # num_control_points = self.shape[0] * self.shape[1]
        
        # get_basis_surface_matrix(self.order_u, self.control_points.shape[0], 2, u_vec, self.knots_u, 
        #     self.order_v, self.control_points.shape[1], 2, v_vec, self.knots_v, 
        #     len(u_vec), data, row_indices, col_indices)

        # basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), num_control_points) )
        
        # return basis2

    def evaluate_points(self, u_vec, v_vec):
        '''
        Evaluates the geometry and material properties at the parametric coordinates.
        '''
        # num_control_points = self.shape[0] * self.shape[1]
        
        # basis0 = self.compute_evaluation_map(u_vec, v_vec)
        # points = basis0.dot(self.control_points.reshape((num_control_points, 3)))

        # return points
        pass

    def evaluate_derivative(self, u_vec, v_vec):
        '''
        Evaluates the derivative of the geometry and material properties at the parametric coordinates.
        '''
        # num_control_points = self.shape[0] * self.shape[1]
        
        # basis1 = self.compute_derivative_evaluation_map(u_vec, v_vec)
        # derivs1 = basis1.dot(self.control_points.reshape((num_control_points, 3)))

        # return derivs1 
        pass

    def evaluate_second_derivative(self, u_vec, v_vec):
        '''
        Evaluates the second derivative of the geometry and material properties at the parametric coordinates.
        '''
        # num_control_points = self.shape[0] * self.shape[1]
        
        # basis2 = self.compute_second_derivative_evaluation_map(u_vec, v_vec)
        # derivs2 = basis2.dot(self.control_points.reshape((num_control_points, 3)))

        # return derivs2
        pass

    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_n:int=50,
                    max_iter:int=100, properties:list=['geometry']):
        '''
        Parameters
        -----------
        points : {np.ndarray, am.MappedArray}
            The points to be projected onto the system.
        targets : list, optional
            The list of primitives to project onto.
        direction : {np.ndarray, am.MappedArray}, optional
            An axis for perfoming projection along an axis. The projection will return the closest point to the axis.
        grid_search_n : int, optional
            The resolution of the grid search prior to the Newton iteration for solving the optimization problem.
        max_iterations : int, optional
            The maximum number of iterations for the Newton iteration.
        properties : list
            The list of properties to be returned (in order) {geometry, parametric_coordinates, (material_name, array_of_properties),...}
        '''
        parametric_coordinates = self.geometry_primitive.project(points=points, direction=direction, grid_search_n=grid_search_n,
                    max_iter=max_iter, return_parametric_coordinates=True)
        # parametric_coordinates_flattened = parametric_coordinates.reshape((-1, parametric_coordinates.shape[-1]))
        
        projection_output = {}

        for property in properties:
            if property == 'parametric_coordinates':
                projection_output[property] = parametric_coordinates
                continue
            elif property == 'geometry':
                primitive = self.geometry_primitive
            elif type(property) is tuple:
                material_name = property[0]
                primitive = self.material_primitives[material_name]
            else:
                raise Exception("Please input a proper property to the projection. If the property is not geometry, please input a tuple of the form" \
                                " ([material_name], [property_name])")
            
            # map = primitive.compute_evaluation_map(parametric_coordinates_flattened[:,0], parametric_coordinates_flattened[:,1])
            map = primitive.compute_evaluation_map(parametric_coordinates[0], parametric_coordinates[1])
            num_control_points = np.cumprod(primitive.control_points.shape[:-1])[-1]
            projected_points = am.array(input=primitive.control_points.reshape((num_control_points,-1)), linear_map=map, shape=points.shape)
            projection_output[property] = projected_points

        if len(projection_output) == 1:
            return list(projection_output.values())[0]
        else:
            return projection_output


    def plot(self, point_types:list=['evaluated_points'], plot_types:list=['mesh', 'wireframe'],
              opacity:float=1., color:str='#00629B', surface_texture:str="",
              additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the geometry of the primitive. Will probably include advanced visualizations for material properties in the future.

        Parameters
        -----------
        points_type : list
            The type of points to be plotted. {evaluated_points, control_points}
        plot_types : list
            The type of plot {mesh, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''

        return self.geometry_primitive.plot(point_types=point_types, plot_types=plot_types, opacity=opacity, color=color, 
                                            surface_texture=surface_texture, additional_plotting_elements=additional_plotting_elements,
                                            show=show)
        

    def assemble(self):
        '''
        Assembles geometry and material control points into a single control points array.
        '''
        self.control_points['geometry'] = self.geometry_primitive.control_points
        for material_property_primitive_name in self.material_primitives:
            material_property_primitive = self.material_primitives[material_property_primitive_name]
            self.control_points[material_property_primitive_name] = material_property_primitive.control_points