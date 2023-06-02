from dataclasses import dataclass
import numpy as np
import scipy.sparse as sps
import array_mapper as am

from caddee.primitives.bsplines.bspline_curve import BSplineCurve
from caddee.primitives.bsplines.bspline_surface import BSplineSurface
from caddee.primitives.bsplines.bspline_volume import BSplineVolume

import vedo

# TODO consider adding constructor to construct section propery map here
@dataclass
class Parameter:
    '''
    Parameters:
    ----------------
    property_type : str
        The type of property that is being parameterized (for example: scale_v)
    order: int
        The degree of the bspline curve to represent the parameter.
    num_dof: int
        The number of degrees of freedom for the parameter (control points for bspline).
    value: np.ndarray
        The prescribed value for parameter
    connection_name: str
        The name that be used to create a csdl hanging input.
    cost_factor: float
        The cost weighting on using the parameter when achieving geometric inputs and constraints.
    '''
    property_type : str
    order : int = 4
    num_dof : int = 10
    value : np.ndarray = None
    connection_name : str = None
    cost_factor : float = 1.



class FFDBlock:
    '''
    Creates an FFD block. The idea is the block must be spatially continuous, but can be comprised of multiple primitives.
    This class will be implemented further as FFD is generalized.

    Parameters
    -------------
    name : str
        The name of the FFD block.
    primitives : list of primitive objects (such as instance of BSplineVolume)
        The primitive(s) that will represent the FFD block.
    embedded_entities : list of entity objects (such a Components or instances of BSplineSurface)
        The primitivies that will be manipulated/deformed by the free form deformation block.
    '''
    def __init__(self, name:str, primitives:dict={}, embedded_entities:dict={}) -> None:
        self.name = name
        self.primitives = primitives.copy()
        self.embedded_entities = embedded_entities.copy()

        self.control_points = None  # nodes and control points are generalized to control points
        self.map = None
        self.embedded_points = None # This will be a MappedArray that has self.control_points as input and self.map as map


# class SectionedRectangularBSplineGeometricFFDBlock(FFDBlock):
class SRBGFFDBlock(FFDBlock):
    '''
    Creates a Sectioned Rectangular B-spline Geometric FFD block
    for geometric manipulation by manipulating the "sections" of the rectangular B-spline FFD block.
    '''
    def __init__(self, name:str, primitive=None, embedded_entities:dict={}, parameters:dict={}) -> None:
        super().__init__(name, primitives={primitive.name: primitive}, embedded_entities=embedded_entities)
        # attributes that define the FFD block
        self.primitive = primitive
        self.parameters = parameters.copy()

        # attributes that define the FFD block's initial properties for use in transformations
        self.section_origins = None
        self.control_points_section_frame = None
        self.initial_scale_v = None
        self.initial_scale_w = None

        # maps that will be used to perform the free form deformations
        self.free_affine_section_properties_map = None
        self.prescribed_section_properties_map = None
        self.rotational_section_properties_map = None
        self.control_points_affine_map = None
        self.embedded_entities_map = None
        self.cost_matrix = None

        # convenience attributes for constructing the maps that define details about the FFD block
        self.property_names = ['rotation_u', 'rotation_v', 'rotation_w', 'translation_u', ' translation_v', 'translation_w', 'scale_u', 'scale_v', 'scale_w']
        self.affine_property_names = self.property_names[3:]
        self.rotational_property_names = self.property_names[:3]
        self.num_properties = len(self.property_names)
        self.num_rotational_properties = len(self.rotational_property_names)
        self.num_affine_properties = len(self.affine_property_names)
        self.num_scaling_properties = 3

        self.num_control_points = np.cumprod(primitive.shape[:-1])[-1]
        self.num_sections = self.primitive.shape[0]
        self.num_control_points_per_section = np.cumprod(primitive.shape[1:-1])[-1]
        self.num_affine_dof = 0
        self.num_affine_free_dof = 0
        self.num_affine_prescribed_dof = 0
        self.num_rotational_dof = 0
        self.num_dof = 0
        self.num_affine_section_properties = self.num_sections*self.num_affine_properties
        self.num_rotational_section_properties = self.num_sections*self.num_rotational_properties
        self.num_affine_free_control_points = 0

        # attributes for storing current states
        self.free_affine_dof = None
        self.prescribed_affine_dof = None
        self.rotational_dof = None
        self.affine_section_properties = None
        self.translations = None
        self.rotational_section_properties = None
        self.affine_deformed_control_points = None
        self.rotated_control_points_local_frame = None
        self.control_points = self.primitive.control_points.reshape((self.num_control_points, -1))

    
    def embed_entity(self, name:str, entity):
        self.embedded_entities[name] = entity
        

    def add_parameter(self, name:str, parameter:Parameter):
        self.parameters[name] = parameter

        # WARNING: This change makes all parameters be prescribed instead of constant. Is this ok?
        if parameter.connection_name is None:
            parameter.connection_name = name

        if parameter.property_type == 'rotation_u' or \
            parameter.property_type == 'rotation_v' or \
            parameter.property_type == 'rotation_w':
            self.num_rotational_dof += parameter.num_dof
            self.num_dof += parameter.num_dof
        elif parameter.property_type == 'translation_u' or \
            parameter.property_type == 'translation_v' or \
            parameter.property_type == 'translation_w' or \
            parameter.property_type == 'scale_v' or \
            parameter.property_type == 'scale_w':
            
            self.num_affine_dof += parameter.num_dof
            self.num_dof += parameter.num_dof

            if parameter.value is not None or parameter.connection_name is not None:
                self.num_affine_prescribed_dof += parameter.num_dof
            else:
                self.num_affine_free_dof += parameter.num_dof
                if self.num_affine_free_control_points == 0:
                    self.num_affine_free_control_points = self.num_control_points
                
        else:
            raise Exception("When adding ffd parameter, please pass in specific FFDParameter object.")

    def add_translation_u(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None, cost_factor:float=1.):
        property_type = 'translation_u'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=cost_factor)
        self.add_parameter(name=name, parameter=parameter)

    def add_translation_v(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None, cost_factor:float=1.):
        property_type = 'translation_v'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=cost_factor)
        self.add_parameter(name=name, parameter=parameter)

    def add_translation_w(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None, cost_factor:float=1.):
        property_type = 'translation_w'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=cost_factor)
        self.add_parameter(name=name, parameter=parameter)
    
    def add_scale_v(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None, cost_factor:float=1.):
        property_type = 'scale_v'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=cost_factor)
        self.add_parameter(name=name, parameter=parameter)

    def add_scale_w(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None, cost_factor:float=1.):
        property_type = 'scale_w'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=cost_factor)
        self.add_parameter(name=name, parameter=parameter)

    def add_rotation_u(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None):
        property_type = 'rotation_u'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=None)
        self.add_parameter(name=name, parameter=parameter)

    def add_rotation_v(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None):
        property_type = 'rotation_v'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=None)
        self.add_parameter(name=name, parameter=parameter)

    def add_rotation_w(self, name:str=None, order:int=4, num_dof:int=10, value:np.ndarray=None, connection_name:str=None):
        property_type = 'rotation_w'
        parameter = Parameter(property_type=property_type, order=order, num_dof=num_dof, 
                value=value, connection_name=connection_name, cost_factor=None)
        self.add_parameter(name=name, parameter=parameter)


    def assemble_affine_section_properties_maps(self):
        '''
        Assembles the section properties map from the affine parameters that have been added to the block.
        section properties map maps parameter dofs --> section properties
        '''
        if self.free_affine_section_properties_map is not None:
            return self.free_affine_section_properties_map, self.prescribed_section_properties_map

        num_sections = self.num_sections

        # allocate sparse section properties_map now that we know how many inputs there are
        free_affine_section_properties_map = sps.lil_array((num_sections*self.num_affine_properties, self.num_affine_free_dof))
        prescribed_section_properties_map = sps.lil_array((num_sections*self.num_affine_properties, self.num_affine_prescribed_dof))

        free_parameter_starting_index = 0
        prescribed_parameter_starting_index = 0
        for parameter in list(self.parameters.values()):
            if parameter.property_type == 'translation_u':
                property_index = 0
            elif parameter.property_type == 'translation_v':
                property_index = 1
            elif parameter.property_type == 'translation_w':
                property_index = 2
            # elif parameter.property_type == FFDScaleXParameter:   NOTE: if scale x is added as a parameter, this gets uncommented.
            #     property_name = 'scale_x'
            #     property_index = 3
            elif parameter.property_type == 'scale_v':
                property_index = 4
            elif parameter.property_type == 'scale_w':
                property_index = 5
            elif parameter.property_type == 'rotation_u' or \
                    parameter.property_type == 'rotation_v' or \
                    parameter.property_type == 'rotation_w':
                continue
            else:
                raise Exception(f"Error trying to add a parameter ({self.name}:{parameter}). Please pass in a specific FFDParameter object.")
    
            order = parameter.order
            parameter_num_dof = parameter.num_dof

            # generate section property map
            if parameter.order == 1:    # if constant
                section_property_map = sps.lil_array((num_sections, parameter_num_dof))
                section_property_map[:,0] = 1.  # TODO make work or piecewise constant.
                section_property_map = section_property_map.tocsc()
            else:
                parameter_bspline_curve = BSplineCurve(name=f'order_{parameter.order}_{parameter.property_type}', order_u=order, control_points=np.zeros((parameter_num_dof,)))   # control points are in CSDL, so only using this to generate map
                parameter_bspline_map = parameter_bspline_curve.compute_evaluation_map(np.linspace(0., 1., num_sections))
                section_property_map = parameter_bspline_map

            # add section property map to section properties map to create a single map
            if parameter.value is not None or parameter.connection_name is not None:
                prescribed_parameter_ending_index = prescribed_parameter_starting_index + parameter_num_dof
                prescribed_section_properties_map[(property_index*num_sections):((property_index+1)*num_sections), prescribed_parameter_starting_index:prescribed_parameter_ending_index] = section_property_map

                prescribed_parameter_starting_index = prescribed_parameter_ending_index
            else:
                free_parameter_ending_index = free_parameter_starting_index + parameter_num_dof
                free_affine_section_properties_map[(property_index*num_sections):((property_index+1)*num_sections), free_parameter_starting_index:free_parameter_ending_index] = section_property_map

                free_parameter_starting_index = free_parameter_ending_index
        
        free_affine_section_properties_map = free_affine_section_properties_map.tocsc()
        prescribed_affine_section_properties_map = prescribed_section_properties_map.tocsc()

        self.free_affine_section_properties_map = free_affine_section_properties_map
        self.prescribed_affine_section_properties_map = prescribed_affine_section_properties_map
        return free_affine_section_properties_map, prescribed_affine_section_properties_map


    def assemble_rotational_section_properties_map(self):
        '''
        Assembles the sectional rotations map from the rotational parameters added to the ffd block.
        Sectional rotations map maps rotational parameter dofs --> section rotations
        '''
        if self.rotational_section_properties_map is not None:
            return self.rotational_section_properties_map

        num_sections = self.num_sections

        # count how many FFD inputs there are
        self.num_rotational_dof = 0
        for parameter in list(self.parameters.values()):
            if parameter.property_type == 'rotation_u' or \
                parameter.property_type == 'rotation_v' or \
                parameter.property_type == 'rotation_w':
                parameter_num_dof = parameter.num_dof
                self.num_rotational_dof += parameter_num_dof

        # allocate sparse section properties_map now that we know how many inputs there are
        rotational_section_properties_map = sps.lil_array((num_sections*self.num_rotational_properties, self.num_rotational_dof))

        parameter_starting_index = 0
        for parameter in list(self.parameters.values()):
            if parameter.property_type == 'translation_u' or \
                parameter.property_type == 'translation_v' or \
                parameter.property_type == 'translation_w' or \
                parameter.property_type == 'scale_v' or \
                parameter.property_type == 'scale_w':
                continue
            elif parameter.property_type == 'rotation_u':
                property_index = 0
            elif parameter.property_type == 'rotation_v':
                property_index = 1
            elif parameter.property_type == 'rotation_w':
                property_index = 2
            else:
                raise Exception(f"Error trying to add a parameter ({self.name}:{parameter}). Please pass in a specific FFDParameter object.")
    
            order = parameter.order
            parameter_num_dof = parameter.num_dof
            parameter_ending_index = parameter_starting_index + parameter_num_dof

            # generate section property map
            if parameter.order == 1:    # if constant
                sectional_rotation_map = sps.lil_array((num_sections, parameter_num_dof))
                sectional_rotation_map[:,0] = 1.  # TODO make work or piecewise constant.
                sectional_rotation_map = sectional_rotation_map.tocsc()
            else:
                parameter_bspline_curve = BSplineCurve(name=f'order_{order}_{parameter.property_type}', order_u=order, control_points=np.zeros((parameter_num_dof,)))   # control points are in CSDL, so only using this to generate map
                parameter_bspline_map = parameter_bspline_curve.compute_evaluation_map(np.linspace(0., 1., num_sections))
                sectional_rotation_map = parameter_bspline_map

            # add section property map to section properties map to create a single map            
            rotational_section_properties_map[(property_index*num_sections):((property_index+1)*num_sections), parameter_starting_index:parameter_ending_index] = sectional_rotation_map

            parameter_starting_index = parameter_ending_index
        
        rotational_section_properties_map = rotational_section_properties_map.tocsc()

        self.rotational_section_properties_map = rotational_section_properties_map
        return rotational_section_properties_map


    def assemble_local_frame_properties(self):
        '''
        Assembles the properties and control points in the local frame so that transformations can be applied as intented.
        '''
        control_points_reshaped = self.control_points.reshape((self.primitive.shape))

        # get section origins
        parametric_coordinates = self.primitive.project(control_points_reshaped[:,0,0,:], return_parametric_coordinates=True, plot=False)
        u_sections = parametric_coordinates[:,0]
        # u_sections = np.linspace(0., 1., self.num_sections)
        section_origins_v = np.ones((self.num_sections,))*0.5
        section_origins_w = np.ones((self.num_sections,))*0.5
        self.section_origins = self.primitive.evaluate_points(u_sections.copy(), section_origins_v, section_origins_w)
        # get uvw basis
        # -- identifies w axis then v axis since they should be easier to identify (sections should be parallel/normal to the cartesian axes)
        w_axis_vectors = (control_points_reshaped[:,:,1:,:] - control_points_reshaped[:,:,:-1,:]).reshape((-1,3))
        w_axis_vectors_greatest_change_axis = np.argmax(np.linalg.norm(w_axis_vectors, axis=0))
        v_axis_vectors = control_points_reshaped[:,1:,:,:] - control_points_reshaped[:,:-1,:,:]
        v_axis_vectors_greatest_change_axis = np.argmax(np.linalg.norm(v_axis_vectors, axis=0))
        identity_matrix = np.eye(3)
        uvw_basis = np.zeros((3,3))
        uvw_basis[:,2] = identity_matrix[w_axis_vectors_greatest_change_axis,:]
        uvw_basis[:,1] = identity_matrix[v_axis_vectors_greatest_change_axis,:]
        uvw_basis[:,0] = np.cross(uvw_basis[:,1], uvw_basis[:,2])

        # get control_points_section_frame
        self.local_to_global_translations = np.repeat(self.section_origins, self.num_control_points_per_section, axis=0).reshape((self.primitive.shape))
        # control_points_reshaped_section_frame = (uvw_basis.T.dot((control_points_reshaped - self.local_to_global_translations).T)).T
        self.local_to_global_rotation = uvw_basis
        self.control_points_section_frame = (control_points_reshaped - self.local_to_global_translations).dot(self.local_to_global_rotation)

        # normalize the section scaling
        self.initial_scale_v = np.max(self.control_points_section_frame[0,:,:,1]) - np.min(self.control_points_section_frame[0,:,:,1])
        self.initial_scale_w = np.max(self.control_points_section_frame[0,:,:,2]) - np.min(self.control_points_section_frame[0,:,:,2])
        self.control_points_section_frame[:,:,:,1] /= self.initial_scale_v
        self.control_points_section_frame[:,:,:,2] /= self.initial_scale_w


    def assemble_affine_block_deformations_map(self):
        '''
        Assembles the local control points map.
        The control points map maps affine section properties --> local FFD control points
        '''
        if self.control_points_affine_map is not None:
            return self.control_points_affine_map

        num_section_properties = self.num_sections*(self.num_affine_properties)
        num_points_per_section = np.cumprod(self.primitive.shape[1:-1])[-1]

        initial_control_points = self.control_points_section_frame.reshape((-1,3)).copy()

        # Preallocate ffd block control points maps
        num_control_points =  self.control_points_section_frame.reshape((-1,3)).shape[0]
        control_points_affine_map = np.zeros((num_control_points, 3, num_section_properties))
        control_points_u_map = sps.lil_array((num_control_points, num_section_properties))
        control_points_v_map = sps.lil_array((num_control_points, num_section_properties))
        control_points_w_map = sps.lil_array((num_control_points, num_section_properties))

        # Assemble ffd block control points maps
        for section_number in range(self.num_sections):
            start_index = section_number*num_points_per_section
            end_index = (section_number+1)*num_points_per_section
            control_points_u_map[start_index:end_index, section_number] = 1.  # Translation x
            control_points_v_map[start_index:end_index, self.num_sections+section_number] = 1. # Translation y
            control_points_w_map[start_index:end_index, (self.num_sections*2)+section_number] = 1. # Translation z
            control_points_u_map[start_index:end_index, (self.num_sections*3)+section_number] = initial_control_points[start_index:end_index, 0] # Scale x
            control_points_v_map[start_index:end_index, (self.num_sections*4)+section_number] = initial_control_points[start_index:end_index, 1]*self.initial_scale_v   # Scale y
            control_points_w_map[start_index:end_index, (self.num_sections*5)+section_number] = initial_control_points[start_index:end_index, 2]*self.initial_scale_w   # Scale z


        self.control_points_u_map = control_points_u_map.tocsc()
        self.control_points_v_map = control_points_v_map.tocsc()
        self.control_points_w_map = control_points_w_map.tocsc()

        # Only converting to tensor to make construction easier (less reshaping). If this slows down assembly, then use reshaping instead.
        control_points_affine_map[:,0,:] = np.array(self.control_points_u_map.todense())
        control_points_affine_map[:,1,:] = np.array(self.control_points_v_map.todense())
        control_points_affine_map[:,2,:] = np.array(self.control_points_w_map.todense())
        NUM_PARAMETRIC_DIMENSIONS = 3      # This type of FFD block has 3 parametric dimensions
        control_points_affine_map_matrix = control_points_affine_map.reshape((num_control_points*NUM_PARAMETRIC_DIMENSIONS, num_section_properties))
        affine_block_deformations_map = sps.csc_array(control_points_affine_map_matrix)

        # Start of alternative method with reshaping along different axes.
        # affine_block_deformations_map = sps.vstack((self.control_points_u_map, self.control_points_v_map, self.control_points_w_map))
        self.affine_block_deformations_map = affine_block_deformations_map

        return self.affine_block_deformations_map


    def assemble_embedded_entities_map(self):
        '''
        Assembles the embedded entities map.
        The embedded entities map maps from the ffd block control points --> embedded entity points
        '''

        if self.embedded_entities_map is not None:
            return self.embedded_entities_map

        if self.num_dof == 0:
            return None

        embedded_entities = self.project(self.embedded_points, grid_search_n=5, plot=False)
        self.embedded_entities_map = embedded_entities.linear_map

        return self.embedded_entities_map


    def assemble_cost_matrix(self):
        '''
        Assembles the cost matrix for penalizing each parameter dof in the geometry solver.
        The penalty can also be looked at as a stiffness property and the geometry solver as an energy minimization.
        '''
        # Preallocate cost matrix (alphas in x.T.dot(alphas).dot(x))
        cost_matrix = sps.csc_array(sps.eye(self.num_affine_free_dof))

        parameter_starting_index = 0
        for parameter in list(self.parameters.values()):
            if parameter.property_type == 'rotation_u' or \
                parameter.property_type == 'rotation_v' or \
                parameter.property_type == 'rotation_w' or \
                parameter.value is not None or parameter.connection_name is not None:
                continue

            parameter_num_dof = parameter.num_dof
            parameter_ending_index = parameter_starting_index + parameter_num_dof
            indices = np.ix_(np.arange(parameter_starting_index, parameter_ending_index))

            cost_matrix[indices, indices] = parameter.cost_factor/parameter_num_dof

            parameter_starting_index = parameter_ending_index

        self.cost_matrix = cost_matrix
        return cost_matrix


    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_n:int=50,
                    max_iter:int=100, return_parametric_coordinates:bool=False, plot:bool=False):
        return self.primitive.project(points=points, direction=direction, grid_search_n=grid_search_n,
                    max_iter=max_iter, return_parametric_coordinates=return_parametric_coordinates, plot=plot)


    def plot(self, plot_embedded_entities:bool=True, plot_types:list=['mesh','point_cloud'], opacity:float=0.3, 
            additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the FFD block and optionally the entities embedded within.
        '''
        plotting_elements = additional_plotting_elements.copy()

        if plot_embedded_entities:
            for embedded_entity in list(self.embedded_entities.values()):
                plotting_elements = embedded_entity.plot(plot_types=['mesh'], opacity=1., additional_plotting_elements=plotting_elements, show=False)

        plotting_elements = self.primitive.plot(plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements, show=False)
        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, f'Free Form Deformation Block: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements

    def plot_sections(self, control_points:np.ndarray=None, offset_sections:bool=False, plot_embedded_entities:bool=True, \
                            plot_types:list=['mesh','point_cloud'], opacity:float=0.3, additional_plotting_elements:list=[], show:bool=True):
        
        plotting_elements = additional_plotting_elements.copy()

        if control_points is None:
            control_points = self.primitive.control_points

        if plot_embedded_entities:
            for embedded_entity in list(self.embedded_entities.values()):
                plotting_elements = embedded_entity.plot(plot_types=['mesh'], opacity=1., additional_plotting_elements=plotting_elements, show=False)

        surface_shape = self.primitive.shape[1:]
        for i in range(self.num_sections):
            if len(control_points.shape) == 2:
                points = control_points[i*self.num_control_points_per_section:(i+1)*self.num_control_points_per_section].copy()
                if offset_sections:
                    points[:,0] += i    # separating surfaces for visualization
                points = points.reshape(surface_shape)
            else:
                points = control_points[i,:,:,:].copy()
                if offset_sections:
                    points[:,:,0] += i    # separating surfaces for visualization
            
            plotting_elements = self.primitive.plot_surface(points, plot_types=plot_types, opacity=opacity, additional_plotting_elements=plotting_elements, show=False)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, f'FFD Block Sections: {self.name}', axes=1, viewup="z", interactive=True)
            return plotting_elements
        else:
            return plotting_elements
        
    def plot_parameter_curves(self, control_points, evaluated_points, show:bool=True):
        plotting_elements = []

        # Plot evaluated poitns
        fig = vedo.pyplot.plot(
            np.linspace(0,1.,self.num_sections), evaluated_points[:self.num_sections],
            "*b",           # markers: *,o,p,h,D,d,v,^,s,x,a
            title='Rotational Section Properties',
            xtitle="u",
            ytitle="Sectional Rotation",
            axes=dict(text_scale=0.8),
            label="Rotational Around u-axis",
        )
        # Let Appropriate evaluation handle this
        # fig += vedo.pyplot.plot(
        #     np.linspace(0,1.,self.num_sections), evaluated_points[self.num_sections:2*self.num_sections],
        #     "pg",           # markers: *,o,p,h,D,d,v,^,s,x,a
        #     like=fig,
        #     label="Rotational Around v-axis",
        # )
        # fig += vedo.pyplot.plot(
        #     np.linspace(0,1.,self.num_sections), evaluated_points[2*self.num_sections:],
        #     "^r",           # markers: *,o,p,h,D,d,v,^,s,x,a
        #     like=fig,
        #     label="Rotational Around w-axis",
        # )
        # fig.add_legend()

        # Add control points?
        # fig += vedo.pyplot.plot(
        #     np.linspace(0,1.,control_points.shape[0]), control_points[control_points.shape[0]:2*self.num_sections],
        #     "ob",           # markers: *,o,p,h,D,d,v,^,s,x,a
        #     like=fig,
        # )
        # fig += vedo.pyplot.plot(
        #     np.linspace(0,1.,control_points.shape[0]), control_points[2*self.num_sections:],
        #     "og",           # markers: *,o,p,h,D,d,v,^,s,x,a
        #     like=fig,
        # )
        # fig += vedo.pyplot.plot(
        #     np.linspace(0,1.,control_points.shape[0]), control_points[self.num_sections:2*self.num_sections],
        #     "or",           # markers: *,o,p,h,D,d,v,^,s,x,a
        #     like=fig,
        # )

        plotting_elements.append(fig)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, interactive=True)

        return plotting_elements

    def plot_pertubations(self):
        '''
        TODO: Decide exactly what this should show and automate the plotting.
        '''
        raise Exception("plot_pertubations is not implemented yet.")

    
    def assemble(self, project_embedded_entities=True):
        '''
        Assembles the maps for the free form deformation.
        '''
        self.assemble_affine_section_properties_maps()
        self.assemble_rotational_section_properties_map()
        self.assemble_affine_block_deformations_map()
        if project_embedded_entities:
            self.assemble_embedded_entities_map()
        self.assemble_cost_matrix()

        # self.assemble_evaluation_map()

    def setup_default_states(self):
        '''
        Generates default values for each of the FFD block states:
        -- free affine dof
        -- prescribed affine dof
        -- rotational dof
        -- affine section properties
        -- rotational section properties
        -- affine transformed control points local frame
        -- rotated control points local frame
        -- control points
        -- embedded points
        '''
        # default free affine dof
        self.free_affine_dof = np.zeros((self.num_affine_free_dof,))

        # default prescribed affine dof
        prescribed_affine_dof = np.zeros((self.num_affine_prescribed_dof,))
        parameter_starting_index = 0
        for parameter in list(self.parameters.values()):
            if parameter.property_type == 'rotation_u' or \
                    parameter.property_type == 'rotation_v' or \
                    parameter.property_type == 'rotation_w':    # If it's rotational then it's not affine
                continue
            if parameter.value is not None:
                dof = parameter.value
            else:
                if parameter.connection_name is not None:
                    dof = np.zeros((parameter.num_dof,))
                else:    # no connection name and no value means it's not prescribed (it's free)
                    continue
            
            parameter_ending_index = parameter_starting_index + parameter.num_dof
            prescribed_affine_dof[parameter_starting_index:parameter_ending_index] = dof
            parameter_starting_index = parameter_ending_index
        self.prescribed_affine_dof = prescribed_affine_dof

        # default rotational dof
        rotational_dof = np.zeros((self.num_rotational_dof,))
        parameter_starting_index = 0
        for parameter in list(self.parameters.values()):
            if parameter.property_type == 'translation_u' or \
                    parameter.property_type == 'translation_v' or \
                    parameter.property_type == 'translation_w' or \
                    parameter.property_type == 'scale_v' or \
                    parameter.property_type == 'scale_w':
                continue

            if parameter.value is not None:
                dof = parameter.value
            else:
                dof = np.zeros((parameter.num_dof,))
            
            parameter_ending_index = parameter_starting_index + parameter.num_dof
            rotational_dof[parameter_starting_index:parameter_ending_index] = dof
            parameter_starting_index = parameter_ending_index
        self.rotational_dof = rotational_dof

        # default affine section properties
        affine_section_properties = np.zeros((self.num_affine_section_properties,))
 
        # Add 1 to scaling parameters to make initial scaling=1.
        ffd_block_scaling_properties_starting_index = self.num_sections*(self.num_affine_properties-self.num_scaling_properties)
        affine_section_properties[ffd_block_scaling_properties_starting_index:] = 1.
        self.affine_section_properties = affine_section_properties
        
        self.translations = np.zeros((self.num_sections*(self.num_affine_properties-self.num_scaling_properties),))   # number of translations

        # default rotational section properties
        self.rotational_section_properties = np.zeros((self.num_rotational_section_properties))

        # default affine tranformed control points in local frame (each section in its section frame)
        self.affine_deformed_control_points = self.control_points_section_frame.reshape((-1,3)) # 3 parametric dimensions

        # default rotated control points local frame (each section in its section frame)
        self.rotated_control_points_local_frame = self.control_points_section_frame.reshape((-1,3)) # 3 parametric dimensions

        # default control points is done in constructor

        # default embedded points
        embedded_points = None
        for entity in list(self.embedded_entities.values()):
            # TODO generalize to primitives in general!!
            if type(entity) is BSplineCurve or \
                    type(entity) is BSplineSurface or \
                    type(entity) is BSplineVolume:
                points = entity.control_points
            elif type(entity) is np.ndarray:
                points = entity
            elif type(entity) is am.MappedArray:
                points = entity.value

            num_points = np.cumprod(entity.shape[:-1])[-1]
            num_dimensions = entity.shape[-1]
            points = points.reshape((num_points,num_dimensions))

            if embedded_points is None:
                embedded_points = points
            else:
                embedded_points = np.vstack((embedded_points, points))
        self.embedded_points = embedded_points


    def setup(self, project_embedded_entities=True):
        '''
        Sets up the FFD block for evaluation by assembling maps and setting up default states.

        This step precomputes everything that can be precomputed to avoid unnecessary recomputation.
        '''
        self.assemble_local_frame_properties()
        self.setup_default_states()
        self.assemble(project_embedded_entities=project_embedded_entities)
    

    def evaluate_affine_section_properties(self, free_affine_dof=None, prescribed_affine_dof=None, plot=False):
        '''
        Evaluates the section properties from input of affine dof (translations and scalings)
        '''
        if free_affine_dof is None:
            free_affine_dof = self.free_affine_dof
        else:
            self.free_affine_dof = free_affine_dof
        if prescribed_affine_dof is None:
            prescribed_affine_dof = self.prescribed_affine_dof
        else:
            self.prescribed_affine_dof = prescribed_affine_dof

        affine_section_properties_free_component = self.free_affine_section_properties_map.dot(free_affine_dof)
        affine_section_properties_prescribed_component = self.prescribed_affine_section_properties_map.dot(prescribed_affine_dof)

        affine_section_properties = np.zeros((self.num_affine_section_properties,))

        ffd_block_scaling_properties_starting_index = self.num_sections*(self.num_affine_properties-self.num_scaling_properties)

        # Use calculated values for non-scaling parameters
        affine_section_properties[:ffd_block_scaling_properties_starting_index] = \
            affine_section_properties_free_component[:ffd_block_scaling_properties_starting_index] \
            + affine_section_properties_prescribed_component[:ffd_block_scaling_properties_starting_index]
        
        # Add 1 to scaling parameters to make initial scaling=1.
        affine_section_properties[ffd_block_scaling_properties_starting_index:] = \
            affine_section_properties_free_component[ffd_block_scaling_properties_starting_index:] \
            + affine_section_properties_prescribed_component[ffd_block_scaling_properties_starting_index:] \
            + 1.  # adding 1 which is initial scale value

        self.affine_section_properties = affine_section_properties
        self.translations = affine_section_properties[:3*self.num_sections]     # 3 because 3 translational properties

        return affine_section_properties
    

    def evaluate_rotational_section_properties(self, rotational_dof=None):
        '''
        Evaluates the section rotations from input of rotational dof
        '''
        if rotational_dof is None:
            rotational_dof = self.rotational_dof
        else:
            self.rotational_dof = rotational_dof        

        rotational_section_properties = self.rotational_section_properties_map.dot(rotational_dof)
        self.rotational_section_properties = rotational_section_properties

        plot=False
        if plot:
            self.plot_parameter_curve(rotational_dof, rotational_section_properties)

        return rotational_section_properties


    def evaluate_affine_block_deformations(self, affine_section_properties=None, plot=False):
        '''
        Evaluates the local control points of the FFD block given the affine section properties and section rotations
        '''
        if affine_section_properties is None:
            affine_section_properties = self.affine_section_properties

        affine_control_points_local_frame_flattened = self.affine_block_deformations_map.dot(affine_section_properties)
        NUM_PARAMETRIC_DIMENSIONS = 3       # This type of FFD block has 3 parametric dimensions by definition.
        affine_deformed_control_points = affine_control_points_local_frame_flattened.reshape(
                                                        (self.num_control_points, NUM_PARAMETRIC_DIMENSIONS))
        self.affine_deformed_control_points = affine_deformed_control_points

        if plot:
            self.plot_sections(control_points=affine_deformed_control_points, offset_sections=True, 
                                plot_embedded_entities=False, opacity=0.75, show=True)

        return affine_deformed_control_points


    def evaluate_rotational_block_deformations(self, affine_deformed_control_points=None, translations=None, \
                                                                            rotational_section_properties=None, plot=False):
        '''
        Evaluates the control points of the FFD block in original coordinate frame by applying the rotational section properties (section rotations).
        '''

        # Processing inputs
        if affine_deformed_control_points is None:
            affine_deformed_control_points = self.affine_deformed_control_points
        else:
            self.affine_deformed_control_points = affine_deformed_control_points
        if translations is None:
            translations_flattened = self.translations
            NUM_PARAMETRIC_DIMENSIONS = 3       # Really, num_translational_properties, but num_translational_props = num_parametric_dims
            translations = translations_flattened.reshape((self.num_sections, NUM_PARAMETRIC_DIMENSIONS))
        else:
            self.translations = translations.reshape((-1,), order='F')
        if rotational_section_properties is None:
            rotational_section_properties = self.rotational_section_properties
        else:
            self.rotational_section_properties = rotational_section_properties

        if len(rotational_section_properties) == 0:
            self.rotated_control_points_local_frame = affine_deformed_control_points
            if plot:
                self.plot_sections(control_points=rotated_control_points_local_frame, offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)
            return affine_deformed_control_points

        # Undo translations so section origin is at origin
        affine_control_points_local_frame_reshaped = affine_deformed_control_points.reshape(self.primitive.shape)
        affine_control_points_local_frame_reshaped_axis_moved = np.moveaxis(affine_control_points_local_frame_reshaped, 0, 2)   # so subtraction is properly casted
        affine_control_points_local_frame_without_translations_axis_moved = affine_control_points_local_frame_reshaped_axis_moved - translations
        affine_control_points_local_frame_without_translations = np.moveaxis(affine_control_points_local_frame_without_translations_axis_moved, 2, 0)

        # Calculate rotation matrices (rotation_matrices_u, rotation_matrices_v, rotation_matrices_w), each (num_section,3,3)
        rotation_u = rotational_section_properties[:self.num_sections]
        rotation_v = rotational_section_properties[self.num_sections:2*self.num_sections]
        rotation_w = rotational_section_properties[2*self.num_sections:3*self.num_sections]
        sin_rotations_u = np.sin(rotation_u)
        cos_rotations_u = np.cos(rotation_u)
        sin_rotations_v = np.sin(rotation_v)
        cos_rotations_v = np.cos(rotation_v)
        sin_rotations_w = np.sin(rotation_w)
        cos_rotations_w = np.cos(rotation_w)

        # Rotation tensors are numpy arrays because they are very dense and the method used has fastest runtime.
        # -- The final matmtul we are doing is ijkl, ijlm --> ijkm
        # ---- i corresponds to sections, so we want element-wise operation
        # ---- j corresponds to points per section, so this is a repeated operation (like an extra column in dot)
        # ---- l is summed over corresponding to the matmul when normally applying matrix multiplication
        # -- For efficiency, we leave j axis out of map because it's the same map. We also loop over sections instead of tensordot for speed.
        # -- In the future, could explore parallelizing the for loop (in python) (not CSDL)
        # -- Also could consider using opt_einsum (usable through np.einsum(..., optimize=True))
        NUM_PARAMETRIC_DIMENSIONS = 3
        rotation_tensor_u = np.tile(np.eye(NUM_PARAMETRIC_DIMENSIONS), (self.num_sections,1,1))
        rotation_tensor_v = rotation_tensor_u.copy()
        rotation_tensor_w = rotation_tensor_u.copy()
        rotation_tensor = rotation_tensor_u.copy()

        rotation_tensor_u[:,1,1] = cos_rotations_u
        rotation_tensor_u[:,1,2] = sin_rotations_u
        rotation_tensor_u[:,2,1] = -sin_rotations_u
        rotation_tensor_u[:,2,2] = cos_rotations_u

        rotation_tensor_v[:,0,0] = cos_rotations_v
        rotation_tensor_v[:,0,2] = -sin_rotations_v
        rotation_tensor_v[:,2,0] = sin_rotations_v
        rotation_tensor_v[:,2,2] = cos_rotations_v

        rotation_tensor_w[:,0,0] = cos_rotations_w
        rotation_tensor_w[:,0,1] = sin_rotations_w
        rotation_tensor_w[:,1,0] = -sin_rotations_w
        rotation_tensor_w[:,1,1] = cos_rotations_w

        # rotation_tensor = np.tensordot(rotation_tensor_u, np.tensordot(rotation_tensor_v, rotation_tensor_w, axes=([3],[2])), axes=([3],[2]))
        # rotation_matrix = rotation_matrix_u.dot(rotation_matrix_v).dot(rotation_matrix_w)
        rotated_control_points_local_frame_without_translations = np.zeros_like(affine_control_points_local_frame_without_translations)
        for i in range(self.num_sections):
            # Combine x,y,z rotation maps
            rotation_tensor[i,:,:] = rotation_tensor_u[i].dot(rotation_tensor_v[i]).dot(rotation_tensor_w[i])
            # Apply rotation to section
            rotated_points = np.tensordot(rotation_tensor[i,:,:], affine_control_points_local_frame_without_translations[i,:,:,:], axes=([-1],[-1]))
            rotated_control_points_local_frame_without_translations[i,:,:,:] = np.moveaxis(rotated_points, 0, -1)


        # # Apply rotation matrices to each section
        # rotated_control_points_local_frame_without_translations = rotation_tensor.dot(affine_control_points_local_frame_without_translations)

        # Add back on translations from the affine transformation
        rotated_control_points_local_frame_without_translations = np.moveaxis(rotated_control_points_local_frame_without_translations, 0, 2)
        rotated_control_points_local_frame_axis_moved = rotated_control_points_local_frame_without_translations + translations
        rotated_control_points_local_frame_reshaped = np.moveaxis(rotated_control_points_local_frame_axis_moved, 2, 0)
        rotated_control_points_local_frame = rotated_control_points_local_frame_reshaped.reshape((self.num_control_points,NUM_PARAMETRIC_DIMENSIONS))
        self.rotated_control_points_local_frame = rotated_control_points_local_frame

        if plot:
            self.plot_sections(control_points=rotated_control_points_local_frame, offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)

        return rotated_control_points_local_frame

        # Next model will perform rotation back to global frame

    def evaluate_control_points(self, rotated_control_points_local_frame:np.ndarray=None, plot:bool=False):
        '''
        Evaluates the control points of the FFD block in original coordinate frame by applying 
        bulk rotation and translation back to original coordinate frame.
        '''
        if rotated_control_points_local_frame is None:
            rotated_control_points_local_frame = self.rotated_control_points_local_frame
        else:
            self.rotated_control_points_local_frame = rotated_control_points_local_frame
        
        NUM_PARAMETRIC_DIMENSIONS = 3
        control_points_rotated_back_wrong_axis = np.tensordot(self.local_to_global_rotation, rotated_control_points_local_frame, axes=([-1],[-1]))
        control_points_rotated_back = np.moveaxis(control_points_rotated_back_wrong_axis, 0, 1)
        control_points_rotated_back_reshaped = control_points_rotated_back.reshape(self.primitive.shape)

        control_points_reshaped = control_points_rotated_back_reshaped + self.local_to_global_translations
        control_points = control_points_reshaped.reshape((self.num_control_points, NUM_PARAMETRIC_DIMENSIONS))
        self.control_points = control_points

        if plot:
            self.plot_sections(control_points=control_points, offset_sections=False, plot_embedded_entities=False, opacity=0.75, show=True)

        return control_points


    def evaluate_embedded_entities(self, control_points=None, plot=False):
        '''
        Evaluates the entities embedded within the FFD block from an input of the FFD control points.
        '''
        if control_points is None:
            control_points = self.control_points

        embedded_entities = self.embedded_entities_map.dot(control_points)

        embedded_entity_starting_index = 0
        for embedded_entity in list(self.embedded_entities.values()):
            embedded_entity_ending_index = embedded_entity_starting_index + np.cumprod(embedded_entity.shape[:-1])[-1]
            embedded_entity.control_points = embedded_entities[embedded_entity_starting_index:embedded_entity_ending_index]
            embedded_entity_starting_index = embedded_entity_ending_index

        if plot:
            plotting_elements = []
            embedded_entity_starting_index = 0
            for embedded_entity in list(self.embedded_entities.values()):
                plotting_elements = embedded_entity.plot(plot_types=['mesh'], opacity=0.5, additional_plotting_elements=plotting_elements, show=False)
            plotting_elements.append(vedo.Points(embedded_entities, r=5).color('green'))
            self.plot_sections(control_points=control_points, offset_sections=False, plot_embedded_entities=False, opacity=0.3, 
                    additional_plotting_elements=plotting_elements, show=True)

        return embedded_entities


    def evaluate(self, ffd_dof):
        '''
        Evaluates the enteties embedded within the FFD block from an input of FFD degrees of freedom.
        '''
        # The intention of this method is to evaluate in "one step" where all the maps are multiplied ahead of time for fast evaluation.
        raise Exception("Not implemented yet. Please use individual evaluation methods.")
    


class TranslationalSRBGFFDBlock(SRBGFFDBlock):
    '''
    Creates a translating Sectioned Rectangular B-spline Geometric FFD block
    for geometric manipulation by manipulating the "sections" of the rectangular B-spline FFD block.

    This block is embedded with parameters to strictly allow bulk translation.
    '''
    def __init__(self, name: str, primitive=None, embedded_entities: dict = {}) -> None:
        super().__init__(name, primitive, embedded_entities, parameters={})
        self.add_translation_u(name='bulk_translation_u', order=1, num_dof=1)
        self.add_translation_v(name='bulk_translation_v', order=1, num_dof=1)
        self.add_translation_w(name='bulk_translation_w', order=1, num_dof=1)


