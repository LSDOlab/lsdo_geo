from dataclasses import dataclass
from src.cython.get_open_uniform_py import get_open_uniform
import numpy as np
import scipy.sparse as sps

import matplotlib.pyplot as plt
from vedo import Points, Plotter, LegendBox

from src.caddee.concept.bsplines.bspline_curve import BSplineCurve
from src.caddee.concept.bsplines.bspline_surface import BSplineSurface
from src.caddee.concept.bsplines.bspline_volume import BSplineVolume

from src.caddee.concept.geometry.geocore.utils.calculate_rotation_mat import calculate_rotation_mat
# import os
# os.chdir("../lsdo_geo/lsdo_kit/design/design_geometry/core")

xglobal = np.array([1,0,0])
yglobal = np.array([0,1,0])
zglobal = np.array([0,0,1])

'''
TODO:

'''

class FFDBlock(object):
    def __init__(self, name: str, control_points, embedded_entities_pointers = [], local_axes=dict(xprime=None, yprime=None, zprime=None), uvw_axis_indices=(0, 1, 2),
                    section_origins=None, local_control_points=None, shape=None, order_u=10, order_v=10, order_w=10,
                    knots_u=None, knots_v=None, knots_w=None):
        
        self.name = name

        self.nxp = control_points.shape[0]
        self.nyp = control_points.shape[1]
        self.nzp = control_points.shape[2]
        
        self.embedded_entities_pointers = embedded_entities_pointers.copy()
        self.embedded_points = [] 
        self.embedded_entities_indices = []

        self.properties_list = []
        self.properties_dict = {}

        # self.property_names = ['rot_x', 'rot_y', 'rot_z', 'trans_x', ' trans_y', 'trans_z', 'scale_y', 'scale_z', 'shape']
        # NOTE: I don't think we'll ever need shape since it's the same as scale with lots of dofs.
        self.property_names = ['rotation_x', 'rotation_y', 'rotation_z', 'translation_x', ' translation_y', 'translation_z', 'scale_x', 'scale_y', 'scale_z']
        self.affine_property_names = self.property_names[3:]
        self.rotational_property_names = self.property_names[:3]
        self.num_properties = len(self.property_names)
        self.num_rotational_properties = len(self.rotational_property_names)
        self.num_affine_properties = len(self.affine_property_names)
        self.parameters = []

        self.num_affine_dof = 0
        self.num_affine_free_dof = 0
        self.num_affine_prescribed_dof = 0
        self.num_rotational_dof = 0
        self.num_dof = 0
        self.num_affine_free_ffd_control_points = 0

        self.num_ffd_control_points = control_points.reshape((-1,3)).shape[0]
        
        if shape is None:
            shape = control_points.shape
        else:
            control_points = control_points.reshape(shape)

        if control_points.shape[0] <= order_u:
            order_u = control_points.shape[0]
        if control_points.shape[1] <= order_v:
            order_v = control_points.shape[1]
        if control_points.shape[2] <= order_w:
            order_w = control_points.shape[2]

        if knots_u is None:
            knots_u = np.zeros(shape[0] + order_u)
            get_open_uniform(order_u, shape[0], knots_u)
        
        if knots_v is None:
            knots_v = np.zeros(shape[1] + order_v)
            get_open_uniform(order_v, shape[1], knots_v)

        if knots_w is None:
            knots_w = np.zeros(shape[2] + order_w)
            get_open_uniform(order_w, shape[2], knots_w)

        # if embedded_entities_pointers == []:
        #     pass
        # else:
        #     self._generate_embedded_coordinates()

        self.b_spline_volume = BSplineVolume(name, order_u, order_v, order_w, knots_u, knots_v, knots_w, shape, control_points)
        self.control_points = self.b_spline_volume.control_points

        self.generate_section_origins(section_origins)
        self._generate_local_coordinate_directions(local_axes)
        self._generate_local_control_points_coordinates()
        self.num_sections = self.nxp

        self.free_section_properties_map = None
        self.prescribed_section_properties_map = None
        self.sectional_rotations_map = None
        self.ffd_control_points_map = None
        self.evaluation_map = None

    def add_ffd_parameter(self, ffd_parameter):
        # if property_name not in self.property_names:
        #     print('WARNING: ', self.name, '_', property_name, ' is not a valid property.')
        #     print('Valid properties: ', self.property_names)
        #     print('FFD parameter', self.name, '_', property_name, ' has not been added.')
        #     return

        if ffd_parameter.num_dof <= ffd_parameter.degree:
            print('WARNING: ', self.name, '_', property_name, " has num <= degree.")
            print('num (number of dof) must be greater than the degree of the curve.')
            print('FFD parameter', self.name, '_', property_name, ' has not been added.')
            return

        if type(ffd_parameter) is FFDRotationXParameter or \
            type(ffd_parameter) is FFDRotationYParameter or \
            type(ffd_parameter) is FFDRotationZParameter:
            self.num_rotational_dof += ffd_parameter.num_dof
            self.num_dof += ffd_parameter.num_dof
        elif type(ffd_parameter) is FFDTranslationXParameter or \
            type(ffd_parameter) is FFDTranslationYParameter or \
            type(ffd_parameter) is FFDTranslationZParameter or \
            type(ffd_parameter) is FFDScaleYParameter or \
            type(ffd_parameter) is FFDScaleZParameter:
            
            self.num_affine_dof += ffd_parameter.num_dof
            self.num_dof += ffd_parameter.num_dof

            if ffd_parameter.value is not None or ffd_parameter.connection_name is not None:
                self.num_affine_prescribed_dof += ffd_parameter.num_dof
            else:
                self.num_affine_free_dof += ffd_parameter.num_dof
                if self.num_affine_free_ffd_control_points == 0:
                    self.num_affine_free_ffd_control_points = self.num_ffd_control_points
                
        else:
            raise Exception("When adding ffd parameter, please pass in specific FFDParameter object.")

        self.parameters.append(ffd_parameter)


    def _generate_midlines(self):
        u_vec = np.linspace(0, 1, self.nxp)

        v_vec_left = np.zeros(self.nxp)
        v_vec_mid  = np.ones(self.nxp) * 0.5
        v_vec_right = np.ones(self.nxp)
        
        w_vec_bot = np.zeros(self.nxp)
        w_vec_mid = np.ones(self.nxp) * 0.5
        w_vec_top = np.ones(self.nxp)

        left_mid_pt = self.b_spline_volume.evaluate_points(u_vec, v_vec_left, w_vec_mid)
        right_mid_pt = self.b_spline_volume.evaluate_points(u_vec, v_vec_right, w_vec_mid)
        bot_mid_pt = self.b_spline_volume.evaluate_points(u_vec, v_vec_mid, w_vec_bot)
        top_mid_pt = self.b_spline_volume.evaluate_points(u_vec, v_vec_mid, w_vec_top)

        self.top_bot_vec = top_mid_pt - bot_mid_pt
        self.right_left_vec = right_mid_pt - left_mid_pt

        #TODO: Discuss the direction of the midlines, which determines where the normal points 
    def generate_section_origins(self, origin):
        num_sections = self.nxp
        self.block_origin = self.b_spline_volume.evaluate_points(np.array([0.5]), np.array([0.5]), np.array([0.5]))

        if origin == None:
            u_vec = np.linspace(0, 1., num_sections)
            v_vec = np.ones(num_sections) * 0.5
            w_vec = np.ones(num_sections) * 0.5
        
            self.section_origins = self.b_spline_volume.evaluate_points(u_vec, v_vec, w_vec)
        elif origin.shape == (num_sections, 3):
            self.section_origins = origin
        else:
            raise Exception('Origin shape must be equal to (nxp, 3)')


    def _generate_local_control_points_coordinates(self):
        self.local_control_points = np.zeros((self.nxp, self.nyp, self.nzp, 3))
        # print(self.section_origins)
        # print(self.control_points)
        # for i in range(self.nxp):

        #     self.local_control_points[i, :, :, :] = self.control_points[i, :, :, :] - self.section_origins[i, :]
        #     print(self.control_points[i, :, :, :] )

        #     print(self.local_control_points[i,:,:,:])
        #     print(self.section_origins[i,:])

        #     global_2_loc_rotmat_x = calculate_rotation_mat(xglobal, self.xprime[i, :])
        #     # global_2_loc_rotmat_y = calculate_rotation_mat(yglobal, self.yprime[i, :])
        #     global_2_loc_rotmat_z = calculate_rotation_mat(zglobal, self.zprime[i, :])

        #     # rot_mat = global_2_loc_rotmat_x.dot(global_2_loc_rotmat_y).dot(global_2_loc_rotmat_z)
        #     rot_mat = global_2_loc_rotmat_x.dot(global_2_loc_rotmat_z)

        #     self.local_control_points[i,:,:,:] = np.matmul(self.local_control_points[i,:,:,:], rot_mat)

        #     print(self.local_control_points[i,:,:,:])

        #     self.initial_scale_y = np.max(self.local_control_points[i,:,:,1]) - np.min(self.local_control_points[i,:,:,1])
        #     self.initial_scale_z = np.max(self.local_control_points[i,:,:,2]) - np.min(self.local_control_points[i,:,:,2])

        #     # Normalizing all of the y-coordinates
        #     self.local_control_points[i,:,:,1] = self.local_control_points[i,:,:,1] / self.initial_scale_y

        #     # Normalizing all of the z-coordinates
        #     self.local_control_points[i,:,:,2] = self.local_control_points[i,:,:,2] / self.initial_scale_z
        expanded_origins = np.repeat(self.section_origins, self.nyp*self.nzp, axis=0).reshape((self.nxp, self.nyp, self.nzp, 3))
        origin_shifted_local_control_points = self.control_points - expanded_origins
        # origin_shifted_local_control_points = self.control_points - self.block_origin
        # global_2_loc_rotmat_x = calculate_rotation_mat(xglobal, self.xprime[0, :])
        # # global_2_loc_rotmat_z = calculate_rotation_mat(zglobal, self.zprime[0, :])

        # # rot_mat = global_2_loc_rotmat_x.dot(global_2_loc_rotmat_z)
        # rot_mat = global_2_loc_rotmat_x
        rot_mat = np.zeros((3,3))
        rot_mat[0,:] = self.xprime[0,:]
        rot_mat[1,:] = self.yprime[0,:]
        rot_mat[2,:] = self.zprime[0,:]
        # rot_mat = rot_mat.T
        self.rotation_matrix = rot_mat
        self.local_control_points = np.matmul(origin_shifted_local_control_points, rot_mat)
        # self.section_origins = np.matmul(self.section_origins, rot_mat)

        self.initial_scale_y = np.max(self.local_control_points[0,:,:,1]) - np.min(self.local_control_points[0,:,:,1])
        self.initial_scale_z = np.max(self.local_control_points[0,:,:,2]) - np.min(self.local_control_points[0,:,:,2])

        # Normalizing all of the y-coordinates
        self.local_control_points[:,:,:,1] = self.local_control_points[:,:,:,1] / self.initial_scale_y

        # Normalizing all of the z-coordinates
        self.local_control_points[:,:,:,2] = self.local_control_points[:,:,:,2] / self.initial_scale_z


    def generate_section_properties_maps(self):
        if self.free_section_properties_map is not None:
            return self.free_section_properties_map, self.prescribed_section_properties_map

        num_sections = self.nxp

        # allocate sparse section properties_map now that we know how many inputs there are
        free_section_properties_map = sps.lil_array((num_sections*self.num_affine_properties, self.num_affine_free_dof))
        prescribed_section_properties_map = sps.lil_array((num_sections*self.num_affine_properties, self.num_affine_prescribed_dof))

        free_parameter_starting_index = 0
        prescribed_parameter_starting_index = 0
        for ffd_parameter in self.parameters:
            if type(ffd_parameter) is FFDTranslationXParameter:
                property_name = 'translation_x'
                property_index = 0
            elif type(ffd_parameter) is FFDTranslationYParameter:
                property_name = 'translation_y'
                property_index = 1
            elif type(ffd_parameter) is FFDTranslationZParameter:
                property_name = 'translation_z'
                property_index = 2
            # elif type(ffd_parameter) is FFDScaleXParameter:   NOTE: if scale x is added as a parameter, this gets uncommented.
            #     property_name = 'scale_x'
            #     property_index = 3
            elif type(ffd_parameter) is FFDScaleYParameter:
                property_name = 'scale_y'
                property_index = 4
            elif type(ffd_parameter) is FFDScaleZParameter:
                property_name = 'scale_z'
                property_index = 5
            elif type(ffd_parameter) is FFDRotationXParameter or \
                    type(ffd_parameter) is FFDRotationYParameter or \
                    type(ffd_parameter) is FFDRotationZParameter:
                continue
            else:
                raise Exception(f"Error trying to add a parameter ({self.name}:{ffd_parameter}). Please pass in a specific FFDParameter object.")
    
            order = ffd_parameter.degree + 1
            parameter_num_dof = ffd_parameter.num_dof

            # generate section property map
            if ffd_parameter.degree == 0:    # if constant
                section_property_map = sps.lil_array((num_sections, parameter_num_dof))
                section_property_map[:,0] = 1.  # TODO make work or piecewise constant.
                section_property_map = section_property_map.tocsc()
            else:
                parameter_bspline_curve = BSplineCurve(name=f'degree_{ffd_parameter.degree}_{property_name}', order_u=order, control_points=np.zeros((parameter_num_dof,)))   # control points are in CSDL, so only using this to generate map
                parameter_bspline_map = parameter_bspline_curve.compute_eval_map_points(np.linspace(0., 1., num_sections))
                section_property_map = parameter_bspline_map

            # add section property map to section properties map to create a single map
            if ffd_parameter.value is not None or ffd_parameter.connection_name is not None:
                prescribed_parameter_ending_index = prescribed_parameter_starting_index + parameter_num_dof
                prescribed_section_properties_map[(property_index*num_sections):((property_index+1)*num_sections), prescribed_parameter_starting_index:prescribed_parameter_ending_index] = section_property_map

                prescribed_parameter_starting_index = prescribed_parameter_ending_index
            else:
                free_parameter_ending_index = free_parameter_starting_index + parameter_num_dof
                free_section_properties_map[(property_index*num_sections):((property_index+1)*num_sections), free_parameter_starting_index:free_parameter_ending_index] = section_property_map

                free_parameter_starting_index = free_parameter_ending_index
        
        free_section_properties_map = free_section_properties_map.tocsc()
        prescribed_section_properties_map = prescribed_section_properties_map.tocsc()

        self.free_section_properties_map = free_section_properties_map
        self.prescribed_section_properties_map = prescribed_section_properties_map
        return free_section_properties_map, prescribed_section_properties_map


    def generate_sectional_rotations_map(self):
        if self.sectional_rotations_map is not None:
            return self.sectional_rotations_map

        num_sections = self.nxp

        # count how many FFD inputs there are
        self.num_rotational_dof = 0
        for ffd_parameter in self.parameters:
            if type(ffd_parameter) is FFDRotationXParameter or \
                type(ffd_parameter) is FFDRotationYParameter or \
                type(ffd_parameter) is FFDRotationZParameter:
                parameter_num_dof = ffd_parameter.num_dof
                self.num_rotational_dof += parameter_num_dof

        # allocate sparse section properties_map now that we know how many inputs there are
        sectional_rotations_map = sps.lil_array((num_sections*self.num_rotational_properties, self.num_rotational_dof))

        parameter_starting_index = 0
        for ffd_parameter in self.parameters:
            if type(ffd_parameter) is FFDTranslationXParameter or \
                type(ffd_parameter) is FFDTranslationYParameter or \
                type(ffd_parameter) is FFDTranslationZParameter or \
                type(ffd_parameter) is FFDScaleYParameter or \
                type(ffd_parameter) is FFDScaleZParameter:
                continue
            elif type(ffd_parameter) is FFDRotationXParameter:
                property_name = 'rotation_x'
                property_index = 0
            elif type(ffd_parameter) is FFDRotationYParameter:
                property_name = 'rotation_y'
                property_index = 1
            elif type(ffd_parameter) is FFDRotationZParameter:
                property_name = 'rotation_z'
                property_index = 2
            else:
                raise Exception(f"Error trying to add a parameter ({self.name}:{ffd_parameter}). Please pass in a specific FFDParameter object.")
    
            order = ffd_parameter.degree + 1
            parameter_num_dof = ffd_parameter.num_dof
            parameter_ending_index = parameter_starting_index + parameter_num_dof

            # generate section property map
            if ffd_parameter.degree == 0:    # if constant
                sectional_rotation_map = sps.lil_array((num_sections, parameter_num_dof))
                sectional_rotation_map[:,0] = 1.  # TODO make work or piecewise constant.
                sectional_rotation_map = sectional_rotation_map.tocsc()
            else:
                parameter_bspline_curve = BSplineCurve(name=f'degree_{ffd_parameter.degree}_{property_name}', order_u=order, control_points=np.zeros((parameter_num_dof,)))   # control points are in CSDL, so only using this to generate map
                parameter_bspline_map = parameter_bspline_curve.compute_eval_map_points(np.linspace(0., 1., num_sections))
                sectional_rotation_map = parameter_bspline_map

            # add section property map to section properties map to create a single map            
            sectional_rotations_map[(property_index*num_sections):((property_index+1)*num_sections), parameter_starting_index:parameter_ending_index] = sectional_rotation_map

            parameter_starting_index = parameter_ending_index
        
        sectional_rotations_map = sectional_rotations_map.tocsc()

        self.sectional_rotations_map = sectional_rotations_map
        return sectional_rotations_map

    
    def generate_ffd_control_points_map(self):
        if self.ffd_control_points_map is not None:
            return self.ffd_control_points_map

        num_section_properties = self.num_sections*(self.num_affine_properties)
        num_points_per_section = self.nyp * self.nzp

        initial_ffd_block_control_points = self.local_control_points.reshape((-1,3)).copy()

        # Preallocate ffd block control points maps
        num_ffd_block_control_points =  self.local_control_points.reshape((-1,3)).shape[0]
        ffd_block_control_points_map = np.zeros((num_ffd_block_control_points, 3, num_section_properties))
        ffd_block_control_points_x_map = sps.lil_array((num_ffd_block_control_points, num_section_properties))
        ffd_block_control_points_y_map = sps.lil_array((num_ffd_block_control_points, num_section_properties))
        ffd_block_control_points_z_map = sps.lil_array((num_ffd_block_control_points, num_section_properties))

        # Assemble ffd block control points maps
        for section_number in range(self.num_sections):
            start_index = section_number*num_points_per_section
            end_index = (section_number+1)*num_points_per_section
            ffd_block_control_points_x_map[start_index:end_index, section_number] = 1.  # Translation x
            ffd_block_control_points_y_map[start_index:end_index, self.num_sections+section_number] = 1. # Translation y
            ffd_block_control_points_z_map[start_index:end_index, (self.num_sections*2)+section_number] = 1. # Translation z
            ffd_block_control_points_x_map[start_index:end_index, (self.num_sections*3)+section_number] = initial_ffd_block_control_points[start_index:end_index, 0] # Scale x
            ffd_block_control_points_y_map[start_index:end_index, (self.num_sections*4)+section_number] = initial_ffd_block_control_points[start_index:end_index, 1]*self.initial_scale_y   # Scale y
            ffd_block_control_points_z_map[start_index:end_index, (self.num_sections*5)+section_number] = initial_ffd_block_control_points[start_index:end_index, 2]*self.initial_scale_z   # Scale z


        self.ffd_control_points_x_map = ffd_block_control_points_x_map.tocsc()
        self.ffd_control_points_y_map = ffd_block_control_points_y_map.tocsc()
        self.ffd_control_points_z_map = ffd_block_control_points_z_map.tocsc()

        ffd_block_control_points_map[:,0,:] = np.array(self.ffd_control_points_x_map.todense())
        ffd_block_control_points_map[:,1,:] = np.array(self.ffd_control_points_y_map.todense())
        ffd_block_control_points_map[:,2,:] = np.array(self.ffd_control_points_z_map.todense())
        self.ffd_control_points_map = ffd_block_control_points_map

        return self.ffd_control_points_map


    def project_points_FFD(self):

        if self.evaluation_map is not None:
            return self.evaluation_map

        if self.num_dof == 0:
            return None

        self._generate_embedded_coordinates()

        embedded_points = np.concatenate(self.embedded_points, axis=0 )
        
        self.evaluation_map = self.b_spline_volume.compute_projection_eval_map(embedded_points)

        return self.evaluation_map


    def generate_cost_matrix(self):
        # Preallocate cost matrix (alphas in x.T.dot(alphas).dot(x))
        cost_matrix = sps.csc_array(sps.eye(self.num_affine_free_dof))

        parameter_starting_index = 0
        for ffd_parameter in self.parameters:
            if type(ffd_parameter) is FFDRotationXParameter or \
                type(ffd_parameter) is FFDRotationYParameter or \
                type(ffd_parameter) is FFDRotationZParameter or \
                ffd_parameter.value is not None or ffd_parameter.connection_name is not None:
                continue

            parameter_num_dof = ffd_parameter.num_dof
            parameter_ending_index = parameter_starting_index + parameter_num_dof
            indices = np.ix_(np.arange(parameter_starting_index, parameter_ending_index))

            cost_matrix[indices, indices] = ffd_parameter.cost_factor/parameter_num_dof

            parameter_starting_index = parameter_ending_index

        self.cost_matrix = cost_matrix
        return cost_matrix



    def evaluate_section_properties(self):
        raise Exception('Sorry, this has not been implemented yet. :( ')
        # To implement, just dot section property map with ffd parameters.
        parameter_matrix = constant_parameter_evaluated + linear_parameter_evaluated + high_order_parameter_evaluated
        parameter = csdl.reshape(parameter_matrix, new_shape=parameter_matrix.shape[0])


    def _generate_local_coordinate_directions(self, local_axes):
        for k, v in local_axes.items():
            if k[0].lower() == 'x':
                if v is None:
                    self.xprime = np.tile(np.array([1,0,0]), (self.nxp,1))
                elif v.shape == (3,):
                    self.xprime = np.tile(v, (self.nxp,1))
                elif v.shape == (self.nxp, 3):
                    self.xprime = v
                else:
                    raise Exception('Xprime must either be a: (3,) array, (nxp, 3) array, or None')

            elif k[0].lower() == 'y':
                if v is None:
                    self.yprime = np.tile(np.array([0,1,0]), (self.nxp,1))
                elif v.shape == (3,):
                    self.yprime = np.tile(v, (self.nxp,1))
                elif v.shape == (self.nxp, 3):
                    self.yprime = v
                else:
                    raise Exception('Yprime must either be a: (3,) array, (nxp, 3) array, or None')
                    
            elif k[0].lower() == 'z':
                if v is None:
                    self.zprime = np.tile(np.array([0,0,1]), (self.nxp,1))
                elif v.shape == (3,):
                    self.zprime = np.tile(v, (self.nxp,1))
                elif v.shape == (self.nxp, 3):
                    self.zprime = v
                else:
                    raise Exception('Zprime must either be a: (3,) array, (nxp, 3) array, or None')
                    
            else:
                raise Exception('Keys need to begin with either an: x,y,z character to denote local axes!')


    def _generate_ffd_origin(self):
        # It doesn't really matter, so let's take the middle
        u_vec = 0.5
        v_vec = 0.5
        w_vec = 0.5
    
        self.ffd_origin = self.b_spline_volume.evaluate_points(u_vec, v_vec, w_vec)

    def _generate_ffd_axes(self):
        # It doesn't particularly matter, so let's ensure it's a normal basis by using the Earth-fixed frame
        self.ffd_xprime = np.array([1, 0, 0])
        self.ffd_yprime = np.array([0, 1, 0])
        self.ffd_zprime = np.array([0, 0, 1])


    def _generate_exterior_points(self, nu, nv, nw):

        v_vec_front, w_vec_front = np.mgrid[0:1:nv*1j, 0:1:nw*1j]
        u_vec_front = np.zeros(v_vec_front.shape) 

        v_vec_back, w_vec_back = np.mgrid[0:1:nv*1j, 0:1:nw*1j]
        u_vec_back = np.ones(v_vec_front.shape)

        u_vec_bot, v_vec_bot = np.mgrid[0:1:nu*1j, 0:1:nv*1j]
        w_vec_bot = np.zeros(v_vec_bot.shape)

        u_vec_top, v_vec_top = np.mgrid[0:1:nu*1j, 0:1:nv*1j]
        w_vec_top = np.ones(v_vec_bot.shape)

        u_vec_left, w_vec_left = np.mgrid[0:1:nu*1j, 0:1:nw*1j]
        v_vec_left = np.zeros(u_vec_left.shape)

        u_vec_right, w_vec_right = np.mgrid[0:1:nu*1j, 0:1:nw*1j]
        v_vec_right = np.ones(u_vec_right.shape)

        u_points = np.concatenate((u_vec_front.flatten(), u_vec_back.flatten(), u_vec_bot.flatten(), u_vec_top.flatten(), u_vec_left.flatten(), u_vec_right.flatten()))
        v_points = np.concatenate((v_vec_front.flatten(), v_vec_back.flatten(), v_vec_bot.flatten(), v_vec_top.flatten(), v_vec_left.flatten(), v_vec_right.flatten()))
        w_points = np.concatenate((w_vec_front.flatten(), w_vec_back.flatten(), w_vec_bot.flatten(), w_vec_top.flatten(), w_vec_left.flatten(), w_vec_right.flatten()))
        
        exterior_points = self.b_spline_volume.evaluate_points(u_points, v_points, w_points)

        return exterior_points
    
    def add_embedded_entities(self, entities):
        for i in entities:
            self.embedded_entities_pointers.append(i)

    def _generate_embedded_coordinates(self):

        for embedded_entity in self.embedded_entities_pointers:
            # print('i: ', i)
            if isinstance(embedded_entity, BSplineCurve) or isinstance(embedded_entity, BSplineSurface) or isinstance(embedded_entity, BSplineVolume):
                # print(i.control_points.shape)
                # print(len(i.control_points))
                self.embedded_points.append(embedded_entity.control_points)  # Control points are given in Cartesian Coordinates
                self.embedded_entities_indices.append(len(embedded_entity.control_points))
            else:
                self.embedded_points.append(embedded_entity.physical_coordinates)  # Here i is a PointSet, make sure that the PointSet is evalauted
                self.embedded_entities_indices.append(len(embedded_entity.physical_coordinates))

    def plot(self, nu, nv, nw):
        exterior_points = self._generate_exterior_points(nu, nv, nw)
        vp_init = Plotter()
        vps = []
        vps1 = Points(exterior_points, r=8, c = 'red')
        vps.append(vps1)       

        if self.embedded_entities_pointers == []:
            pass
        
        elif self.embedded_points == []:
            self._generate_embedded_coordinates()

        
        for i in self.embedded_points:
            vps2 =  Points(i, r=8, c='blue')
            vps.append(vps2)

        vp_init.show(vps, 'Bspline Volume', axes=1, viewup="z", interactive = True)


    def evaluate(self, control_points=None):
        if control_points is not None:
            self.control_points = control_points
        embedded_points = self.evaluation_map.dot(self.control_points)
        return embedded_points


@dataclass
class FFDParameter:
    '''
    Inputs:
    - degree: The degree of the bspline curve to represent the parameter.
    - num_dof: The number of degrees of freedom for the parameter (control points for bspline).
    - cost_factor: The cost weighting on using the parameter when achieving geometric inputs and constraints.
    - value: The prescribed value for parameter
    - connection_name: The name that be used to create a csdl hanging input.
    '''
    degree : int = 0
    num_dof : int = 10
    value : np.ndarray = None
    connection_name : str = None

@dataclass
class FFDTranslationXParameter(FFDParameter):
    cost_factor : float = 1.

@dataclass
class FFDTranslationYParameter(FFDParameter):
    cost_factor : float = 1.

@dataclass
class FFDTranslationZParameter(FFDParameter):
    cost_factor : float = 1.

@dataclass
class FFDScaleYParameter(FFDParameter):
    cost_factor : float = 1.

@dataclass
class FFDScaleZParameter(FFDParameter):
    cost_factor : float = 1.

@dataclass
class FFDRotationXParameter(FFDParameter):
    pass

@dataclass
class FFDRotationYParameter(FFDParameter):
    pass

@dataclass
class FFDRotationZParameter(FFDParameter):
    pass



if __name__ == "__main__":

    from src.caddee.concept.geometry.geocore.utils.generate_ffd import create_ffd
    nxp = 5
    nyp = 5
    nzp = 5

    point000 = np.array([170. ,0. ,100.])
    point010 = np.array([130., 230., 100.])
    point001 = np.array([170., 0., 170.])
    point011 = np.array([130., 230., 170.])
    
    point100 = np.array([240. ,0. ,100.])
    point101 = np.array([240. ,0. ,170.])
    point110 = np.array([200. ,230. ,100.])
    point111 = np.array([200. ,230. ,170.])

    control_points = np.zeros((2,2,2,3))
    
    control_points[0,0,0,:] = point000
    control_points[0,0,1,:] = point001

    control_points[0,1,0,:] = point010
    control_points[0,1,1,:] = point011
    
    control_points[1,0,0,:] = point100
    control_points[1,0,1,:] = point101
    
    control_points[1,1,0,:] = point110
    control_points[1,1,1,:] = point111

    ffd_control_points = create_ffd(control_points, nxp, nyp, nzp)

    # ''' Camber surface creation script for this case '''
    # path_name = '../examples/CAD/'
    # file_name = 'eVTOL.stp'
    # geo = DesignGeometry(path_name + file_name)

    # wing_surface_names = [
    # 'Surf_WFWKRQIMCA, Wing, 0, 12', 'Surf_WFWKRQIMCA, Wing, 0, 13', 
    # 'Surf_WFWKRQIMCA, Wing, 0, 14', 'Surf_WFWKRQIMCA, Wing, 0, 15', 
    # ]

    # bspline_entities = [geo.input_bspline_entity_dict[wing_surface_names[0]],
    #    geo.input_bspline_entity_dict[wing_surface_names[1]], 
    #    geo.input_bspline_entity_dict[wing_surface_names[2]],
    #    geo.input_bspline_entity_dict[wing_surface_names[3]]]


    # local_axes = {'xprime': np.array([1,0,0]), 'yprime': np.array([0,1,0]), 'zprime': np.array([0,0,1]) }

    # test_ffd = FFD('test', ffd_control_points, embedded_entities_pointers=bspline_entities)
    test_ffd = FFD('test', ffd_control_points)
    
    test_ffd.add_shape_parameter('rot_x', 'linear', 2, 3, False, val=1.0)
    test_ffd.add_shape_parameter('rot_x', 'quadratic', 3, 4, False, val=1.0)

    test_ffd.add_shape_parameter('rot_y', 'linear', 3,4, True, val=2.0)

    # print('full dict: ', test_ffd.properties_dict)
    # print('\n')
    # print('rot_x: ', test_ffd.properties_dict['rot_x']['parameters'])   
    # print('\n')
    # print('rot_y: ',test_ffd.properties_dict['rot_y']['parameters'])
    # print('\n')

    # print('properties_list: ', test_ffd.properties_dict.keys())
    for property_name, property_dict in test_ffd.properties_dict.items():
        print('property_name: ', property_name)
        print('property_dict: ', property_dict)
        
        # property_var = ffd_block_csdl_model.create_input(property_name)
        # parameters_list = test_ffd.properties_dict[property_name]
        # print('parameters_list: ', parameters_list)
        # print(property_dict['parameters'])
        for parameter_name, parameter_info in property_dict['parameters'].items():
                print('parameter_name: ', parameter_name)
                # for parameter_info in parameter_dict():
                    # print('parameter_name: ', parameter_name)
                print('parameter_info: ', parameter_info)
    
    # print(test_ffd.properties_dict['rot_x']['parameters'][1])    
    

    # print(test_ffd.embedded_entities_indices)

    # print('ORIGINAL CONTROL PTS: ', test_ffd.control_points[0,:,:,:])
    # print('ORIGIN: ', test_ffd.origin[0,:])
    # print('LOCAL CONTROL PTS: ', test_ffd.control_points[0,:,:,:] - test_ffd.origin[0,:])

    # print(test_ffd.local_control_points)
    # print(test_ffd.BSplineVolume.control_points)

    test_ffd.plot(nxp, nyp, nzp)

    # test_ffd.translate_control_points(offset=np.array([10., 50., 100.]))

    # print(test_ffd.BSplineVolume.control_points)

    # test_ffd.plot(nu, nv, nw)



class RectangularFFDBlock(FFDBlock):
    pass


class TranslatingFFDBlock(RectangularFFDBlock):
    def __init__(self, name: str, control_points, embedded_entities_pointers=[], local_axes=dict(xprime=None, yprime=None, zprime=None),
            uvw_axis_indices=(0, 1, 2), section_origins=None, local_control_points=None, shape=None, order_u=10, order_v=10, order_w=10,
            knots_u=None, knots_v=None, knots_w=None):
        super().__init__(name, control_points, embedded_entities_pointers, local_axes, uvw_axis_indices, section_origins,
                local_control_points, shape, order_u, order_v, order_w, knots_u, knots_v, knots_w)
        

