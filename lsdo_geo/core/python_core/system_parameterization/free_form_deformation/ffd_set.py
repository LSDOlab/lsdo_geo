import numpy as np
import scipy.sparse as sps

import matplotlib.pyplot as plt
import vedo
import time

from joblib import Parallel, delayed
from math import sqrt

class FFDSet:
    '''
    A set of free form deformation (FFD) blocks. This object is created to vectorize calculations.
    '''
    def __init__(self, name:str=None, ffd_blocks:dict={}) -> None:
        '''
        Parameters
        -----------
        name : str
            The name of the set of FFD blocks.
        ffd_blocks : list of FFDBlock objects
            The list containing the FFD blocks that form the set.
        '''
        self.name = name
        self.ffd_blocks = ffd_blocks

    def evaluate(self, inputs):
        '''
        Evaluates the FFD model formed from the set of FFD blocks.
        '''

from caddee.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

# class SRBGFFDSet(FFDSet):     # Could potentially inherit in future, but have to figure out how that relationship will work
class SRBGFFDSet:
    '''
    A set of sectioned rectangular b-spline geometric free form deformation (SRBGFFD) blocks. This object is created to vectorize calculations.
    # NOTE: The evaluation output will output a vector of all the points it has specifically manipulated.
                It will be up to post-processing to parse the output and assign the points correctly.
    # Tangential NOTE: Can't collapse entire evaluation into a single map because of nonlinear rotations :(
                        -- Because of this, a general FFDSet object seems prohibitive since the evaluations of each different type of FFD block is different.
    '''
    def __init__(self, name:str=None, ffd_blocks:dict={}) -> None:
        '''
        Parameters
        -----------
        name : str
            The name of the set of SRBGFFD blocks.
        ffd_blocks : dict {name string : SRBGFFDBlock objects}
            The list containing the SRBGFFD blocks that form the set.
        '''
        # Attributes for defining the object
        self.name = name
        self.ffd_blocks = ffd_blocks

        # maps that will be used to perform the free form deformations
        self.free_affine_section_properties_map = None
        self.prescribed_affine_section_properties_map = None
        self.rotational_section_properties_map = None
        self.affine_block_deformations_map = None
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

        # self.num_control_points = np.cumprod(primitive.shape[:-1])[-1]
        self.num_dof = 0
        self.num_affine_dof = 0
        self.num_affine_free_dof = 0
        self.num_affine_prescribed_dof = 0
        self.num_rotational_dof = 0
        self.num_sections = 0
        self.num_section_properties = 0
        self.num_affine_sections = 0
        self.num_affine_section_properties = 0
        self.num_rotational_section_properties = 0
        self.num_control_points = 0
        self.num_affine_control_points = 0
        self.num_affine_free_control_points = 0
        self.num_embedded_points = 0

        # attributes for storing current states
        self.free_affine_dof = None
        self.prescribed_affine_dof = None
        self.rotational_dof = None
        self.affine_section_properties = None
        self.translations = None
        self.rotational_section_properties = None
        self.affine_deformed_control_points = None
        self.rotated_control_points_local_frame = None
        self.control_points = None
        self.embedded_points = None

        # Promotion of key FFD Block attributes to the collectivized level
        self.embedded_entities = {}
        for ffd_block in list(self.ffd_blocks.values()):
            self.embedded_entities.update(ffd_block.embedded_entities)



    def add_ffd_block(self, ffd_block:SRBGFFDBlock):
        '''
        Adds a Sectioned Rectangular B-spline Geometric Free Form Deformation Block to the set of SRBGFFDBlocks.
        '''
        self.ffd_blocks[ffd_block.name] = ffd_block


    def assemble_affine_section_properties_maps(self):
        '''
        Assembles affine section property maps for the set of SRBGFFD blocks. (affine dof --> affine section properties)
        '''
        if self.free_affine_section_properties_map is not None:
            return self.free_affine_section_properties_map, self.prescribed_affine_section_properties_map

        if self.num_affine_dof == 0:
            return

        # Assemble whole maps from individual FFD block maps. They are diagonal block matrices.
        # -- free section properties map (affine free dof --> section properties)
        # -- prescribed section properties map (affine prescribed dof --> section properties)
        free_affine_section_properties_maps = []
        prescribed_affine_section_properties_maps = []
        for ffd_block in list(self.active_ffd_blocks.values()):
            if ffd_block.num_affine_dof == 0:
                continue

            if ffd_block.num_affine_free_dof != 0:
                free_affine_section_properties_maps.append(ffd_block.free_affine_section_properties_map)
            if ffd_block.num_affine_prescribed_dof != 0:
                prescribed_affine_section_properties_maps.append(ffd_block.prescribed_affine_section_properties_map)

        if free_affine_section_properties_maps:
            free_affine_section_properties_map = sps.block_diag(tuple(free_affine_section_properties_maps))
            self.free_affine_section_properties_map = free_affine_section_properties_map.tocsc()
        else:
            self.free_affine_section_properties_map = np.zeros((self.num_affine_section_properties,0))

        if prescribed_affine_section_properties_maps:
            prescribed_affine_section_properties_map = sps.block_diag(tuple(prescribed_affine_section_properties_maps))
            self.prescribed_affine_section_properties_map = prescribed_affine_section_properties_map.tocsc()
        else:
            self.free_affine_section_properties_map = np.zeros((self.num_affine_section_properties,0))
        

        return self.free_affine_section_properties_map, self.prescribed_affine_section_properties_map


    def assemble_rotational_section_properties_map(self):
        '''
        Assembles the map that maps from prescribed rotational dof to rotational section properties (rotational dof --> rotational section properties)
        '''
        if self.rotational_section_properties_map is not None:
            return self.rotational_section_properties_map

        if self.num_rotational_dof == 0:
            self.rotational_section_properties_map = np.array([]).reshape((0,0))
            return
        
        # Assemble whole maps from individual FFD block maps. They are diagonal block matrices.
        # -- sectional rotations map (rotational dof --> section properties)
        rotational_section_property_maps = []
        for ffd_block in list(self.ffd_blocks.values()):
            if ffd_block.num_rotational_dof == 0:
                continue

            rotational_section_property_maps.append(ffd_block.rotational_section_properties_map)

        rotational_section_property_maps = tuple(rotational_section_property_maps)

        rotational_section_properties_map = sps.block_diag(rotational_section_property_maps)
        
        self.rotational_section_properties_map = rotational_section_properties_map.tocsc()

        return self.rotational_section_properties_map

    
    def assemble_affine_block_deformations_map(self):
        '''
        Assembles the local control points map.
        The control points map maps (affine section properties --> local FFD set control points)
        '''
        if self.affine_block_deformations_map is not None:
            return self.affine_block_deformations_map

        # if self.num_affine_dof == 0:
        #     return

        ffd_block_local_control_points_maps = []

        # Assemble whole maps from individual FFD block maps. They are diagonal block matrices.
        # --  map (section properties --> local affine control points)
        for ffd_block in list(self.active_ffd_blocks.values()):
            ffd_block_local_control_points_maps.append(ffd_block.affine_block_deformations_map)
        
        if self.num_dof != 0:
            affine_block_deformations_map = sps.block_diag(tuple(ffd_block_local_control_points_maps)).tocsc()
        else:
            affine_block_deformations_map = sps.csc_matrix((0,0))
        self.affine_block_deformations_map = affine_block_deformations_map
        return self.affine_block_deformations_map


    def assemble_local_to_global_control_points_map(self):
        '''
        Assembles maps (rotation matrices) for converting control points from local frame to global frame. (local control points --> global control points)

        NOTE: The current approach is to rotate each block individually instead of assembling a large map to save memory and computation time.
        '''
        local_to_global_control_points_maps = []
        for ffd_block in list(self.ffd_blocks.values()):
            if ffd_block.num_dof == 0:
                continue

            local_to_global_control_points_maps.append(ffd_block.local_to_global_rotation)
        self.local_to_global_control_points_maps = local_to_global_control_points_maps
        return self.local_to_global_control_points_maps


    def assemble_embedded_entities_map(self):
        '''
        Assembles the map to calculate the embedded entities (points) from the ffd set control points. (ffd set control points --> embedded entities)

        developer note: This only evaluates what is embedded in this set. It does not organize the output
                to account for non-parameterized points or points parameterized using other methods.
        '''
        if self.embedded_entities_map is not None:
            return self.embedded_entities_map

        embedded_entities_maps = []
        for ffd_block in list(self.active_ffd_blocks.values()):
            embedded_entities_maps.append(ffd_block.embedded_entities_map)
        if self.num_dof != 0:
            embedded_entities_map = sps.block_diag(tuple(embedded_entities_maps)).tocsc()
        else:
            embedded_entities_map = sps.csc_matrix((0,0))
        self.embedded_entities_map = embedded_entities_map
        return self.embedded_entities_map

    
    def calculate_ffd_map(self):
        '''
        TODO Multiplies baseline maps to create map from ffd dof to ffd local control points.

        This map is to avoid repeatedly performing the same large matrix-matrix or matrix-vector multiplications every iteration.
        '''
        if self.ffd_free_dof_to_affine_ffd_control_points_map is not None:
            return self.ffd_free_dof_to_affine_ffd_control_points_map

        

        return self.ffd_free_dof_to_affine_ffd_control_points_map


    def assemble_cost_matrix(self):
        if self.num_affine_free_dof == 0:
            return

        if self.cost_matrix is not None:
            return self.cost_matrix

        ffd_block_cost_matrices_with_free_dof = []
        for ffd_block in self.ffd_blocks:
            if ffd_block.num_affine_free_dof != 0:
                ffd_block_cost_matrices_with_free_dof.append(ffd_block.cost_matrix)
        ffd_block_cost_matrices_with_free_dof = tuple(ffd_block_cost_matrices_with_free_dof)

        cost_matrix = sps.block_diag(ffd_block_cost_matrices_with_free_dof)
        self.cost_matrix = cost_matrix.tocsc()

        return self.cost_matrix


    # def evaluate_section_properties(self):
    #     raise Exception('Sorry, this has not been implemented yet. :( ')
    #     # To implement, just dot section property map with ffd parameters.
    #     parameter_matrix = constant_parameter_evaluated + linear_parameter_evaluated + high_order_parameter_evaluated
    #     parameter = csdl.reshape(parameter_matrix, new_shape=parameter_matrix.shape[0])


    def assemble(self, project_embedded_entities=True):
        if len(self.ffd_blocks) == 0:
            return

        self.assemble_affine_section_properties_maps()
        self.assemble_rotational_section_properties_map()
        self.assemble_affine_block_deformations_map()
        if project_embedded_entities:
            self.assemble_embedded_entities_map()
        self.assemble_cost_matrix()

    def count_attributes(self):
        '''
        Sums the attributes from all the children FFD blocks to determine the number of each type of state.
        '''
        self.num_dof = 0
        self.num_affine_dof = 0
        self.num_affine_free_dof = 0
        self.num_affine_prescribed_dof = 0
        self.num_rotational_dof = 0
        self.num_sections = 0
        self.num_section_properties = 0
        self.num_affine_sections = 0
        self.num_affine_section_properties = 0
        self.num_rotational_section_properties = 0
        self.num_control_points = 0
        self.num_affine_control_points = 0
        self.num_affine_free_control_points = 0
        self.num_embedded_points = 0

        for ffd_block in list(self.active_ffd_blocks.values()):
            self.num_dof += ffd_block.num_dof
            self.num_affine_dof += ffd_block.num_affine_dof
            self.num_affine_free_dof += ffd_block.num_affine_free_dof
            self.num_affine_prescribed_dof += ffd_block.num_affine_prescribed_dof
            self.num_rotational_dof += ffd_block.num_rotational_dof
            self.num_sections += ffd_block.num_sections
            self.num_section_properties += ffd_block.num_affine_properties * ffd_block.num_sections
            self.num_control_points += ffd_block.num_control_points
            if ffd_block.num_affine_dof != 0:
                # self.num_affine_sections += ffd_block.num_sections
                self.num_affine_sections += ffd_block.num_sections
                self.num_affine_section_properties += ffd_block.num_sections * ffd_block.num_affine_properties
                self.num_affine_control_points += ffd_block.num_control_points
                if ffd_block.num_affine_free_dof != 0:
                    self.num_affine_free_control_points += ffd_block.num_control_points
            if ffd_block.num_rotational_dof != 0:
                self.num_rotational_section_properties += ffd_block.num_sections * ffd_block.num_rotational_properties
            # if project_points:
                # self.num_embedded_points += ffd_block.embedded_entities_map.shape[0]
            self.num_embedded_points += ffd_block.embedded_entities_map.shape[0]

    def setup_default_states(self):
        '''
        Generates the default states from the default states of the ffd blocks.
        '''
        self.free_affine_dof = np.array([])
        self.prescribed_affine_dof = np.array([])
        self.rotational_dof = np.array([])
        self.affine_section_properties = np.array([])
        self.translations = np.array([])
        self.rotational_section_properties = np.array([])
        self.affine_deformed_control_points = np.array([]).reshape((0,0))
        self.rotated_control_points_local_frame = np.array([]).reshape((0,0))
        self.control_points = np.array([]).reshape((0,0))
        self.embedded_points = np.array([]).reshape((0,0))

        for ffd_block in list(self.active_ffd_blocks.values()):
            if self.free_affine_dof.size == 0:
                self.free_affine_dof = ffd_block.free_affine_dof
            else:
                self.free_affine_dof = np.append(self.free_affine_dof, ffd_block.free_affine_dof)
            
            if self.prescribed_affine_dof.size == 0:
                self.prescribed_affine_dof = ffd_block.prescribed_affine_dof
            else:
                self.prescribed_affine_dof = np.append(self.prescribed_affine_dof, ffd_block.prescribed_affine_dof)

            if self.rotational_dof.size == 0:
                self.rotational_dof = ffd_block.rotational_dof
            else:
                self.rotational_dof = np.append(self.rotational_dof, ffd_block.rotational_dof)

            if self.affine_section_properties.size == 0:
                self.affine_section_properties = ffd_block.affine_section_properties
            else:
                self.affine_section_properties = np.append(self.affine_section_properties, ffd_block.affine_section_properties)

            if self.translations.size == 0:
                self.translations = ffd_block.translations
            else:
                self.translations = np.append(self.translations, ffd_block.translations)

            if ffd_block.num_rotational_dof != 0:
                if self.rotational_section_properties.size == 0:
                    self.rotational_section_properties = ffd_block.rotational_section_properties
                else:
                    self.rotational_section_properties = np.append(self.rotational_section_properties, ffd_block.rotational_section_properties)

            if self.affine_deformed_control_points.size == 0:
                self.affine_deformed_control_points = ffd_block.affine_deformed_control_points
            else:
                self.affine_deformed_control_points = np.vstack((self.affine_deformed_control_points, ffd_block.affine_deformed_control_points))

            if self.rotated_control_points_local_frame.size == 0:
                self.rotated_control_points_local_frame = ffd_block.rotated_control_points_local_frame
            else:
                self.rotated_control_points_local_frame = np.vstack((self.rotated_control_points_local_frame, ffd_block.rotated_control_points_local_frame))

            if self.control_points.size == 0:
                self.control_points = ffd_block.control_points
            else:
                self.control_points = np.vstack((self.control_points, ffd_block.control_points))

            if self.embedded_points.size == 0:
                self.embedded_points = ffd_block.embedded_points
            else:
                self.embedded_points = np.vstack((self.embedded_points, ffd_block.embedded_points))

        # if self.num_dof == 0:
        #     self.free_affine_dof = np.array([])
        #     self.prescribed_affine_dof = np.array([])
        #     self.rotational_dof = np.array([])
        #     self.affine_section_properties = np.array([])
        #     self.translations = np.array([])
        #     self.rotational_section_properties = np.array([])
        #     self.affine_deformed_control_points = np.array([]).reshape((0,0))
        #     self.rotated_control_points_local_frame = np.array([]).reshape((0,0))
        #     self.control_points = np.array([]).reshape((0,0))
        #     self.embedded_points = np.array([]).reshape((0,0))


    def setup(self, project_embedded_entities=True):
        '''
        Sets up the FFD set for evaluation by assembling maps and setting up default states.

        This step precomputes everything that can be precomputed to avoid unnecessary recomputation.
        '''
        # Get active FFD blocks
        self.active_ffd_blocks = {}
        for ffd_block in list(self.ffd_blocks.values()):
            if ffd_block.num_dof != 0:
                self.active_ffd_blocks[ffd_block.name] = ffd_block

        # Setup active FFD blocks
        # t1 = time.time()
        # print('starting FFD projections')
        # if project_points:
        #     geometry_control_points_maps = Parallel(n_jobs=14)(delayed(ffd_block_project)(ffd_block, project_points) for ffd_block in self.active_ffd_blocks)
        #     for i, ffd_block in enumerate(self.active_ffd_blocks):
        #         ffd_block.evaluation_map = geometry_control_points_maps[i]
        # t2 = time.time()
        # print('FFD projection duration: ', t2-t1)

        # parallelize this!!
        for ffd_block in list(self.ffd_blocks.values()):
            ffd_block.setup(project_embedded_entities=project_embedded_entities)

        self.count_attributes()
        self.setup_default_states()
        self.assemble(project_embedded_entities=project_embedded_entities)

    # def evaluate(self, control_points=None):
    #     if control_points is not None:
    #         self.control_points = control_points
    #     embedded_points = self.evaluation_map.dot(self.control_points)
    #     return embedded_points


    def evaluate_affine_section_properties(self, free_affine_dof=None, prescribed_affine_dof=None):
        '''
        Evaluates the section properties from input of ffd dof
        '''
        if self.num_affine_dof == 0:
            NUM_PARAMETRIC_DIMENSIONS = 3
            translations = np.zeros((self.num_sections, NUM_PARAMETRIC_DIMENSIONS))
            return

        if free_affine_dof is None:
            free_affine_dof = self.free_affine_dof
        if prescribed_affine_dof is None:
            prescribed_affine_dof = self.prescribed_affine_dof

        affine_section_properties_free_component = self.free_affine_section_properties_map.dot(free_affine_dof)
        affine_section_properties_prescribed_component = self.prescribed_affine_section_properties_map.dot(prescribed_affine_dof)

        affine_section_properties = np.zeros((self.num_affine_section_properties,))
        translations = None

        ffd_block_section_properties_starting_index = 0
        for ffd_block in list(self.active_ffd_blocks.values()):
            ffd_block_scaling_properties_starting_index = ffd_block_section_properties_starting_index + ffd_block.num_sections*(ffd_block.num_affine_properties-ffd_block.num_scaling_properties)
            ffd_block_section_properties_ending_index = ffd_block_section_properties_starting_index + ffd_block.num_affine_section_properties

            # Use calculated values for non-scaling parameters
            affine_section_properties[ffd_block_section_properties_starting_index:ffd_block_scaling_properties_starting_index] = \
                affine_section_properties_free_component[ffd_block_section_properties_starting_index:ffd_block_scaling_properties_starting_index] \
                + affine_section_properties_prescribed_component[ffd_block_section_properties_starting_index:ffd_block_scaling_properties_starting_index]
            
            # Add 1 to scaling parameters to make initial scaling=1.
            affine_section_properties[ffd_block_scaling_properties_starting_index:ffd_block_section_properties_ending_index] = \
                affine_section_properties_free_component[ffd_block_scaling_properties_starting_index:ffd_block_section_properties_ending_index] \
                + affine_section_properties_prescribed_component[ffd_block_scaling_properties_starting_index:ffd_block_section_properties_ending_index] \
                + 1.  # adding 1 which is initial scale value
            
            if translations is None:
                translations = affine_section_properties[ffd_block_section_properties_starting_index:ffd_block_scaling_properties_starting_index]
            else:
                translations = np.append(translations, 
                                    affine_section_properties[ffd_block_section_properties_starting_index:ffd_block_scaling_properties_starting_index])
            
            ffd_block_section_properties_starting_index = ffd_block_section_properties_ending_index

        self.affine_section_properties = affine_section_properties
        self.translations = translations

        return affine_section_properties
    

    def evaluate_rotational_section_properties(self, rotational_dof:np.ndarray=None):
        '''
        Evaluates the section rotations from input of ffd dof
        '''
        if rotational_dof is None:
            rotational_dof = self.rotational_dof

        rotational_section_properties = self.rotational_section_properties_map.dot(rotational_dof)
        self.rotational_section_properties = rotational_section_properties

        plot=False
        if plot:
            ffd_block_dof_starting_index = 0
            ffd_block_property_starting_index = 0
            for ffd_block in list(self.active_ffd_blocks.values()):
                ffd_block_dof_ending_index = ffd_block_dof_starting_index + ffd_block.num_rotational_dof
                ffd_block_property_ending_index = ffd_block_property_starting_index + ffd_block.num_sections*ffd_block.num_rotational_properties

                ffd_block.plot_parameter_curves(rotational_dof[ffd_block_dof_starting_index:ffd_block_dof_ending_index],
                                                rotational_section_properties[ffd_block_property_starting_index:ffd_block_property_ending_index],
                                                show=True)
                
                ffd_block_dof_starting_index = ffd_block_dof_ending_index
                ffd_block_property_starting_index = ffd_block_property_ending_index

        return rotational_section_properties


    def evaluate_affine_block_deformations(self, affine_section_properties=None, plot=False):
        '''
        Evaluates the local control points of the FFD block given the affine section properties and section rotations
        '''
        if affine_section_properties is None:
            affine_section_properties = self.affine_section_properties

        affine_deformed_control_points_flattened = self.affine_block_deformations_map.dot(affine_section_properties)
        NUM_PARAMETRIC_DIMENSIONS = 3       # This type of FFD block has 3 parametric dimensions by definition.
        affine_deformed_control_points = affine_deformed_control_points_flattened.reshape((self.num_control_points, NUM_PARAMETRIC_DIMENSIONS))
        self.affine_deformed_control_points = affine_deformed_control_points

        if plot:
            starting_index = 0
            for ffd_block in list(self.active_ffd_blocks.values()):
                ending_index = starting_index + ffd_block.num_control_points
                ffd_block_affine_tranformed_control_points = affine_deformed_control_points[starting_index:ending_index]
                starting_index = ending_index
                ffd_block.plot_sections(control_points=ffd_block_affine_tranformed_control_points, offset_sections=True, plot_embedded_entities=False, opacity=0.75, show=True)

        return affine_deformed_control_points


    def evaluate_rotational_block_deformations(self, affine_deformed_control_points=None, translations=None, \
                                                                            rotational_section_properties=None, plot=False):
        '''
        Evaluates the control points of the FFD block in original coordinate frame by applying the rotational section properties (section rotations).
        '''

        # Processing inputs
        if affine_deformed_control_points is None:
            affine_deformed_control_points = self.affine_deformed_control_points
        if translations is None:
            translations_flattened = self.translations
            if translations_flattened is not None:
                NUM_PARAMETRIC_DIMENSIONS = 3
                translations = translations_flattened.reshape((self.num_sections, NUM_PARAMETRIC_DIMENSIONS), order='F')
            else:
                NUM_PARAMETRIC_DIMENSIONS = 3
                translations = np.zeros((self.num_sections, NUM_PARAMETRIC_DIMENSIONS))
        if rotational_section_properties is None:
            rotational_section_properties = self.rotational_section_properties

        # For simplicity for now (and to avoid storage/construction of a huge, dense rotation tensor, call each FFD rotation separately)
        NUM_PARAMETRIC_DIMENSIONS = 3
        rotated_control_points_local_frame = np.zeros((self.num_control_points,NUM_PARAMETRIC_DIMENSIONS))
        starting_index_control_points = 0
        starting_index_sections = 0
        starting_index_section_properties = 0
        for ffd_block in list(self.active_ffd_blocks.values()):
            ending_index_control_points = starting_index_control_points + ffd_block.num_control_points
            ending_index_sections = starting_index_sections + ffd_block.num_sections
            ending_index_section_properties = starting_index_section_properties + ffd_block.num_sections*ffd_block.num_rotational_properties

            if ffd_block.num_rotational_dof != 0:
                ffd_block_rotated_control_points_local_frame = ffd_block.evaluate_rotational_block_deformations(
                    affine_deformed_control_points[starting_index_control_points:ending_index_control_points], 
                    translations[starting_index_sections:ending_index_sections],
                    rotational_section_properties[starting_index_section_properties:ending_index_section_properties], plot=plot)
                starting_index_section_properties = ending_index_section_properties
            else:
                ffd_block_rotated_control_points_local_frame = affine_deformed_control_points[starting_index_control_points:ending_index_control_points]

            rotated_control_points_local_frame[starting_index_control_points:ending_index_control_points] = ffd_block_rotated_control_points_local_frame
            starting_index_control_points = ending_index_control_points
            starting_index_sections = ending_index_sections

        self.rotated_control_points_local_frame = rotated_control_points_local_frame

        return rotated_control_points_local_frame

        # Next model will perform rotation back to global frame

    def evaluate_control_points(self, rotated_control_points_local_frame:np.ndarray=None, plot:bool=False):
        '''
        Evaluates the control points of the FFD block in original coordinate frame by applying 
        bulk rotation and translation back to original coordinate frame.
        '''
        if rotated_control_points_local_frame is None:
            rotated_control_points_local_frame = self.rotated_control_points_local_frame
        
        NUM_PARAMETRIC_DIMENSIONS = 3
        control_points = np.zeros((self.num_control_points,NUM_PARAMETRIC_DIMENSIONS))
        starting_index = 0
        for ffd_block in list(self.active_ffd_blocks.values()):
            ending_index = starting_index + ffd_block.num_control_points

            ffd_block_control_points = ffd_block.evaluate_control_points(
                rotated_control_points_local_frame[starting_index:ending_index], plot=plot)

            control_points[starting_index:ending_index] = ffd_block_control_points
            starting_index = ending_index

        self.control_points = control_points

        ## TODO Add global plotting since it would make sense since everything is back in global frame

        return control_points


    def evaluate_embedded_entities(self, control_points=None, plot=False):
        '''
        Evaluates the entities embedded within the FFD block from an input of the FFD control points.
        '''
        if control_points is None:
            control_points = self.control_points

        embedded_entities = self.embedded_entities_map.dot(control_points)
        self.embedded_points = embedded_entities

        embedded_entity_starting_index = 0
        for ffd_block in list(self.active_ffd_blocks.values()):
            for embedded_entity in list(ffd_block.embedded_entities.values()):
                embedded_entity_ending_index = embedded_entity_starting_index + np.cumprod(embedded_entity.shape[:-1])[-1]
                embedded_entity.control_points = embedded_entities[embedded_entity_starting_index:embedded_entity_ending_index]
                embedded_entity_starting_index = embedded_entity_ending_index

        if plot:
            plotting_elements = []
            control_points_starting_index = 0
            for ffd_block in list(self.active_ffd_blocks.values()):
                for embedded_entity in list(ffd_block.embedded_entities.values()):
                    plotting_elements = embedded_entity.plot(plot_types=['mesh'], opacity=0.5, additional_plotting_elements=plotting_elements, show=False)
                plotting_elements.append(vedo.Points(embedded_entities, r=5).color('green'))
                control_points_ending_index = control_points_starting_index + ffd_block.num_control_points
                plotting_elements.append(ffd_block.plot_sections(control_points=control_points[control_points_starting_index:control_points_ending_index],
                            offset_sections=False, plot_embedded_entities=False, opacity=0.3, additional_plotting_elements=plotting_elements, show=False))

                control_points_starting_index = control_points_ending_index

            plotter = vedo.Plotter()
            plotter.show(plotting_elements, f'Embedded Entities in {self.name}', axes=1, viewup="z", interactive=True)

        return embedded_entities
    

    def assemble_csdl(self):
        '''
        Returns the CADDEE model for the FFD.
        '''
        from caddee.csdl_core.system_parameterization_csdl.ffd_csdl.ffd_csdl import FFDCSDL
        return FFDCSDL(ffd_set=self)


def ffd_block_project(ffd_block, project_points):
    if project_points:
        ffd_block.assemble()
        return ffd_block.project(ffd_block.embedded_points, grid_search_n=5)
    elif ffd_block.evaluation_map is not None:
        return ffd_block.evaluation_map
    else:
        return None


if __name__ == "__main__":
    pass