# Importing all packages 
import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

import lsdo_geo
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from geometry_functions import setup_geometry
import time
# import pickle


lfs.num_workers=1
save_meshes = True
shutdown_inline = True

'''
Setting up 5 FFD sections:
- left + right wings
- left + right transition regions
- center section
'''

# recorder 
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# file_name = 'bwbv2_no_wingtip_coarse_refined.stp'
file_name = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
file_path = os.getcwd() + '/examples/advanced_examples/blended_wing_body/' 

# importing geometry
# Distribution Statement A: Approved for public release; distribution is unlimited. PA# AFRL-2025-3820.
geometry = lsdo_geo.import_geometry(
    file_path + file_name,
    parallelize=False,
)


oml_indices = [key for key in geometry.functions.keys()]
wing_c_indices = [0,1,8,9]
wing_r_transition_indices = [2,3]
wing_r_indices = [4,5,6,7]
wing_l_transition_indices = [10,11]
wing_l_indices = [12,13,14,15] 

#region Geometry Setup
left_wing_transition = geometry.declare_component(wing_l_transition_indices)
left_wing = geometry.declare_component(wing_l_indices)
right_wing_transition = geometry.declare_component(wing_r_transition_indices)
right_wing = geometry.declare_component(wing_r_indices)
center_wing = geometry.declare_component(wing_c_indices)
oml = geometry.declare_component(oml_indices)

wing_parameterization=15
num_v = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]
wing_refit_bspline = lfs.BSplineSpace(num_parametric_dimensions=2, degree=1, coefficients_shape=(wing_parameterization, num_v))
left_wing_function_set = left_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))
right_wing_function_set = right_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))
for i, function in left_wing_function_set.functions.items():
    geometry.functions[i] = function
    left_wing.functions[i] = function

for i, function in right_wing_function_set.functions.items():
    geometry.functions[i] = function
    right_wing.functions[i] = function
# endregion

centerbody_chord_stretches = csdl.Variable(shape=(3,), value=0., name='centerbody_chord_stretches')
wing_chord_stretching_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.]), name='wing_chord_stretching_b_spline_coefficients')
centerbody_span = csdl.Variable(value=10., name='centerbody_span')
transition_span = csdl.Variable(value=4.891, name='transition_span')
wing_span = csdl.Variable(value=25.852 - 9.891, name='wing_span')
wing_sweep_translation = csdl.Variable(value=0., name='wing_sweep_translation')
centerbody_dihedral_translations = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]), name='centerbody_dihedral_translations')
wing_dihedral_translation_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.]), name='wing_dihedral_translation_b_spline_coefficients')
centerbody_twists = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]), name='centerbody_twists')
wing_twists = csdl.Variable(shape=(4,), value=np.array([0., 0., 0., 0.]), name='wing_twists')
percent_change_in_thickness_dof = csdl.Variable(shape=(8,8), value=0., name='percent_change_in_thickness_dof')
normalized_percent_camber_change_dof = csdl.Variable(shape=(6,8), value=0., name='normalized_percent_camber_change_dof')

geometry_values_dict = {
    'centerbody_chord_stretches': centerbody_chord_stretches,
    'wing_chord_stretching_b_spline_coefficients': wing_chord_stretching_b_spline_coefficients,
    'centerbody_span': centerbody_span,
    'transition_span': transition_span,
    'wing_span': wing_span,
    'wing_sweep_translation': wing_sweep_translation,
    'centerbody_dihedral_translations': centerbody_dihedral_translations,
    'wing_dihedral_translation_b_spline_coefficients': wing_dihedral_translation_b_spline_coefficients,
    'centerbody_twists': centerbody_twists,
    'wing_twists': wing_twists,
    'percent_change_in_thickness_dof': percent_change_in_thickness_dof,
    'normalized_percent_camber_change_dof': normalized_percent_camber_change_dof,
}

centerbody_chord_stretches = geometry_values_dict['centerbody_chord_stretches']
wing_chord_stretching_b_spline_coefficients = geometry_values_dict['wing_chord_stretching_b_spline_coefficients']
centerbody_span = geometry_values_dict['centerbody_span']
transition_span = geometry_values_dict['transition_span']
wing_span = geometry_values_dict['wing_span']
wing_sweep_translation = geometry_values_dict['wing_sweep_translation']
centerbody_dihedral_translations = geometry_values_dict['centerbody_dihedral_translations']
wing_dihedral_translation_b_spline_coefficients = geometry_values_dict['wing_dihedral_translation_b_spline_coefficients']
centerbody_twists = geometry_values_dict['centerbody_twists']
wing_twists = geometry_values_dict['wing_twists']
percent_change_in_thickness_dof = geometry_values_dict['percent_change_in_thickness_dof']
normalized_percent_camber_change_dof = geometry_values_dict['normalized_percent_camber_change_dof']

centerline_LE = np.array([-1.1, 0, 0])
centerline_TE = np.array([30.5, 0, 0])
centerbody_mid_LE_R = np.array([3.899, 2.5, 0])
centerbody_mid_TE_R = np.array([30.5, 2.5, 0.])
C_t_joint_LE_R = np.array([9.813, 5, 0])
C_t_joint_TE_R = np.array([30.5, 5, 0])
transition_mid_LE_R = np.array([16.539, 8.617, 0.841])
transition_mid_TE_R = np.array([24.006, 8.617, 0.860])
wing_root_LE_R = np.array([17.815, 9.891, 1.040])
wing_root_TE_R = np.array([23.815, 9.891, 1.040])
wing_sect1_LE_R = np.array([20.931, 14.33, 1.341])
wing_sect1_TE_R = np.array([26.096, 14.33, 1.377])
wing_sect2_LE_R = np.array([25.903, 21.413, 1.822])
wing_sect2_TE_R = np.array([29.735, 21.413, 1.916])
wing_tip_LE_R = np.array([29.019, 25.852, 2.123])
wing_tip_TE_R = np.array([32.016, 25.852, 2.254])


dz_cl = np.array([0, 0, 2.25])
dz_cm = np.array([0, 0, 2.25])
dz_ctj = np.array([0, 0, 1.514025])
dz_tm = np.array([0, 0, 1.514025])
dz_wr = np.array([0, 0, 0.5])
dz_ws1 = np.array([0, 0, 0.5])
dz_ws2 = np.array([0, 0, 0.5])
dz_wt = np.array([0, 0, 0.3])

x_val_half = np.array([
    [centerline_LE[0], centerline_TE[0]+1.],
    [centerbody_mid_LE_R[0]-1, centerbody_mid_TE_R[0]+1],
    [C_t_joint_LE_R[0]-0.5, C_t_joint_TE_R[0]+1.],
    [transition_mid_LE_R[0]-0.5, transition_mid_TE_R[0]+0.5],
    [wing_root_LE_R[0]-0.5, wing_root_TE_R[0]+0.5],
    [wing_sect1_LE_R[0]-0.5, wing_sect1_TE_R[0]+0.5],
    [wing_sect2_LE_R[0]-0.5, wing_sect2_TE_R[0]+0.5],
    [wing_tip_LE_R[0]-0.5, wing_tip_TE_R[0]+0.5],
])
x_val_full = np.concatenate(
    (x_val_half[1:,:][::-1,:], x_val_half)
)

z_val_half = np.array([
    [centerline_LE[2], centerline_TE[2]],
    [centerbody_mid_LE_R[2], centerbody_mid_TE_R[2]],
    [C_t_joint_LE_R[2], C_t_joint_TE_R[2]],
    [transition_mid_LE_R[2], transition_mid_TE_R[2]],
    [wing_root_LE_R[2], wing_root_TE_R[2]],
    [wing_sect1_LE_R[2], wing_sect1_TE_R[2]],
    [wing_sect2_LE_R[2], wing_sect2_TE_R[2]],
    [wing_tip_LE_R[2], wing_tip_TE_R[2]],
])
z_delta_half = np.array([
    [-dz_cl[2]-0.5,dz_cl[2]+0.5],  
    [-dz_cm[2]-0.5,dz_cm[2]+0.5],  
    [-dz_ctj[2]-0.2,dz_ctj[2]+0.2],  
    [-dz_tm[2]-0.2,dz_tm[2]+0.2],  
    [-dz_wr[2]-0.2,dz_wr[2]+0.2],  
    [-dz_ws1[2]-0.2,dz_ws1[2]+0.2],  
    [-dz_ws2[2]-0.2,dz_ws2[2]+0.2],  
    [-dz_wt[2]-0.2,dz_wt[2]+0.2], 
])

z_val_full = np.concatenate(
    (z_val_half[1:,:][::-1,:] + z_delta_half[1:,:][::-1,:], z_val_half + z_delta_half)
)

# y_half = np.array([0, 5, 9.891, 25.852 + 0.25])
y_half = np.array([0, 2.5, 5, 8.617, 9.891, 14.33, 21.413, 25.852 + 0.25])
# y_half = np.array([0, 25.852])
y_full = np.concatenate((-y_half[1:][::-1], y_half))
num_ffd_sections = x_val_full.shape[0]
FFD_block_points = np.zeros((2,num_ffd_sections,2,3))
for i in range(num_ffd_sections):
    x_val = x_val_full[i]
    z_val = z_val_full[i]
    x_grid, z_grid = np.meshgrid(x_val, z_val, indexing='ij')
    y_val = y_full[i]

    cross_section = np.zeros((2,2,3))
    cross_section[:,:,1] = y_val
    cross_section[:,:,0] = x_grid
    cross_section[:,:,2] = z_grid

    FFD_block_points[:,i,:] = cross_section

num_ffd_coefficients_chordwise = 8

BWB_ffd_block = lsdo_geo.construct_ffd_block_from_corners(
    entities=geometry,
    corners=FFD_block_points,
    num_coefficients=(num_ffd_coefficients_chordwise,2,2),
    degree=(2,2,1)
)
# BWB_ffd_block.plot()
center_section_index = num_ffd_sections // 2
num_centerbody_ffd_sections = 3
num_wing_ffd_sections = 4

ffd_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(
    name="ffd_sectional_parameterization",
    parameterized_points=BWB_ffd_block.coefficients,
    principal_parametric_dimension=1,
)
# ffd_sectional_parameterization.plot()

space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

wing_chord_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                        coefficients=wing_chord_stretching_b_spline_coefficients, name='wing_chord_stretching_b_spline')

centerbody_span_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='centerbody_span_stretching_b_spline')

wing_span_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wing_span_stretching_b_spline')

wing_sweep_translation_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wing_sweep_translation_b_spline')

wing_dihedral_translation_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                            coefficients=wing_dihedral_translation_b_spline_coefficients, name='dihedral_translation_b_spline')

# twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
#                                 coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])*np.pi/180), name='twist_b_spline')

# Extract DOFs
# centerbody_chord_stretches = csdl.Variable(shape=(num_centerbody_ffd_sections,), value=0., name='centerbody_chord_stretches')

# centerbody_span = csdl.Variable(value=10., name='centerbody_span')
# transition_span = csdl.Variable(value=4.891, name='transition_span')
# wing_span = csdl.Variable(value=25.852 - 9.891, name='wing_span')

initial_centerbody_span = 10.
initial_transition_span = 4.891
initial_wing_span = 25.852 - 9.891
centerbody_outer_span_translation = (centerbody_span - initial_centerbody_span)/2   # Divide by 2 because halfspan
centerbody_span_stretching_b_spline.coefficients = centerbody_span_stretching_b_spline.coefficients.set(
    csdl.slice[1], centerbody_outer_span_translation
)
transition_outer_span_translation = centerbody_outer_span_translation + (transition_span - initial_transition_span)
wing_span_stretching_b_spline.coefficients = wing_span_stretching_b_spline.coefficients.set(
    csdl.slice[0], transition_outer_span_translation
)
wing_span_stretching_b_spline.coefficients = wing_span_stretching_b_spline.coefficients.set(
    csdl.slice[1], transition_outer_span_translation + (wing_span - initial_wing_span)
)

# wing_sweep_translation = csdl.Variable(value=0., name='wing_sweep_translation')
wing_sweep_translation_b_spline.coefficients = wing_sweep_translation_b_spline.coefficients.set(
    csdl.slice[1], wing_sweep_translation
)

# endregion Create Parameterization Objects

# region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))

chord_stretches = csdl.Variable(shape=(num_ffd_sections,), value=0.)
wing_chord_stretches = wing_chord_stretching_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
wing_translation_stretch = 0.9*wing_chord_stretches[0] + 0.1*centerbody_chord_stretches[-1]
chord_stretches = chord_stretches.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                        centerbody_chord_stretches)
chord_stretches = chord_stretches.set(csdl.slice[-num_wing_ffd_sections:], wing_chord_stretches)
chord_stretches = chord_stretches.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                        centerbody_chord_stretches[::-1])
chord_stretches = chord_stretches.set(csdl.slice[:num_wing_ffd_sections],
                                        wing_chord_stretches[::-1])
chord_stretches = chord_stretches.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                        wing_translation_stretch)
chord_stretches = chord_stretches.set(csdl.slice[num_wing_ffd_sections],
                                        wing_translation_stretch)

centerbody_span_translations = centerbody_span_stretching_b_spline.evaluate(np.linspace(0., 1., num_centerbody_ffd_sections))
wing_span_translations = wing_span_stretching_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
wing_transition_section_translation = 1.1*wing_span_translations[0] + (-0.1)*wing_span_translations[-1]

span_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
span_translations = span_translations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                            centerbody_span_translations)
span_translations = span_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_span_translations)

span_translations = span_translations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                            -centerbody_span_translations[::-1])
span_translations = span_translations.set(csdl.slice[:num_wing_ffd_sections],
                                            -wing_span_translations[::-1])

span_translations = span_translations.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                            wing_transition_section_translation)
span_translations = span_translations.set(csdl.slice[(num_wing_ffd_sections)],
                                            -wing_transition_section_translation)

wing_sweep_translations = wing_sweep_translation_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
transition_sweep_translation = -0.1*wing_sweep_translations[-1]

sweep_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
sweep_translations = sweep_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_sweep_translations)
sweep_translations = sweep_translations.set(csdl.slice[:num_wing_ffd_sections], wing_sweep_translations[::-1])

sweep_translations = sweep_translations.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                            transition_sweep_translation)
sweep_translations = sweep_translations.set(csdl.slice[num_wing_ffd_sections],
                                            transition_sweep_translation)

wing_dihedral_translations = wing_dihedral_translation_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
transition_dihedral = 0.9*wing_dihedral_translations[0] + 0.1*centerbody_dihedral_translations[-1]
dihedral_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
dihedral_translations = dihedral_translations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                                    centerbody_dihedral_translations)
dihedral_translations = dihedral_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_dihedral_translations)
dihedral_translations = dihedral_translations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                                    centerbody_dihedral_translations[::-1])
dihedral_translations = dihedral_translations.set(csdl.slice[:num_wing_ffd_sections], wing_dihedral_translations[::-1])
dihedral_translations = dihedral_translations.set(csdl.slice[-(num_wing_ffd_sections+1)], transition_dihedral)
dihedral_translations = dihedral_translations.set(csdl.slice[num_wing_ffd_sections], transition_dihedral)

transition_twist = 0.9*wing_twists[0] + 0.1*centerbody_twists[-1]

twist_rotations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
twist_rotations = twist_rotations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections], centerbody_twists)
twist_rotations = twist_rotations.set(csdl.slice[-num_wing_ffd_sections:], wing_twists)
twist_rotations = twist_rotations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1], centerbody_twists[::-1])
twist_rotations = twist_rotations.set(csdl.slice[:num_wing_ffd_sections], wing_twists[::-1])
twist_rotations = twist_rotations.set(csdl.slice[-(num_wing_ffd_sections+1)], transition_twist)
twist_rotations = twist_rotations.set(csdl.slice[num_wing_ffd_sections], transition_twist)

sectional_parameters = lsdo_geo.VolumeSectionalParameterizationInputs()
sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretches)
sectional_parameters.add_sectional_translation(axis=csdl.Variable(value=np.array([0., 1., 0.])), translation=span_translations)
sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translations)
sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_translations)
sectional_parameters.add_sectional_rotation(axis=csdl.Variable(value=np.array([0., 1., 0.])), rotation=twist_rotations)

ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)

# Apply shape variables
original_block_thickness = BWB_ffd_block.coefficients.value[:, :, 1, 2] - BWB_ffd_block.coefficients.value[:, :, 0, 2]

percent_change_in_thickness = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections), value=0.)
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,:num_ffd_sections//2+1], percent_change_in_thickness_dof)
percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,num_ffd_sections//2+1:], percent_change_in_thickness_dof[:,-2::-1])
delta_block_thickness = (percent_change_in_thickness / 100) * original_block_thickness
thickness_upper_translation = 1/2 * delta_block_thickness
thickness_lower_translation = -thickness_upper_translation
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,1,2], ffd_coefficients[:,:,1,2] + thickness_upper_translation)
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,0,2], ffd_coefficients[:,:,0,2] + thickness_lower_translation)

# Parameterize camber change as normalized by the original block (kind of like chord) length
block_length = BWB_ffd_block.coefficients.value[1, :, 0, 0] - BWB_ffd_block.coefficients.value[0, :, 0, 0]
block_length = csdl.expand(block_length, (num_ffd_coefficients_chordwise, num_ffd_sections), 'j->ij')

normalized_percent_camber_change = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections), value=0.)
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,:num_ffd_sections//2+1],
                                                                        normalized_percent_camber_change_dof)
normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,num_ffd_sections//2+1:], 
                                                                        normalized_percent_camber_change_dof[:,-2::-1])
camber_change = (normalized_percent_camber_change / 100) * block_length
ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,:,2], 
                                        ffd_coefficients[:,:,:,2] + 
                                        csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk'))

geometry_coefficients = BWB_ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
geometry.set_coefficients(geometry_coefficients) # type: ignore

# region Generate Parameterization Video
jax_inputs = [centerbody_chord_stretches, wing_chord_stretching_b_spline_coefficients, centerbody_span, transition_span, wing_span, 
                wing_sweep_translation, centerbody_dihedral_translations, wing_dihedral_translation_b_spline_coefficients,
                centerbody_twists, wing_twists, percent_change_in_thickness_dof, normalized_percent_camber_change_dof]
# jax outputs is a list containing all the geometry coefficients (geometry.functions[:].coefficients)
jax_outputs = [geometry_function.coefficients for geometry_function in geometry.functions.values()] \
            + [BWB_ffd_block.coefficients]

recorder = csdl.get_current_recorder()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=jax_inputs,
    additional_outputs=jax_outputs,
    gpu=False
)

jax_sim.run()

# Geometry Variables Video
import vedo

video = vedo.Video(name="examples/advanced_examples/blended_wing_body/bwb_no_solver_geometric_variables.mp4", fps=11, backend='opencv')

parameters_with_offsets = [
    (centerbody_chord_stretches, 10.),
    (wing_chord_stretching_b_spline_coefficients, 3.),
    (centerbody_span, centerbody_span.value), 
    (transition_span, transition_span.value), 
    (wing_span, wing_span.value*0.3),
    (wing_sweep_translation, 10.),
    (centerbody_dihedral_translations, 3.),
    (wing_dihedral_translation_b_spline_coefficients, 3.),
    (centerbody_twists, 30*np.pi/180),
    (wing_twists, 30*np.pi/180), # 21 dv not including shape variables
    # (percent_change_in_thickness_dof, 100.), # 64 of these
    # (normalized_percent_camber_change_dof, 100.)    # 48 of these
]

camera = {
    # 'pos': (4*num_bodies + 1, 0, -1*num_bodies/2),
    'pos': (-50, -50, 50),
    # 'pos': (5*num_bodies + 1, 0, 0),
    'focalPoint': (10, -10, 0),
    # 'focalPoint': (0, 0, 0),
    'viewup': (0, 0, 1),
}

for parameter, offset in parameters_with_offsets:
    if len(parameter.value.shape) == 1:
        for i in range(len(parameter.value)):
            values = np.hstack((np.linspace(parameter.value[i], parameter.value[i] + offset, 11).flatten(),
                            np.linspace(parameter.value[i] + offset, parameter.value[i], 11).flatten()))
            for value in values:
                parameter_value = parameter.value
                parameter_value[i] = value
                jax_sim[parameter] = parameter_value
                jax_sim.run()
                frame = geometry.plot(show=False)
                ffd_block_frame = BWB_ffd_block.plot(show=False, plot_embedded_points=False)
                if len(parameter.value) == 1:
                    text = f"Parameter: {parameter.name}, Value: {value:.2f}"
                else:
                    text = f"Parameter: {parameter.name}[{i}], Value: {value:.2f}"
                vedo_text = vedo.Text2D(text, pos="bottom-left", s=2, c='black')
                video_plotter = vedo.Plotter(offscreen=True, title="BWB Geometry Variables",
                                                size=(1920, 1200))
                # video_plotter.show(frame, viewup='z')
                video_plotter.show(frame + ffd_block_frame + [vedo_text], camera=camera)
                video.add_frame()
    elif len(parameter.value.shape) == 2:
        for i in range(parameter.value.shape[0]):
            for j in range(parameter.value.shape[1]):
                values = np.hstack((np.linspace(parameter.value[i, j], parameter.value[i, j] + offset, 11).flatten(),
                                np.linspace(parameter.value[i, j] + offset, parameter.value[i, j], 11).flatten()))
                for value in values:
                    parameter_value = parameter.value
                    parameter_value[i, j] = value
                    jax_sim[parameter] = parameter_value
                    jax_sim.run()
                    frame = geometry.plot(show=False)
                    ffd_block_frame = BWB_ffd_block.plot(show=False, plot_embedded_points=False)
                    text = f"Parameter: {parameter.name}[{i,j}], Value: {value:.2f}"
                    vedo_text = vedo.Text2D(text, pos="bottom-left", s=2, c='black')
                    video_plotter = vedo.Plotter(offscreen=True, title="BWB Geometry Variables",
                                                    size=(1920, 1200))
                    # video_plotter.show(frame, viewup='z')
                    video_plotter.show(frame + ffd_block_frame + [vedo_text], camera=camera)
                    video.add_frame()

video.close()
# endregion Generate Parameterization Video