import numpy as np
import imageio.v2 as imageio
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo

recorder = csdl.Recorder(inline=True)
recorder.start()

# region imports and constructions
# import wing
imported_wing = lsdo_geo.import_geometry("examples/example_geometries/rectangular_wing.stp", parallelize=False)
# imported_wing.functions[3].plot(point_types=["coefficients"], plot_types=['point_cloud'])
wing_new_function_space = lfs.BSplineSpace(
    num_parametric_dimensions=2,
    degree=(2,3),
    coefficients_shape=(3,imported_wing.functions[3].coefficients.shape[1]),
    knots = (np.array([0., 0., 0., 1., 1., 1.]), imported_wing.functions[3].space.knots[1])
    )
function_2_new_coefficients = np.linspace(imported_wing.functions[2].coefficients[0].value, imported_wing.functions[2].coefficients[-1].value, wing_new_function_space.coefficients_shape[0])
function_3_new_coefficients = np.linspace(imported_wing.functions[3].coefficients[0].value, imported_wing.functions[3].coefficients[-1].value, wing_new_function_space.coefficients_shape[0])
imported_wing.functions[2] = lfs.Function(wing_new_function_space, function_2_new_coefficients)
imported_wing.functions[3] = lfs.Function(wing_new_function_space, function_3_new_coefficients)
# imported_wing.functions[2].plot(point_types=["coefficients"], plot_types=['point_cloud'])
# imported_wing.functions[3].plot(point_types=["coefficients"], plot_types=['point_cloud'])
# imported_wing.plot()
# exit()

# construct fuselage
num_control_points_top = 4
num_control_points_corner = 1
num_control_points_side_top = 2

fuselage_b_spline_space = lfs.BSplineSpace(
    num_parametric_dimensions=2,
    degree=(1,4),
    coefficients_shape=(2, 2*(num_control_points_top + num_control_points_corner + num_control_points_side_top) -1)
    )

fuselage_top = 0.5
fuselage_side = 0.5
fuselage_front_x_coordinate = 0.
fuselage_back_x_coordinate = 3.
normalized_fuselage_radius = 0.5
radius_variable = normalized_fuselage_radius*fuselage_side
fuselage_top_control_points = np.zeros((2, num_control_points_top, 3))
fuselage_corner_control_point = np.array([
    [[fuselage_front_x_coordinate, fuselage_side, fuselage_top]],
    [[fuselage_back_x_coordinate, fuselage_side, fuselage_top]]
])
fuselage_side_top_control_points = np.zeros((2, num_control_points_side_top, 3))
fuselage_top_control_points[:,:,2] = fuselage_top
fuselage_top_control_points[:,-1, 1] = fuselage_side - radius_variable
fuselage_top_control_points[1,:,0] = fuselage_back_x_coordinate
fuselage_side_top_control_points[:,:, 1] = fuselage_side
fuselage_side_top_control_points[:,0, 2] = fuselage_top - radius_variable
fuselage_side_top_control_points[:,1:, 2] = np.linspace(0., fuselage_top - radius_variable, num_control_points_side_top-1, endpoint=False)[::-1]
fuselage_side_top_control_points[1,:,0] = fuselage_back_x_coordinate

fuselage_side_bottom_control_points = fuselage_side_top_control_points.copy()[:,:-1]
fuselage_side_bottom_control_points[:,:, 2] = -fuselage_side_top_control_points[:,:-1,2]
fuselage_bottom_corner_control_points = fuselage_corner_control_point.copy()
fuselage_bottom_corner_control_points[:,:, 2] = -fuselage_bottom_corner_control_points[:,:, 2]
fuselage_bottom_control_points = fuselage_top_control_points.copy()[:,::-1,:]
fuselage_bottom_control_points[:,:,2] = -fuselage_top

fuselage_half_control_points = np.concatenate((fuselage_top_control_points, fuselage_corner_control_point, fuselage_side_top_control_points,
                                               fuselage_side_bottom_control_points, fuselage_bottom_corner_control_points, fuselage_bottom_control_points
                                               ), axis=1)
fuselage_half_surface = lfs.Function(fuselage_b_spline_space, fuselage_half_control_points)
# fuselage_half_surface.plot()
fuselage_other_half_surface = fuselage_half_surface.copy()
fuselage_other_half_surface.coefficients = fuselage_half_surface.coefficients.set(csdl.slice[:,:,1],
                                                                                  -fuselage_half_surface.coefficients[:, :, 1])
# fuselage_other_half_surface.plot()
# fuselage = lsdo_geo.Geometry(functions={0 : fuselage_half_surface, 1 : fuselage_other_half_surface})
# fuselage.plot()

nose_cone_length = 1.
normalized_nose_cone_smoothing = 0.
nose_cone_front_point = np.array([fuselage_front_x_coordinate - 2*nose_cone_length, 0., 0.])
nose_cone_control_points = np.zeros((5, fuselage_half_surface.coefficients.shape[1], 3))
nose_cone_control_points[0] = fuselage_half_surface.coefficients.value[0]
nose_cone_control_points[1] = fuselage_half_surface.coefficients.value[0]
nose_cone_control_points[1,:,0] = fuselage_front_x_coordinate - normalized_nose_cone_smoothing * 2*nose_cone_length
nose_cone_control_points[2] = nose_cone_front_point
nose_cone_control_points[3] = fuselage_other_half_surface.coefficients.value[0]
nose_cone_control_points[3,:,0] = nose_cone_control_points[1,:,0]
nose_cone_control_points[4] = fuselage_other_half_surface.coefficients.value[0]
nose_cone_b_spline_space = lfs.BSplineSpace(
    num_parametric_dimensions=2,
    degree=(2,4),
    coefficients_shape=(5, fuselage_half_surface.coefficients.shape[1], 3)
)
nose_cone_surface = lfs.Function(nose_cone_b_spline_space, nose_cone_control_points)
# nose_cone_surface.plot()
# fuselage = lsdo_geo.Geometry(functions={0 : fuselage_half_surface, 1 : fuselage_other_half_surface, 2 : nose_cone_surface})
# fuselage.plot()

tail_cone_length = 2.
normalized_tail_cone_smoothing = 0.
tail_cone_back_point = np.array([fuselage_back_x_coordinate + 2*tail_cone_length, 0., 0.])
tail_cone_control_points = np.zeros((5, fuselage_half_surface.coefficients.shape[1], 3))
tail_cone_control_points[0] = fuselage_half_surface.coefficients.value[-1]
tail_cone_control_points[1] = fuselage_half_surface.coefficients.value[-1]
tail_cone_control_points[1,:,0] = fuselage_back_x_coordinate + normalized_tail_cone_smoothing * 2*tail_cone_length
tail_cone_control_points[2] = tail_cone_back_point
tail_cone_control_points[3] = fuselage_other_half_surface.coefficients.value[-1]
tail_cone_control_points[3,:,0] = tail_cone_control_points[1,:,0]
tail_cone_control_points[4] = fuselage_other_half_surface.coefficients.value[-1]
tail_cone_b_spline_space = lfs.BSplineSpace(
    num_parametric_dimensions=2,
    degree=(2,4),
    coefficients_shape=(5, fuselage_half_surface.coefficients.shape[1], 3)
)
tail_cone_surface = lfs.Function(tail_cone_b_spline_space, tail_cone_control_points)
# tail_cone_surface.plot()
# fuselage = lsdo_geo.Geometry(functions={0 : fuselage_half_surface, 1 : fuselage_other_half_surface, 2 : nose_cone_surface, 3 : tail_cone_surface})
# fuselage.plot()

half_wing = imported_wing.copy().create_subset(function_indices=[i for i in range(6)], name="half_wing")
half_wing_geo = lsdo_geo.Geometry(functions=half_wing.functions)
half_wing_geo.translate(np.array([0., fuselage_side, 0.]))

other_half_wing_geo = half_wing_geo.copy()
for function in other_half_wing_geo.functions.values():
    function.coefficients = function.coefficients.set(csdl.slice[:,:,1], -function.coefficients[:,:,1])
# half_wing.plot()

tail = imported_wing.copy()
tail_geo = lsdo_geo.Geometry(functions=tail.functions)
tail_geo.translate(np.array([fuselage_back_x_coordinate+tail_cone_length, 0., 0.]))
tail_scale_factor = 0.025
for function in tail_geo.functions.values():
    function.coefficients = function.coefficients * tail_scale_factor
tail_chord_scale_factor = 2.
for function in tail_geo.functions.values():
    function.coefficients = function.coefficients.set(csdl.slice[:,:,0], function.coefficients[:,:,0]*tail_chord_scale_factor)

vtail = half_wing_geo.copy()
vtail.translate(np.array([fuselage_back_x_coordinate+tail_cone_length, -fuselage_side, 0.]))
vtail.rotate(np.array([0., 0., 0.]), np.array([1., 0., 0.]), np.pi/2)
vtail_scale_factor = 0.025
vtail_chord_scale_factor = 2.
for function in vtail.functions.values():
    function.coefficients = function.coefficients * vtail_scale_factor
for function in vtail.functions.values():
    function.coefficients = function.coefficients.set(csdl.slice[:,:,0], function.coefficients[:,:,0]*vtail_chord_scale_factor)

geometry_functions = {}
geometry_functions[0] = fuselage_half_surface
geometry_functions[1] = fuselage_other_half_surface
geometry_functions[2] = nose_cone_surface
geometry_functions[3] = tail_cone_surface
for i, function in enumerate(half_wing_geo.functions.values()):
    geometry_functions[4+i] = function
for i in range(len(half_wing_geo.functions)):
    geometry_functions[4+len(half_wing_geo.functions)+i] = other_half_wing_geo.functions[i]
for i, function in enumerate(tail_geo.functions.values()):
    geometry_functions[4+2*len(half_wing_geo.functions)+i] = function
for i, function in enumerate(vtail.functions.values()):
    geometry_functions[4+2*len(half_wing_geo.functions)+len(tail_geo.functions)+i] = function
geometry = lsdo_geo.Geometry(functions=geometry_functions)
# geometry.plot()

fuselage = geometry.declare_component(function_indices=[0,1,2,3], name="fuselage")
wing = geometry.declare_component(function_indices=[i for i in range(4, 4+2*len(half_wing_geo.functions))], name="wing")
tail = geometry.declare_component(function_indices=[i for i in range(4+2*len(half_wing_geo.functions), 4+2*len(half_wing_geo.functions)+len(tail_geo.functions))], name="tail")
vtail = geometry.declare_component(function_indices=[i for i in range(4+2*len(half_wing_geo.functions)+len(tail_geo.functions), 4+2*len(half_wing_geo.functions)+len(tail_geo.functions)+len(vtail.functions))], name="vtail")

half_wing = geometry.declare_component(function_indices=[i for i in range(4, 4+len(half_wing_geo.functions))], name="half_wing")
other_half_wing = geometry.declare_component(function_indices=[i for i in range(4+len(half_wing_geo.functions), 4+2*len(half_wing_geo.functions))], name="other_half_wing")
nose_cone = geometry.declare_component(function_indices=[2], name="nose_cone")
tail_cone = geometry.declare_component(function_indices=[3], name="tail_cone")

# endregion imports and constructions

# region geometry parameterization
parameterization_solver = lsdo_geo.ParameterizationSolver()

fuselage_body_length = csdl.Variable(value=0.2)
fuselage_width = csdl.Variable(value=0.03)
fuselage_height = csdl.Variable(value=0.03)
nose_cone_length = csdl.Variable(value=0.05)
tail_cone_length = csdl.Variable(value=0.1)
normalized_fuselage_fillet_radius = csdl.Variable(value=0.5)
normalized_nose_cone_smoothing = csdl.Variable(value=0.5)
normalized_tail_cone_smoothing = csdl.Variable(value=0.5)

wing_sweep = csdl.Variable(value=0.)
wing_span = csdl.Variable(value=0.3)
wing_dihedral = csdl.Variable(value=0.)
wing_root_chord = csdl.Variable(value=0.05)
wing_mid_chord_relative_and_normalized = csdl.Variable(value=0.8)
wing_tip_chord_relative_and_normalized = csdl.Variable(value=0.5)
wing_thickness_to_chord_ratio = csdl.Variable(value=0.12)


# region fuselage parameterization
fuselage_width_stretch = csdl.Variable(value=0.)
fuselage_height_stretch = csdl.Variable(value=0.)
fuselage_length_stretch = csdl.Variable(value=0.)
nose_cone_length_stretch = csdl.Variable(value=0.)
tail_cone_length_stretch = csdl.Variable(value=0.)

fuselage_ffd_block = lsdo_geo.construct_ffd_block_around_entities(fuselage, num_coefficients=(2,2,2), degree=1)
fuselage_sectional_parameterization = lsdo_geo.SectionalParameterization(parameterized_points=fuselage_ffd_block.coefficients,
                                                                         principal_parametric_dimension=1)
fuselage_sectional_parameters = lsdo_geo.SectionalParameters()
#     translations=[(np.array([0., 1., 0.]), csdl.concatenate((-fuselage_width_stretch, fuselage_width_stretch)))],
#     stretches=[(np.array([1., 0., 0.]), csdl.concatenate((fuselage_length_stretch, fuselage_length_stretch))),
#                (np.array([0., 0., 1.]), csdl.concatenate((fuselage_height_stretch, fuselage_height_stretch)))]
# )
fuselage_sectional_parameters.add_translation(np.array([0., 1., 0.]), csdl.concatenate((-fuselage_width_stretch, fuselage_width_stretch)))
fuselage_sectional_parameters.add_stretch(np.array([1., 0., 0.]), csdl.concatenate((fuselage_length_stretch, fuselage_length_stretch)))
fuselage_sectional_parameters.add_stretch(np.array([0., 0., 1.]), csdl.concatenate((fuselage_height_stretch, fuselage_height_stretch)))

fuselage_ffd_coefficients = fuselage_sectional_parameterization.evaluate(fuselage_sectional_parameters, plot=False)
fuselage_ffd_block.coefficients = fuselage_ffd_coefficients
# fuselage_ffd_block.plot()
fuselage_coefficients = fuselage_ffd_block.evaluate_ffd(coefficients=fuselage_ffd_coefficients)
fuselage.set_coefficients(fuselage_coefficients)

fuselage_half_coefficients = fuselage.functions[0].coefficients
sharpness_scaling = 1/(fuselage_half_coefficients[:, num_control_points_top, 1] + fuselage_half_coefficients[:, num_control_points_top, 2])/2
max_fillet_size = 1/sharpness_scaling*csdl.minimum(sharpness_scaling*fuselage_half_coefficients[:, num_control_points_top, 1],
                             sharpness_scaling*fuselage_half_coefficients[:, num_control_points_top, 2])
fuselage_half_coefficients = fuselage_half_coefficients.set(
    csdl.slice[:, num_control_points_top-1, 1],
    fuselage_half_coefficients[:, num_control_points_top, 1] - normalized_fuselage_fillet_radius*max_fillet_size,
)
fuselage_half_coefficients = fuselage_half_coefficients.set(
    csdl.slice[:, num_control_points_top+1, 2],
    fuselage_half_coefficients[:, num_control_points_top, 2] - normalized_fuselage_fillet_radius*max_fillet_size,
)
fuselage_half_coefficients = fuselage_half_coefficients.set(
    csdl.slice[:, -(num_control_points_top-1 + 1), 1],
    fuselage_half_coefficients[:, -num_control_points_top - 1, 1] - normalized_fuselage_fillet_radius*max_fillet_size,
)
fuselage_half_coefficients = fuselage_half_coefficients.set(
    csdl.slice[:, -(num_control_points_top+1 + 1), 2],
    fuselage_half_coefficients[:, -num_control_points_top - 1, 2] + normalized_fuselage_fillet_radius*max_fillet_size,
)

fuselage.functions[0].coefficients = fuselage_half_coefficients
fuselage.functions[1].coefficients = fuselage_half_coefficients.set(
    csdl.slice[:, :, 1],
    -fuselage_half_coefficients[:, :, 1],
)

nose_cone_front_control_point = fuselage.functions[2].coefficients[2,0]
nose_cone_front_control_point = nose_cone_front_control_point.set(csdl.slice[0], fuselage.functions[2].coefficients[2,0,0] - nose_cone_length_stretch*2)
fuselage_front_x = fuselage.functions[0].coefficients[0, 0, 0]
nose_cone_length_from_control_points = fuselage_front_x - nose_cone_front_control_point[0]
tail_cone_back_control_point = fuselage.functions[3].coefficients[2,0]
tail_cone_back_control_point = tail_cone_back_control_point.set(csdl.slice[0], fuselage.functions[3].coefficients[2,0,0] + tail_cone_length_stretch*2)
fuselage_back_x = fuselage.functions[0].coefficients[-1, 0, 0]
tail_cone_length_from_control_points = tail_cone_back_control_point[0] - fuselage_back_x

fuselage.functions[2].coefficients = fuselage.functions[2].coefficients.set(csdl.slice[0,:,:], fuselage.functions[0].coefficients[0])
fuselage.functions[2].coefficients = fuselage.functions[2].coefficients.set(csdl.slice[1,:,:], fuselage.functions[0].coefficients[0])
fuselage.functions[2].coefficients = fuselage.functions[2].coefficients.set(csdl.slice[1,:,0], fuselage_front_x - normalized_nose_cone_smoothing * nose_cone_length_from_control_points)
fuselage.functions[2].coefficients = fuselage.functions[2].coefficients.set(csdl.slice[3,:,:], fuselage.functions[1].coefficients[0])
fuselage.functions[2].coefficients = fuselage.functions[2].coefficients.set(csdl.slice[3,:,0], fuselage.functions[2].coefficients[1,:,0])
fuselage.functions[2].coefficients = fuselage.functions[2].coefficients.set(csdl.slice[4,:,:], fuselage.functions[1].coefficients[0])
fuselage.functions[3].coefficients = fuselage.functions[3].coefficients.set(csdl.slice[0,:,:], fuselage.functions[0].coefficients[1])
fuselage.functions[3].coefficients = fuselage.functions[3].coefficients.set(csdl.slice[1,:,:], fuselage.functions[0].coefficients[1])
fuselage.functions[3].coefficients = fuselage.functions[3].coefficients.set(csdl.slice[1,:,0], fuselage_back_x + normalized_tail_cone_smoothing * tail_cone_length_from_control_points)
fuselage.functions[3].coefficients = fuselage.functions[3].coefficients.set(csdl.slice[3,:,:], fuselage.functions[1].coefficients[1])
fuselage.functions[3].coefficients = fuselage.functions[3].coefficients.set(csdl.slice[3,:,0], fuselage.functions[3].coefficients[1,:,0])
fuselage.functions[3].coefficients = fuselage.functions[3].coefficients.set(csdl.slice[4,:,:], fuselage.functions[1].coefficients[1])

fuselage.functions[2].coefficients = fuselage.functions[2].coefficients.set(csdl.slice[2,:,0], fuselage.functions[2].coefficients[2,:,0] - 2*nose_cone_length_stretch)
fuselage.functions[3].coefficients = fuselage.functions[3].coefficients.set(csdl.slice[2,:,0], fuselage.functions[3].coefficients[2,:,0] + 2*tail_cone_length_stretch)

fuselage_rigid_body_translation_x = csdl.Variable(value=0.)
fuselage.translate(csdl.concatenate((fuselage_rigid_body_translation_x, csdl.Variable(value=np.zeros(2)))))

# endregion fuselage parameterization


# region wing parameterization
wing_sweep_translation = csdl.Variable(value=0.)
wing_span_stretch = csdl.Variable(value=0.)
wing_dihedral_translation = csdl.Variable(value=0.)

wing_chord_stretches = csdl.Variable(value=np.array([0., 0., 0.]))
wing_thickness_stretches = csdl.Variable(value=np.array([0., 0., 0.]))

wing_twist_rotations = csdl.Variable(value=np.array([0., 0., 0.]))


half_wing_ffd_block = lsdo_geo.construct_ffd_block_around_entities(half_wing, num_coefficients=(2, 3, 2), degree=(1,2,1))
half_wing_sectional_parameterization = lsdo_geo.SectionalParameterization(parameterized_points=half_wing_ffd_block.coefficients,
                                                                         principal_parametric_dimension=1)
half_wing_sectional_parameters = lsdo_geo.SectionalParameters()
half_wing_sectional_parameters.add_translation(np.array([1., 0., 0.]), csdl.concatenate((csdl.Variable(value=0.), wing_sweep_translation/2, wing_sweep_translation)))
half_wing_sectional_parameters.add_translation(np.array([0., 1., 0.]), csdl.concatenate((csdl.Variable(value=0.), wing_span_stretch/2, wing_span_stretch)))
half_wing_sectional_parameters.add_translation(np.array([0., 0., 1.]), csdl.concatenate((csdl.Variable(value=0.), wing_dihedral_translation/2, wing_dihedral_translation)))
half_wing_sectional_parameters.add_stretch(np.array([1., 0., 0.]), wing_chord_stretches)
half_wing_sectional_parameters.add_stretch(np.array([0., 0., 1.]), wing_thickness_stretches)
half_wing_sectional_parameters.add_rotation(np.array([0., 0., 1.]), wing_twist_rotations)

half_wing_ffd_coefficients = half_wing_sectional_parameterization.evaluate(half_wing_sectional_parameters, plot=False)
half_wing_coefficients = half_wing_ffd_block.evaluate_ffd(coefficients=half_wing_ffd_coefficients)
half_wing.set_coefficients(half_wing_coefficients)

wing_rigid_body_translation = csdl.Variable(value=np.zeros(3))
half_wing.translate(wing_rigid_body_translation)

for i, function in other_half_wing.functions.items():
    function.coefficients = half_wing.functions[i-len(half_wing.functions)].coefficients.set(csdl.slice[:,:,1], -half_wing.functions[i-len(half_wing.functions)].coefficients[:,:,1])

# endregion wing parameterization



# # region tail parameterization
# tail_sweep_translation = csdl.Variable(value=0.)
# tail_span_stretch = csdl.Variable(value=0.)
# tail_dihedral_translation = csdl.Variable(value=0.)

# tail_chord_stretches = csdl.Variable(value=np.array([0., 0., 0.]))
# tail_thickness_stretches = csdl.Variable(value=np.array([0., 0., 0.]))

# tail_twist_rotations = csdl.Variable(value=np.array([0., 0., 0.]))


# half_tail_ffd_block = lsdo_geo.construct_ffd_block_around_entities(half_tail, num_coefficients=(2, 3, 2), degree=(1,2,1))
# half_tail_sectional_parameterization = lsdo_geo.SectionalParameterization(parameterized_points=half_tail_ffd_block.coefficients,
#                                                                          principal_parametric_dimension=1)
# half_tail_sectional_parameters = lsdo_geo.SectionalParameters()
# half_tail_sectional_parameters.add_translation(np.array([1., 0., 0.]), csdl.concatenate((csdl.Variable(value=0.), tail_sweep_translation/2, tail_sweep_translation)))
# half_tail_sectional_parameters.add_translation(np.array([0., 1., 0.]), csdl.concatenate((csdl.Variable(value=0.), tail_span_stretch/2, tail_span_stretch)))
# half_tail_sectional_parameters.add_translation(np.array([0., 0., 1.]), csdl.concatenate((csdl.Variable(value=0.), tail_dihedral_translation/2, tail_dihedral_translation)))
# half_tail_sectional_parameters.add_stretch(np.array([1., 0., 0.]), tail_chord_stretches)
# half_tail_sectional_parameters.add_stretch(np.array([0., 0., 1.]), tail_thickness_stretches)
# half_tail_sectional_parameters.add_rotation(np.array([0., 0., 1.]), tail_twist_rotations)

# half_tail_ffd_coefficients = half_tail_sectional_parameterization.evaluate(half_tail_sectional_parameters, plot=False)
# half_tail_coefficients = half_tail_ffd_block.evaluate_ffd(coefficients=half_tail_ffd_coefficients)
# half_tail.set_coefficients(half_tail_coefficients)

# tail_rigid_body_translation = csdl.Variable(value=np.zeros(3))
# half_tail.translate(tail_rigid_body_translation)

# for i, function in other_half_tail.functions.items():
#     function.coefficients = half_tail.functions[i-len(half_tail.functions)].coefficients.set(csdl.slice[:,:,1], -half_tail.functions[i-len(half_tail.functions)].coefficients[:,:,1])

# # endregion tail parameterization

# temp tail parameterization is just a rigid body translation for now
tail_rigid_body_translation = csdl.Variable(value=np.zeros(3))
tail.translate(tail_rigid_body_translation)
vtail.translate(tail_rigid_body_translation)



fuselage_right_side = fuselage.functions[0].evaluate(np.array([0.5, 0.5])).flatten()
fuselage_left_side = fuselage.functions[1].evaluate(np.array([0.5, 0.5])).flatten()
fuselage_top = fuselage.functions[0].evaluate(np.array([0.5, 0.])).flatten()
fuselage_bottom = fuselage.functions[0].evaluate(np.array([0.5, 1.])).flatten()
nose_cone_front = fuselage.functions[2].evaluate(np.array([0.5, 0.5])).flatten()
nose_cone_top_back = fuselage.functions[2].evaluate(np.array([0., 0.])).flatten()
tail_cone_top_front = fuselage.functions[3].evaluate(np.array([0., 0.])).flatten()
tail_cone_back = fuselage.functions[3].evaluate(np.array([0.5, 0.5])).flatten()

fuselage_body_length_computed = tail_cone_top_front[0] - nose_cone_top_back[0]
fuselage_width_computed = fuselage_right_side[1] - fuselage_left_side[1]
fuselage_height_computed = fuselage_top[2] - fuselage_bottom[2]
nose_cone_length_computed = nose_cone_top_back[0] - nose_cone_front[0]
tail_cone_length_computed = tail_cone_back[0] - tail_cone_top_front[0]

# for wing, u is along the span and v is along the chord
tip_u = 1.
root_u = 0.
mid_u = 0.5
leading_edge_v = 0.
trailing_edge_v = 1.
quarter_chord_v = 0.6
max_thickness_v_upper = 0.66
max_thickness_v_lower = 0.33
wing_root_leading_edge = half_wing.functions[7].evaluate(np.array([root_u, leading_edge_v]), plot=False).flatten()
wing_root_trailing_edge = half_wing.functions[7].evaluate(np.array([root_u, trailing_edge_v]), plot=False).flatten()
wing_tip_leading_edge = half_wing.functions[7].evaluate(np.array([tip_u, leading_edge_v]), plot=False).flatten()
wing_tip_trailing_edge = half_wing.functions[7].evaluate(np.array([tip_u, trailing_edge_v]), plot=False).flatten()
wing_mid_leading_edge = half_wing.functions[7].evaluate(np.array([mid_u, leading_edge_v]), plot=False).flatten()
wing_mid_trailing_edge = half_wing.functions[7].evaluate(np.array([mid_u, trailing_edge_v]), plot=False).flatten()
wing_root_quarter_chord = half_wing.functions[7].evaluate(np.array([root_u, quarter_chord_v]), plot=False).flatten()
wing_tip_quarter_chord = half_wing.functions[7].evaluate(np.array([tip_u, quarter_chord_v]), plot=False).flatten()
wing_root_max_thickness_location_upper = half_wing.functions[7].evaluate(np.array([root_u, max_thickness_v_upper]), plot=False).flatten()
wing_root_max_thickness_location_lower = half_wing.functions[6].evaluate(np.array([root_u, max_thickness_v_lower]), plot=False).flatten()
wing_tip_max_thickness_location_upper = half_wing.functions[7].evaluate(np.array([tip_u, max_thickness_v_upper]), plot=False).flatten()
wing_tip_max_thickness_location_lower = half_wing.functions[6].evaluate(np.array([tip_u, max_thickness_v_lower]), plot=False).flatten()
wing_mid_max_thickness_location_upper = half_wing.functions[7].evaluate(np.array([mid_u, max_thickness_v_upper]), plot=False).flatten()
wing_mid_max_thickness_location_lower = half_wing.functions[6].evaluate(np.array([mid_u, max_thickness_v_lower]), plot=False).flatten()


wing_sweep_vector_computed = wing_tip_quarter_chord - wing_root_quarter_chord
wing_span_computed = (wing_tip_quarter_chord[1] - wing_root_quarter_chord[1])*2
wing_dihedral_vector_computed = wing_sweep_vector_computed
wing_root_chord_computed = wing_root_trailing_edge[0] - wing_root_leading_edge[0]
wing_tip_chord_computed = wing_tip_trailing_edge[0] - wing_tip_leading_edge[0]
wing_mid_chord_computed = wing_mid_trailing_edge[0] - wing_mid_leading_edge[0]
wing_root_thickness_computed = wing_root_max_thickness_location_upper[2] - wing_root_max_thickness_location_lower[2]
wing_tip_thickness_computed = wing_tip_max_thickness_location_upper[2] - wing_tip_max_thickness_location_lower[2]
wing_mid_thickness_computed = wing_mid_max_thickness_location_upper[2] - wing_mid_max_thickness_location_lower[2]
wing_mid_chord_relative_and_normalized_computed = (wing_mid_chord_computed/wing_root_chord - 0.5)/0.5
wing_mid_chord = (wing_mid_chord_relative_and_normalized/2 + 0.5) * wing_root_chord
max_allowable_wing_tip_chord = (wing_mid_chord - wing_root_chord) * 2 + wing_root_chord
wing_tip_chord_relative_and_normalized_computed = (wing_tip_chord_computed/max_allowable_wing_tip_chord)
wing_tip_chord = wing_tip_chord_relative_and_normalized * max_allowable_wing_tip_chord
wing_root_thickness_to_chord_ratio_computed = wing_root_thickness_computed / wing_root_chord_computed
wing_tip_thickness_to_chord_ratio_computed = wing_tip_thickness_computed / wing_tip_chord_computed
wing_mid_thickness_to_chord_ratio_computed = wing_mid_thickness_computed / wing_mid_chord_computed


tail_root_leading_edge = tail.functions[19].evaluate(np.array([root_u, leading_edge_v]), plot=False).flatten()
# vtail_root_leading_edge = vtail.functions[31].evaluate(np.array([root_u, leading_edge_v]), plot=True).flatten()


fuselage_wing_leading_edge_mount_location = fuselage.functions[0].evaluate(np.array([0.25, 0.5])).flatten()
fuselage_tail_leading_edge_mount_location = fuselage.functions[3].evaluate(np.array([0.5, 0.5])).flatten()
fuselage_tail_leading_edge_mount_location = fuselage_tail_leading_edge_mount_location.set(csdl.slice[0], fuselage_tail_leading_edge_mount_location[0] - 1/10*tail_cone_length)

parameterization_solver.add_state(fuselage_width_stretch)
parameterization_solver.add_state(fuselage_height_stretch)
parameterization_solver.add_state(fuselage_length_stretch)
parameterization_solver.add_state(nose_cone_length_stretch)
parameterization_solver.add_state(tail_cone_length_stretch)
parameterization_solver.add_state(fuselage_rigid_body_translation_x)

parameterization_solver.add_state(wing_sweep_translation)
parameterization_solver.add_state(wing_span_stretch)
parameterization_solver.add_state(wing_dihedral_translation)
parameterization_solver.add_state(wing_chord_stretches)
parameterization_solver.add_state(wing_thickness_stretches)
parameterization_solver.add_state(wing_rigid_body_translation)

parameterization_solver.add_state(tail_rigid_body_translation)

geometric_variables = lsdo_geo.GeometricVariables()
geometric_variables.add_variable(fuselage_body_length_computed, fuselage_body_length)
geometric_variables.add_variable(fuselage_width_computed, fuselage_width)
geometric_variables.add_variable(fuselage_height_computed, fuselage_height)
geometric_variables.add_variable(nose_cone_length_computed, nose_cone_length)
geometric_variables.add_variable(tail_cone_length_computed, tail_cone_length)

geometric_variables.add_variable(wing_sweep_vector_computed[0], csdl.tan(wing_sweep) * wing_sweep_vector_computed[1])
geometric_variables.add_variable(wing_span_computed, wing_span)
geometric_variables.add_variable(wing_dihedral_vector_computed[2], csdl.tan(wing_dihedral) * wing_dihedral_vector_computed[1])
geometric_variables.add_variable(wing_root_chord_computed, wing_root_chord)
geometric_variables.add_variable(wing_tip_chord_computed, wing_tip_chord)
geometric_variables.add_variable(wing_mid_chord_computed, wing_mid_chord)
geometric_variables.add_variable(wing_root_thickness_computed, wing_thickness_to_chord_ratio*wing_root_chord)
geometric_variables.add_variable(wing_tip_thickness_computed, wing_thickness_to_chord_ratio*wing_tip_chord)
geometric_variables.add_variable(wing_mid_thickness_computed, wing_thickness_to_chord_ratio*wing_mid_chord)

parameterization_solver.add_equality_constraint(fuselage_wing_leading_edge_mount_location - wing_root_leading_edge, np.array([0., 0., 0.]))
parameterization_solver.add_equality_constraint(fuselage_tail_leading_edge_mount_location - tail_root_leading_edge, np.array([0., 0., 0.]))
parameterization_solver.add_equality_constraint(wing_root_leading_edge[0], 0.)

parameterization_solver.evaluate(geometric_variables)

geometry.plot()
# exit()
# endregion geometry parameterization


# Define design variable bounds (lower and upper bounds for each variable)
design_variable_bounds = {
    'fuselage_body_length': [fuselage_body_length.value.item()*0.75, fuselage_body_length.value.item()*1.5],
    'fuselage_width': [fuselage_width.value.item()*0.5, fuselage_width.value.item()*1.5],
    'fuselage_height': [fuselage_height.value.item()*0.5, fuselage_height.value.item()*1.5],
    'nose_cone_length': [nose_cone_length.value.item()*0.5, nose_cone_length.value.item()*1.5],
    'tail_cone_length': [tail_cone_length.value.item()*0.5, tail_cone_length.value.item()*1.5],
    'normalized_fuselage_fillet_radius': [0.5, 1.],
    'normalized_nose_cone_smoothing': [0., 1.],
    'normalized_tail_cone_smoothing': [0., 1.],
    'wing_sweep': [0., np.pi/12],
    # 'wing_span': [wing_span.value.item()*0.5, wing_span.value.item()*2],
    'wing_dihedral': [0., np.pi/12],
    'wing_root_chord': [0.03, 0.1],
    'wing_mid_chord_relative_and_normalized': [0.25, 1.],
    'wing_tip_chord_relative_and_normalized': [0.25, 1.],
    'wing_thickness_to_chord_ratio': [0.08, 0.24],
}
sim_inputs = [fuselage_body_length, fuselage_width, fuselage_height, nose_cone_length, tail_cone_length,
              normalized_fuselage_fillet_radius, normalized_nose_cone_smoothing, normalized_tail_cone_smoothing,
              wing_sweep, 
            #   wing_span, 
              wing_dihedral, wing_root_chord, wing_mid_chord_relative_and_normalized, 
              wing_tip_chord_relative_and_normalized, wing_thickness_to_chord_ratio]
sim_outputs = [geometry_function.coefficients for geometry_function in geometry.functions.values()]


sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=sim_inputs,
    additional_outputs=sim_outputs,
    gpu=False
)


from scipy.stats import qmc
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


# # region latin hypercube sampling


# # Number of samples to generate
# # n_samples = 300
# n_samples = 100

# # Extract bounds in the same order as sim_inputs
# variable_names = ['fuselage_body_length', 'fuselage_width', 'fuselage_height', 'nose_cone_length', 'tail_cone_length',
#                   'normalized_fuselage_fillet_radius', 'normalized_nose_cone_smoothing', 'normalized_tail_cone_smoothing',
#                   'wing_sweep', 'wing_dihedral', 'wing_root_chord', 'wing_mid_chord_relative_and_normalized', 
#                   'wing_tip_chord_relative_and_normalized', 'wing_thickness_to_chord_ratio']

# lower_bounds = np.array([design_variable_bounds[name][0] for name in variable_names])
# upper_bounds = np.array([design_variable_bounds[name][1] for name in variable_names])

# # Generate Latin Hypercube Sampling
# sampler = qmc.LatinHypercube(d=len(variable_names), seed=42)
# unit_samples = sampler.random(n=n_samples)

# # Scale samples to actual bounds
# lhs_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

# print(f"Generated {n_samples} Latin Hypercube samples")
# print(f"Sample shape: {lhs_samples.shape}")
# print(f"Variable order: {variable_names}")

# # Compute baseline values (current values of sim_inputs)
# baseline_values = np.array([jax_input.value.item() for jax_input in sim_inputs])

# # Compute input norms for each sample
# sample_input_difference_norms = []
# for sample in lhs_samples:
#     input_difference_norm = np.linalg.norm(sample - baseline_values)
#     sample_input_difference_norms.append(input_difference_norm)


# sample_results = []
# sample_timings = []

# video_output_path = Path("examples/additional_examples/hand_launched_uavs/hand_launched_uavs_lhs_samples.mp4")
# video_output_path.parent.mkdir(parents=True, exist_ok=True)
# camera = {
#     "position": (-0.75, -0.5, 0.5),
#     "focal_point": (0.1, 0.0, 0.0),
#     "viewup": (0.0, 0.0, 1.0),
# }
# # geometry.plot(camera=camera)
# # exit()

# plotter = pv.Plotter(off_screen=True, window_size=(1920, 1200))
# video_writer = imageio.get_writer(
#     str(video_output_path),
#     fps=11,
#     codec="libx264",
#     macro_block_size=None,
# )

# try:
#     for i, sample in enumerate(lhs_samples):
#         print(f"Running sample {i+1}/{n_samples}")

#         # Set design variable values
#         for j, var_value in enumerate(sample):
#             sim_inputs[j].value = np.array([var_value])
#             sim[sim_inputs[j]] = var_value

#         # Run simulation
#         sim.run()

#         plotter.clear_actors()
#         plotting_elements = geometry.plot(show=False)
        
#         for element in plotting_elements:
#             mesh = element.get("mesh")
#             kwargs = element.get("kwargs", {}).copy()
#             if mesh is None:
#                 continue
#             # Geometry plotting uses this internal key; PyVista add_mesh does not accept it.
#             kwargs.pop("show_edges", None)
#             plotter.add_mesh(mesh, **kwargs)

#         plotter.camera_position = (
#             camera["position"],
#             camera["focal_point"],
#             camera["viewup"],
#         )
#         plotter.render()
#         frame = plotter.screenshot(return_img=True)
#         video_writer.append_data(frame)
# finally:
#     video_writer.close()

# plotter.close()


# # endregion latin hypercube sampling


# region separate latin hypercube sampling for fuselage and wing

# Define variables for fuselage and wing
fuselage_variable_names = ['fuselage_body_length', 'fuselage_width', 'fuselage_height', 'nose_cone_length', 'tail_cone_length',
                           'normalized_fuselage_fillet_radius', 'normalized_nose_cone_smoothing', 'normalized_tail_cone_smoothing']
wing_variable_names = ['wing_sweep', 'wing_dihedral', 'wing_root_chord', 'wing_mid_chord_relative_and_normalized', 
                       'wing_tip_chord_relative_and_normalized', 'wing_thickness_to_chord_ratio']

# Extract bounds for fuselage and wing
fuselage_lower_bounds = np.array([design_variable_bounds[name][0] for name in fuselage_variable_names])
fuselage_upper_bounds = np.array([design_variable_bounds[name][1] for name in fuselage_variable_names])

wing_lower_bounds = np.array([design_variable_bounds[name][0] for name in wing_variable_names])
wing_upper_bounds = np.array([design_variable_bounds[name][1] for name in wing_variable_names])

# Generate Latin Hypercube Sampling for fuselage and wing
n_fuselage_samples = 5
n_wing_samples = 10

# fuselage_sampler = qmc.LatinHypercube(d=len(fuselage_variable_names), seed=40)
fuselage_sampler = qmc.LatinHypercube(d=len(fuselage_variable_names), seed=38)
fuselage_unit_samples = fuselage_sampler.random(n=n_fuselage_samples)
fuselage_lhs_samples = qmc.scale(fuselage_unit_samples, fuselage_lower_bounds, fuselage_upper_bounds)

wing_sampler = qmc.LatinHypercube(d=len(wing_variable_names), seed=41)
wing_unit_samples = wing_sampler.random(n=n_wing_samples)
wing_lhs_samples = qmc.scale(wing_unit_samples, wing_lower_bounds, wing_upper_bounds)

print(f"Generated {n_fuselage_samples} fuselage samples")
print(f"Generated {n_wing_samples} wing samples")
print(f"Total combinations: {n_fuselage_samples * n_wing_samples}")

# Store screenshots for creating grid
screenshot_grid = []

camera = {
    "position": (-0.35, -0.25, 0.25),
    "focal_point": (0.06, 0.0, 0.0),
    "viewup": (0.0, 0.0, 1.0),
}

plotter = pv.Plotter(off_screen=True, window_size=(1920, 1200))
# plotter.camera.parallel_projection = True
# for element in geometry_plot:
#     mesh = element.get("mesh")
#     kwargs = element.get("kwargs", {}).copy()
#     kwargs.pop("show_edges", None)
#     plotter.add_mesh(mesh, **kwargs)
# plotter.camera_position = (
#             camera["position"],
#             camera["focal_point"],
#             camera["viewup"],
#         )
# plotter.render()
# plotter.show()
# exit()

try:
    combination_index = 0
    for f_idx, fuselage_sample in enumerate(fuselage_lhs_samples):
        # Set fuselage variables (indices 0-7 in sim_inputs)
        for j, var_value in enumerate(fuselage_sample):
            sim_inputs[j].value = np.array([var_value])
            sim[sim_inputs[j]] = var_value
        
        # Generate wing-only header screenshots (only on first fuselage iteration)
        if f_idx == 0:
            for w_idx, wing_sample in enumerate(wing_lhs_samples):
                # Set wing variables (indices 8-13 in sim_inputs)
                for j, var_value in enumerate(wing_sample):
                    sim_inputs[8 + j].value = np.array([var_value])
                    sim[sim_inputs[8 + j]] = var_value
                
                # Run simulation
                sim.run()
                
                plotter.clear_actors()
                wing_plotting_elements = wing.plot(show=False)
                
                for element in wing_plotting_elements:
                    mesh = element.get("mesh")
                    kwargs = element.get("kwargs", {}).copy()
                    if mesh is None:
                        continue
                    kwargs.pop("show_edges", None)
                    plotter.add_mesh(mesh, **kwargs)
                
                plotter.camera_position = (
                    camera["position"],
                    camera["focal_point"],
                    camera["viewup"],
                )
                plotter.render()
                frame = plotter.screenshot(return_img=True)
                
                # Save wing header screenshot
                wing_header_folder = Path("examples/additional_examples/hand_launched_uavs/lhs_sample_screenshots/wing_headers")
                wing_header_folder.mkdir(parents=True, exist_ok=True)
                wing_header_path = wing_header_folder / f"wing_{w_idx:02d}.png"
                pil_image = Image.fromarray(frame)
                pil_image.save(str(wing_header_path))
        
        # Generate fuselage-only header screenshot (all fuselage variations)
        wing_idx = 0
        wing_sample = wing_lhs_samples[wing_idx]
        # Set wing variables (indices 8-13 in sim_inputs)
        for j, var_value in enumerate(wing_sample):
            sim_inputs[8 + j].value = np.array([var_value])
            sim[sim_inputs[8 + j]] = var_value
        
        # Run simulation
        sim.run()
        
        plotter.clear_actors()
        fuselage_plotting_elements = fuselage.plot(show=False)
        
        for element in fuselage_plotting_elements:
            mesh = element.get("mesh")
            kwargs = element.get("kwargs", {}).copy()
            if mesh is None:
                continue
            kwargs.pop("show_edges", None)
            plotter.add_mesh(mesh, **kwargs)
        
        plotter.camera_position = (
            camera["position"],
            camera["focal_point"],
            camera["viewup"],
        )
        plotter.render()
        frame = plotter.screenshot(return_img=True)
        
        # Save fuselage header screenshot
        fuselage_header_folder = Path("examples/additional_examples/hand_launched_uavs/lhs_sample_screenshots/fuselage_headers")
        fuselage_header_folder.mkdir(parents=True, exist_ok=True)
        fuselage_header_path = fuselage_header_folder / f"fuselage_{f_idx:02d}.png"
        pil_image = Image.fromarray(frame)
        pil_image.save(str(fuselage_header_path))
        
        for w_idx, wing_sample in enumerate(wing_lhs_samples):
            print(f"Running combination {combination_index+1}/{n_fuselage_samples*n_wing_samples} (fuselage {f_idx+1}/{n_fuselage_samples}, wing {w_idx+1}/{n_wing_samples})")
            
            # Set wing variables (indices 8-13 in sim_inputs)
            for j, var_value in enumerate(wing_sample):
                sim_inputs[8 + j].value = np.array([var_value])
                sim[sim_inputs[8 + j]] = var_value
            
            # Run simulation
            sim.run()
            
            plotter.clear_actors()
            plotting_elements = geometry.plot(show=False)
            
            for element in plotting_elements:
                mesh = element.get("mesh")
                kwargs = element.get("kwargs", {}).copy()
                if mesh is None:
                    continue
                kwargs.pop("show_edges", None)
                plotter.add_mesh(mesh, **kwargs)
            
            plotter.camera_position = (
                camera["position"],
                camera["focal_point"],
                camera["viewup"],
            )
            plotter.render()
            frame = plotter.screenshot(return_img=True)
            
            # Store screenshot and save to file
            screenshot_grid.append(frame)
            
            # Save individual screenshot
            screenshot_folder = Path("examples/additional_examples/hand_launched_uavs/lhs_sample_screenshots/combinations")
            screenshot_folder.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_folder / f"fuselage_{f_idx:02d}_wing_{w_idx:02d}.png"
            pil_image = Image.fromarray(frame)
            pil_image.save(str(screenshot_path))
            
            combination_index += 1
finally:
    plotter.close()

# Create 5x10 grid of screenshots with component plots as headers
print("Creating 5x10 grid with component plots as headers...")

# Get dimensions from first screenshot
screenshot_height, screenshot_width = screenshot_grid[0].shape[:2]
grid_rows = n_fuselage_samples
grid_cols = n_wing_samples

# Create blank canvas for full grid (with headers)
total_width = screenshot_width + grid_cols * screenshot_width
total_height = screenshot_height + grid_rows * screenshot_height
grid_image = Image.new('RGB', (total_width, total_height), color='white')

# Add wing header screenshots (column labels)
wing_header_folder = Path("examples/additional_examples/hand_launched_uavs/lhs_sample_screenshots/wing_headers")
for col in range(grid_cols):
    wing_header_path = wing_header_folder / f"wing_{col:02d}.png"
    wing_header = Image.open(wing_header_path)
    wing_header = wing_header.resize((screenshot_width, screenshot_height))
    x = screenshot_width + col * screenshot_width
    y = 0
    grid_image.paste(wing_header, (x, y))

# Add fuselage header screenshots (row labels) and combination screenshots
fuselage_header_folder = Path("examples/additional_examples/hand_launched_uavs/lhs_sample_screenshots/fuselage_headers")
for row in range(grid_rows):
    if row >= 0:
        fuselage_header_path = fuselage_header_folder / f"fuselage_{row:02d}.png"
        fuselage_header = Image.open(fuselage_header_path)
        fuselage_header = fuselage_header.resize((screenshot_width, screenshot_height))
        x = 0
        y = screenshot_height + row * screenshot_height
        grid_image.paste(fuselage_header, (x, y))

# Paste combination screenshots
for idx, screenshot in enumerate(screenshot_grid):
    row = idx // grid_cols
    col = idx % grid_cols
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(screenshot)
    
    # Paste into grid
    x = screenshot_width + col * screenshot_width
    y = screenshot_height + row * screenshot_height
    grid_image.paste(pil_image, (x, y))

# Save grid
grid_output_path = Path("examples/additional_examples/hand_launched_uavs/hand_launched_uavs_separate_lhs_grid.png")
grid_output_path.parent.mkdir(parents=True, exist_ok=True)
grid_image.save(str(grid_output_path))

print(f"Saved grid to {grid_output_path}")
print(f"Individual screenshots saved to examples/additional_examples/hand_launched_uavs/lhs_sample_screenshots/")

# endregion separate latin hypercube sampling for fuselage and wing


