# region Imports and Setup
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

import lsdo_geo as lg
from lsdo_geo import (
    Geometry,
    construct_ffd_block_around_entities,
    SectionalParameterization,
    SectionalParameters,
    ParameterizationSolver,
    GeometricVariables,
    import_geometry,
)

recorder = csdl.Recorder(inline=True)
recorder.start()

# Import initial geometry that will be deformed
geometry = import_geometry(
    "examples/example_geometries/laser_powered_uav.stp",
    parallelize=False,
)
# Extract initial CAD geometry meshes to plot as reference ghost in video (#B6B1A9, 0.3 opacity)
initial_geometry_elements = geometry.plot(show=False)
initial_meshes = [
    elem["mesh"].copy() if isinstance(elem, dict) and "mesh" in elem else elem.copy()
    for elem in initial_geometry_elements
]

# endregion Imports and Setup

# region Component Declarations
# Declare 6 distinct components covering all 152 surfaces
wing = geometry.declare_component(function_search_names=['wing'], name='wing')
tail = geometry.declare_component(function_search_names=['tail'], name='tail')
left_fuselage = geometry.declare_component(function_search_names=['leftfuse'], name='left_fuselage')
right_fuselage = geometry.declare_component(function_search_names=['rightfuse'], name='right_fuselage')
left_propeller = geometry.declare_component(function_search_names=['lprop', 'ldisk', 'lspin'], name='left_propeller')
right_propeller = geometry.declare_component(function_search_names=['rprop', 'rdisk', 'rspin'], name='right_propeller')
# endregion Component Declarations

# region Key Locations
# Wing key locations
wing_le_center = wing.project(np.array([14.03, 0.0, 0.5]), plot=False)
wing_te_center = wing.project(np.array([17.99, 0.0, 0.5]), plot=False)
wing_le_left = wing.project(np.array([14.89, -31.966, 1.88]), plot=False)
wing_te_left = wing.project(np.array([16.63, -31.966, 1.88]), plot=False)
wing_le_right = wing.project(np.array([14.89, 31.966, 1.88]), plot=False)
wing_te_right = wing.project(np.array([16.63, 31.966, 1.88]), plot=False)

# Tail key locations
tail_le_center = tail.project(np.array([27.50, 0.0, 4.7]), plot=False)
tail_te_center = tail.project(np.array([30.50, 0.0, 4.7]), plot=False)
tail_le_left = tail.project(np.array([28.98, -8.31, -3.04]), plot=False)
tail_te_left = tail.project(np.array([29.98, -8.31, -3.04]), plot=False)
tail_le_right = tail.project(np.array([28.98, 8.31, -3.04]), plot=False)
tail_te_right = tail.project(np.array([29.98, 8.31, -3.04]), plot=False)

# Propeller radius reference points
lprop_top = left_propeller.project(np.array([30.5, -7.0, 4.5]), plot=False)
lprop_bottom = left_propeller.project(np.array([30.5, -7.0, -3.0]), plot=False)
rprop_top = right_propeller.project(np.array([30.5, 7.0, 4.5]), plot=False)
rprop_bottom = right_propeller.project(np.array([30.5, 7.0, -3.0]), plot=False)

# Fuselage attachment reference points
tail_attach_l_para = tail.project(np.array([28.5, -7.0, 0.5]), plot=False)
lfuse_tail_attach_para = left_fuselage.project(np.array([28.5, -7.0, 0.5]), plot=False)
tail_attach_r_para = tail.project(np.array([28.5, 7.0, 0.5]), plot=False)
rfuse_tail_attach_para = right_fuselage.project(np.array([28.5, 7.0, 0.5]), plot=False)

lfuse_rear_para = left_fuselage.project(np.array([30.0, -7.0, 0.75]), plot=False)
lprop_front_para = left_propeller.project(np.array([30.0, -7.0, 0.75]), plot=False)
rfuse_rear_para = right_fuselage.project(np.array([30.0, 7.0, 0.75]), plot=False)
rprop_front_para = right_propeller.project(np.array([30.0, 7.0, 0.75]), plot=False)

# Discretization lines for planform integration
y_left = np.linspace(-31.9657, -7.0, 9)
y_center = np.linspace(-7.0, 7.0, 5)[1:-1]
y_right = np.linspace(7.0, 31.9657, 9)
y_wing_all = np.concatenate([y_left, y_center, y_right])
le_x = np.where(np.abs(y_wing_all) <= 7.0, 14.00, 14.00 + (np.abs(y_wing_all) - 7.0)/(31.9657 - 7.0) * (14.873 - 14.00))
te_x = np.where(np.abs(y_wing_all) <= 7.0, 18.00, 18.00 + (np.abs(y_wing_all) - 7.0)/(31.9657 - 7.0) * (16.622 - 18.00))
z_nom = np.where(np.abs(y_wing_all) <= 7.0, 0.5, 0.5 + (np.abs(y_wing_all) - 7.0)/(31.9657 - 7.0) * (1.88 - 0.5))
pts_wing_le = np.stack([le_x, y_wing_all, z_nom + 0.5], axis=1)
pts_wing_te = np.stack([te_x, y_wing_all, z_nom + 0.5], axis=1)
wing_le_line_para = wing.project(pts_wing_le, direction=np.array([0., 0., -1.]), grid_search_density_parameter=10.)
wing_te_line_para = wing.project(pts_wing_te, direction=np.array([0., 0., -1.]), grid_search_density_parameter=10.)

t_tail_left = np.linspace(0, 1, 7)
pts_tail_le_left = (1 - t_tail_left)[:, None] * np.array([28.98, -8.31, -3.04]) + t_tail_left[:, None] * np.array([27.50, -5.50, 4.7])
pts_tail_te_left = (1 - t_tail_left)[:, None] * np.array([29.98, -8.31, -3.04]) + t_tail_left[:, None] * np.array([30.50, -5.50, 4.7])
t_tail_center = np.linspace(0, 1, 5)[1:-1]
pts_tail_le_center = (1 - t_tail_center)[:, None] * np.array([27.50, -5.50, 4.7]) + t_tail_center[:, None] * np.array([27.50, 5.50, 4.7])
pts_tail_te_center = (1 - t_tail_center)[:, None] * np.array([30.50, -5.50, 4.7]) + t_tail_center[:, None] * np.array([30.50, 5.50, 4.7])
t_tail_right = np.linspace(0, 1, 7)
pts_tail_le_right = (1 - t_tail_right)[:, None] * np.array([27.50, 5.50, 4.7]) + t_tail_right[:, None] * np.array([28.98, 8.31, -3.04])
pts_tail_te_right = (1 - t_tail_right)[:, None] * np.array([30.50, 5.50, 4.7]) + t_tail_right[:, None] * np.array([29.98, 8.31, -3.04])
tail_le_line_para = tail.project(np.vstack([pts_tail_le_left, pts_tail_le_center, pts_tail_le_right]), plot=False)
tail_te_line_para = tail.project(np.vstack([pts_tail_te_left, pts_tail_te_center, pts_tail_te_right]), plot=False)
# endregion Key Locations

# region Create Parameterization Objects (FFD Blocks & Sectional Parameterizations)
# FFD Blocks
wing_ffd_block = construct_ffd_block_around_entities(entities=wing, num_coefficients=(2, 5, 2), degree=(1, 1, 1), name="wing_ffd_block")
tail_ffd_block = construct_ffd_block_around_entities(entities=tail, num_coefficients=(2, 3, 2), degree=(1, 1, 1), name="tail_ffd_block")
left_fuselage_ffd_block = construct_ffd_block_around_entities(entities=left_fuselage, num_coefficients=(2, 2, 2), degree=(1, 1, 1), name="left_fuselage_ffd_block")
right_fuselage_ffd_block = construct_ffd_block_around_entities(entities=right_fuselage, num_coefficients=(2, 2, 2), degree=(1, 1, 1), name="right_fuselage_ffd_block")
left_propeller_ffd_block = construct_ffd_block_around_entities(entities=left_propeller, num_coefficients=(2, 2, 2), degree=(1, 1, 1), name="left_propeller_ffd_block")
right_propeller_ffd_block = construct_ffd_block_around_entities(entities=right_propeller, num_coefficients=(2, 2, 2), degree=(1, 1, 1), name="right_propeller_ffd_block")

# Sectional Parameterizations
wing_sectional_parameterization = SectionalParameterization(name="wing_sectional_parameterization", parameterized_points=wing_ffd_block.coefficients, principal_parametric_dimension=1)
tail_sectional_parameterization = SectionalParameterization(name="tail_sectional_parameterization", parameterized_points=tail_ffd_block.coefficients, principal_parametric_dimension=1)
left_fuselage_sectional_parameterization = SectionalParameterization(name="left_fuselage_sectional_parameterization", parameterized_points=left_fuselage_ffd_block.coefficients, principal_parametric_dimension=0)
right_fuselage_sectional_parameterization = SectionalParameterization(name="right_fuselage_sectional_parameterization", parameterized_points=right_fuselage_ffd_block.coefficients, principal_parametric_dimension=0)
left_propeller_sectional_parameterization = SectionalParameterization(name="left_propeller_sectional_parameterization", parameterized_points=left_propeller_ffd_block.coefficients, principal_parametric_dimension=0)
right_propeller_sectional_parameterization = SectionalParameterization(name="right_propeller_sectional_parameterization", parameterized_points=right_propeller_ffd_block.coefficients, principal_parametric_dimension=0)

# B-spline Function Spaces
space_3dof_linear = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
space_2dof_linear = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
space_1dof_constant = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))

# Parameterization States
wing_chord_stretch_bs = lfs.Function(space=space_3dof_linear, coefficients=csdl.Variable(shape=(3,), value=np.zeros(3)), name="wing_chord_stretch")
wing_span_stretch_bs = lfs.Function(space=space_2dof_linear, coefficients=csdl.Variable(shape=(2,), value=np.zeros(2)), name="wing_span_stretch")

tail_chord_stretch_bs = lfs.Function(space=space_3dof_linear, coefficients=csdl.Variable(shape=(3,), value=np.zeros(3)), name="tail_chord_stretch")
tail_span_stretch_bs = lfs.Function(space=space_2dof_linear, coefficients=csdl.Variable(shape=(2,), value=np.zeros(2)), name="tail_span_stretch")
# tail_trans_x_bs = lfs.Function(space=space_1dof_constant, coefficients=csdl.Variable(shape=(1,), value=np.zeros(1)), name="tail_trans_x")

# left_fuselage_stretch_bs = lfs.Function(space=space_2dof_linear, coefficients=csdl.Variable(shape=(2,), value=np.zeros(2)), name="left_fuselage_stretch")
# right_fuselage_stretch_bs = lfs.Function(space=space_2dof_linear, coefficients=csdl.Variable(shape=(2,), value=np.zeros(2)), name="right_fuselage_stretch")

fuselage_stretch = csdl.Variable(value=0.)
fuselage_translation = csdl.Variable(shape=(3,), value=0.)
right_fuselage_translation = fuselage_translation
left_fuselage_translation = fuselage_translation.set(csdl.slice[1], -fuselage_translation[1])

# left_prop_stretch_bs = lfs.Function(space=space_1dof_constant, coefficients=csdl.Variable(shape=(1,), value=np.zeros(1)), name="left_prop_stretch")
# right_prop_stretch_bs = lfs.Function(space=space_1dof_constant, coefficients=csdl.Variable(shape=(1,), value=np.zeros(1)), name="right_prop_stretch")
# left_prop_trans_x_bs = lfs.Function(space=space_1dof_constant, coefficients=csdl.Variable(shape=(1,), value=np.zeros(1)), name="left_prop_trans_x")
# right_prop_trans_x_bs = lfs.Function(space=space_1dof_constant, coefficients=csdl.Variable(shape=(1,), value=np.zeros(1)), name="right_prop_trans_x")

propeller_stretch = csdl.Variable(value=0.)

propeller_translation = csdl.Variable(shape=(3,), value=0.)
right_propeller_translation = propeller_translation
left_propeller_translation = propeller_translation.set(
    csdl.slice[1], -propeller_translation[1])

# endregion Create Parameterization Objects

# region Evaluate Forward Parameterization Map
# Wing
u_wing = np.linspace(0, 1, wing_sectional_parameterization.num_sections).reshape((-1, 1))
wing_params = SectionalParameters()
wing_params.add_stretch(axis=0, stretch=wing_chord_stretch_bs.evaluate(u_wing))
wing_params.add_translation(axis=1, translation=wing_span_stretch_bs.evaluate(u_wing))
wing_ffd_coeffs = wing_sectional_parameterization.evaluate(wing_params, plot=False)
wing.set_coefficients(wing_ffd_block.evaluate_ffd(wing_ffd_coeffs, plot=False))

# Tail
u_tail = np.linspace(0, 1, tail_sectional_parameterization.num_sections).reshape((-1, 1))
tail_params = SectionalParameters()
tail_params.add_stretch(axis=0, stretch=tail_chord_stretch_bs.evaluate(u_tail))
tail_params.add_translation(axis=1, translation=tail_span_stretch_bs.evaluate(u_tail))
tail_params.add_translation(axis=0, translation=tail_trans_x_bs.evaluate(u_tail))
tail_ffd_coeffs = tail_sectional_parameterization.evaluate(tail_params, plot=False)
tail.set_coefficients(tail_ffd_block.evaluate_ffd(tail_ffd_coeffs, plot=False))

# Fuselages
# u_fuse = np.linspace(0, 1, left_fuselage_sectional_parameterization.num_sections).reshape((-1, 1))
fuselage_sectional_stretches = csdl.concatenate((-fuselage_stretch/2, fuselage_stretch/2))

lf_params = SectionalParameters()
lf_params.add_translation(axis=0, translation=fuselage_sectional_stretches)
left_fuselage.set_coefficients(left_fuselage_ffd_block.evaluate_ffd(left_fuselage_sectional_parameterization.evaluate(lf_params, plot=False), plot=False))
left_fuselage.translate(translation=left_fuselage_translation)

rf_params = SectionalParameters()
rf_params.add_translation(axis=0, translation=fuselage_sectional_stretches)
right_fuselage.set_coefficients(right_fuselage_ffd_block.evaluate_ffd(right_fuselage_sectional_parameterization.evaluate(rf_params, plot=False), plot=False))

# Propellers (radial stretch on Y & Z, longitudinal translation on X)

u_prop = np.linspace(0, 1, 2).reshape((-1, 1))
lp_params = SectionalParameters()
lp_params.add_stretch(axis=1, stretch=propeller_stretch)
lp_params.add_stretch(axis=2, stretch=propeller_stretch)
# lp_params.add_translation(axis=0, translation=left_prop_trans_x_bs.evaluate(u_prop))
left_propeller.set_coefficients(left_propeller_ffd_block.evaluate_ffd(left_propeller_sectional_parameterization.evaluate(lp_params, plot=False), plot=False))
left_propeller.translate(translation=left_propeller_translation)

rp_params = SectionalParameters()
rp_params.add_stretch(axis=1, stretch=propeller_stretch)
rp_params.add_stretch(axis=2, stretch=propeller_stretch)
# rp_params.add_translation(axis=0, translation=right_prop_trans_x_bs.evaluate(u_prop))
right_propeller.set_coefficients(right_propeller_ffd_block.evaluate_ffd(right_propeller_sectional_parameterization.evaluate(rp_params, plot=False), plot=False))
right_propeller.translate(translation=right_propeller_translation)

# Tail incidence angle rotation
tail_incidence_angle = csdl.Variable(name="tail_incidence_angle", value=np.array([2.0]))
tail_qc_center_eval = tail.evaluate(tail_le_center) + 0.25 * (tail.evaluate(tail_te_center) - tail.evaluate(tail_le_center))
tail.rotate(rotation_origin=tail_qc_center_eval, axis_vector=np.array([0., 1., 0.]), angles=tail_incidence_angle, units="degrees")

# Computed Wing Metrics
wing_span_comp = geometry.evaluate(wing_le_right)[1] - geometry.evaluate(wing_le_left)[1]
wing_root_chord_comp = geometry.evaluate(wing_te_center)[0] - geometry.evaluate(wing_le_center)[0]
wing_tip_chord_l_comp = geometry.evaluate(wing_te_left)[0] - geometry.evaluate(wing_le_left)[0]
wing_tip_chord_r_comp = geometry.evaluate(wing_te_right)[0] - geometry.evaluate(wing_le_right)[0]
wing_taper_comp = (wing_tip_chord_l_comp + wing_tip_chord_r_comp) / (2 * wing_root_chord_comp)

wing_cs = csdl.linear_combination(geometry.evaluate(wing_le_line_para), geometry.evaluate(wing_te_line_para), 5)
u_vw = wing_cs[1:, :] - wing_cs[:-1, :]
v_vw = wing_cs[:, 1:] - wing_cs[:, :-1]
pa_w = 0.5 * csdl.cross(u_vw[:, :-1], v_vw[:-1, :], axis=2) + 0.5 * csdl.cross(u_vw[:, 1:], v_vw[1:, :], axis=2)
wing_area_comp = csdl.sum(csdl.norm(pa_w, axes=(2,)))
wing_ar_comp = wing_span_comp**2 / wing_area_comp

# Computed Tail Metrics
tail_span_comp = geometry.evaluate(tail_le_right)[1] - geometry.evaluate(tail_le_left)[1]
tail_root_chord_comp = geometry.evaluate(tail_te_center)[0] - geometry.evaluate(tail_le_center)[0]
tail_tip_chord_l_comp = geometry.evaluate(tail_te_left)[0] - geometry.evaluate(tail_le_left)[0]
tail_tip_chord_r_comp = geometry.evaluate(tail_te_right)[0] - geometry.evaluate(tail_le_right)[0]

tail_cs = csdl.linear_combination(geometry.evaluate(tail_le_line_para), geometry.evaluate(tail_te_line_para), 4)
u_vt = tail_cs[1:, :] - tail_cs[:-1, :]
v_vt = tail_cs[:, 1:] - tail_cs[:, :-1]
pa_t = 0.5 * csdl.cross(u_vt[:, :-1], v_vt[:-1, :], axis=2) + 0.5 * csdl.cross(u_vt[:, 1:], v_vt[1:, :], axis=2)
tail_area_comp = csdl.sum(csdl.norm(pa_t, axes=(2,)))
tail_ar_comp = tail_span_comp**2 / tail_area_comp
tail_taper_comp = (tail_tip_chord_l_comp + tail_tip_chord_r_comp) / (2 * tail_root_chord_comp)

wing_qc_eval = geometry.evaluate(wing_le_center) + 0.25 * (geometry.evaluate(wing_te_center) - geometry.evaluate(wing_le_center))
tail_qc_eval = geometry.evaluate(tail_le_center) + 0.25 * (geometry.evaluate(tail_te_center) - geometry.evaluate(tail_le_center))
tail_moment_arm_comp = tail_qc_eval[0] - wing_qc_eval[0]

# Computed Propeller Radii
lprop_radius_comp = (geometry.evaluate(lprop_top)[2] - geometry.evaluate(lprop_bottom)[2]) / 2.0
rprop_radius_comp = (geometry.evaluate(rprop_top)[2] - geometry.evaluate(rprop_bottom)[2]) / 2.0

# Connection Invariants
conn_tail_lf = geometry.evaluate(tail_attach_l_para) - geometry.evaluate(lfuse_tail_attach_para)
conn_tail_rf = geometry.evaluate(tail_attach_r_para) - geometry.evaluate(rfuse_tail_attach_para)
conn_prop_lf = geometry.evaluate(lfuse_rear_para) - geometry.evaluate(lprop_front_para)
conn_prop_rf = geometry.evaluate(rfuse_rear_para) - geometry.evaluate(rprop_front_para)
# endregion Evaluate Forward Parameterization Map

# region Target Design Variables
wing_area_dv = csdl.Variable(name="wing_area", value=np.array([205.0]))
wing_ar_dv = csdl.Variable(name="wing_aspect_ratio", value=np.array([21.0]))
wing_taper_dv = csdl.Variable(name="wing_taper_ratio", value=np.array([0.45]))

tail_area_dv = csdl.Variable(name="tail_area", value=np.array([62.0]))
tail_ar_dv = csdl.Variable(name="tail_aspect_ratio", value=np.array([4.1]))
tail_taper_dv = csdl.Variable(name="tail_taper_ratio", value=tail_taper_comp.value)  # Keep tail taper ratio fixed for this example
tail_moment_arm_dv = csdl.Variable(name="tail_moment_arm", value=np.array([14.0]))

propeller_radius_dv = csdl.Variable(name="propeller_radius", value=np.array([4.0]))
# endregion Target Design Variables

# region Setup and Evaluate Geometry Parameterization Solver
print("=== Initial Geometry Parameters ===")
print(f"Scale Factor = {scale_factor:.2f}")
print(f"Wing: Area = {wing_area_comp.value[0]:.2f}, AR = {wing_ar_comp.value[0]:.2f}, Taper = {wing_taper_comp.value[0]:.4f}, Span = {wing_span_comp.value[0]:.2f}")
print(f"Tail: Area = {tail_area_comp.value[0]:.2f}, AR = {tail_ar_comp.value[0]:.2f}, Taper = {tail_taper_comp.value[0]:.4f}, Span = {tail_span_comp.value[0]:.2f}")
print(f"Tail Moment Arm = {tail_moment_arm_comp.value[0]:.2f}")
print(f"Tail Incidence Angle = {tail_incidence_angle.value[0]:.2f} deg")
print(f"Propeller Radii: Left = {lprop_radius_comp.value[0]:.3f}, Right = {rprop_radius_comp.value[0]:.3f}")
print()

parameterization_solver = ParameterizationSolver()

# Solver States
parameterization_solver.add_state(wing_chord_stretch_bs.coefficients)
parameterization_solver.add_state(wing_span_stretch_bs.coefficients)
parameterization_solver.add_state(tail_chord_stretch_bs.coefficients)
parameterization_solver.add_state(tail_span_stretch_bs.coefficients)
parameterization_solver.add_state(fuselage_stretch)
parameterization_solver.add_state(fuselage_translation)
parameterization_solver.add_state(propeller_stretch)
parameterization_solver.add_state(propeller_translation)
# parameterization_solver.add_state(tail_trans_x_bs.coefficients)
# parameterization_solver.add_state(left_fuselage_stretch_bs.coefficients)
# parameterization_solver.add_state(right_fuselage_stretch_bs.coefficients)
# parameterization_solver.add_state(left_prop_stretch_bs.coefficients)
# parameterization_solver.add_state(right_prop_stretch_bs.coefficients)
# parameterization_solver.add_state(left_prop_trans_x_bs.coefficients)
# parameterization_solver.add_state(right_prop_trans_x_bs.coefficients)

# Constraints
parameterization_solver.add_equality_constraint(wing_tip_chord_l_comp, wing_tip_chord_r_comp)
parameterization_solver.add_equality_constraint(tail_tip_chord_l_comp, tail_tip_chord_r_comp)
parameterization_solver.add_equality_constraint(conn_tail_lf, conn_tail_lf.value)
parameterization_solver.add_equality_constraint(conn_tail_rf, conn_tail_rf.value)
parameterization_solver.add_equality_constraint(conn_prop_lf, conn_prop_lf.value)
parameterization_solver.add_equality_constraint(conn_prop_rf, conn_prop_rf.value)

# Geometric Variables
geometric_variables = GeometricVariables()
geometric_variables.add_variable(wing_area_comp, wing_area_dv)
geometric_variables.add_variable(wing_ar_comp, wing_ar_dv)
geometric_variables.add_variable(wing_taper_comp, wing_taper_dv)
geometric_variables.add_variable(tail_area_comp, tail_area_dv)
geometric_variables.add_variable(tail_ar_comp, tail_ar_dv)
geometric_variables.add_variable(tail_taper_comp, tail_taper_dv)
geometric_variables.add_variable(tail_moment_arm_comp, tail_moment_arm_dv)
# geometric_variables.add_variable(lprop_radius_comp, propeller_radius_dv)
geometric_variables.add_variable(rprop_radius_comp, propeller_radius_dv)

print("Solving geometry parameterization...")
start_time = time.time()
parameterization_solver.evaluate(geometric_variables)
end_time = time.time()
print(f"Solver completed in {end_time - start_time:.2f} seconds.\n")

print("=== Solved Geometry Parameters ===")
print(f"Wing: Area = {wing_area_comp.value[0]:.2f} (target {wing_area_dv.value[0]:.2f}), AR = {wing_ar_comp.value[0]:.2f} (target {wing_ar_dv.value[0]:.2f}), Taper = {wing_taper_comp.value[0]:.4f} (target {wing_taper_dv.value[0]:.4f}), Span = {wing_span_comp.value[0]:.2f}")
print(f"Tail: Area = {tail_area_comp.value[0]:.2f} (target {tail_area_dv.value[0]:.2f}), AR = {tail_ar_comp.value[0]:.2f} (target {tail_ar_dv.value[0]:.2f}), Taper = {tail_taper_comp.value[0]:.4f} (target {tail_taper_dv.value[0]:.4f}), Moment Arm = {tail_moment_arm_comp.value[0]:.2f} (target {tail_moment_arm_dv.value[0]:.2f}), Incidence Angle = {tail_incidence_angle.value[0]:.2f} deg, Span = {tail_span_comp.value[0]:.2f}")
print(f"Tail Moment Arm = {tail_moment_arm_comp.value[0]:.2f} (target {tail_moment_arm_dv.value[0]:.2f})")
print(f"Tail Incidence Angle = {tail_incidence_angle.value[0]:.2f} deg")
print(f"Propeller Radii: Left = {lprop_radius_comp.value[0]:.3f}, Right = {rprop_radius_comp.value[0]:.3f} (target {propeller_radius_dv.value[0]:.3f})")
print()

# endregion Setup and Evaluate Geometry Parameterization Solver

# region Perturbation Video Generation with JaxSimulator
import pyvista as pv
import shutil

print("=== Setting up JaxSimulator for Geometric Variable Perturbations ===")

jax_inputs = [
    wing_area_dv,
    wing_ar_dv,
    wing_taper_dv,
    tail_area_dv,
    tail_ar_dv,
    tail_moment_arm_dv,
    tail_incidence_angle,
    propeller_radius_dv,
]
jax_outputs = [f.coefficients for f in geometry.functions.values()]

print("Compiling JaxSimulator for video parameter sweep...")
sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=jax_inputs,
    additional_outputs=jax_outputs,
    gpu=False,
)
sim.run()
print("JaxSimulator compiled successfully!\n")

output_dir = "examples/showcase_examples/laser_powered_uav"
video_path = os.path.join(output_dir, "laser_powered_uav_perturbations.mp4")

pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True, window_size=[1920, 1088])
# fps = 20
fps = 10
plotter.open_movie(video_path, framerate=fps)

camera_pos = (-25.0, -55.0, 55.0)
focal_pt = (18.0, 0.0, 1.0)
view_up = (0.0, 0.0, 1.0)

video_components = [
    (wing, "#3498db", "Wing"),
    (tail, "#e67e22", "Tail"),
    (left_fuselage, "#95a5a6", "Left Fuselage"),
    (right_fuselage, "#95a5a6", "Right Fuselage"),
    (left_propeller, "#e74c3c", "Left Propeller"),
    (right_propeller, "#e74c3c", "Right Propeller"),
]

# Pre-merge initial CAD geometry meshes into a single reference ghost mesh for high-performance rendering
valid_ghost_meshes = [m for m in initial_meshes if m is not None]
ghost_mesh = valid_ghost_meshes[0].merge(valid_ghost_meshes[1:]) if len(valid_ghost_meshes) > 1 else valid_ghost_meshes[0]

def render_frame(title_str, val_str, delta_str=""):
    plotter.clear()

    # 1. Initial geometry reference ghost plotted in #B6B1A9 with 0.3 opacity on all frames
    plotter.add_mesh(
        ghost_mesh,
        color="#B6B1A9",
        opacity=0.3,
        smooth_shading=True,
        show_edges=False,
    )

    # 2. Current perturbed geometry with vivid component colors
    for comp, col, name in video_components:
        comp_meshes = [elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem for elem in comp.plot(show=False)]
        comp_meshes = [m for m in comp_meshes if m is not None]
        if comp_meshes:
            merged_comp = comp_meshes[0].merge(comp_meshes[1:]) if len(comp_meshes) > 1 else comp_meshes[0]
            plotter.add_mesh(
                merged_comp,
                color=col,
                smooth_shading=True,
                specular=0.6,
                specular_power=20,
                ambient=0.25,
                diffuse=0.75,
                show_edges=False,
            )

    plotter.enable_lightkit()
    plotter.set_background("#12151c", top="#1e2330")

    hud_text = (
        f"LSDO_GEO: Laser-Powered UAV Parameterization\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Design Variable: {title_str}\n"
        f"Current Value:   {val_str}\n"
        f"Perturbation:    {delta_str}\n"
        f"Ghost Reference: Initial CAD Geometry (#B6B1A9, 0.3 opacity)"
    )
    plotter.add_text(
        hud_text,
        position="upper_left",
        font_size=12,
        color="white",
        font="courier",
        shadow=True,
    )

    plotter.camera.position = camera_pos
    plotter.camera.focal_point = focal_pt
    plotter.camera.up = view_up
    plotter.write_frame()

# List of geometric design variables to loop over with nominal values and perturbation offsets
variables_to_sweep = [
    {
        'variable': wing_area_dv,
        'name': 'Wing Area',
        'unit': 'm²',
        'nominal': wing_area_dv.value,
        'offset': wing_area_dv.value*0.5,
    },
    {
        'variable': wing_ar_dv,
        'name': 'Wing Aspect Ratio',
        'unit': '',
        'nominal': wing_ar_dv.value,
        'offset': wing_ar_dv.value*0.5,
    },
    {
        'variable': wing_taper_dv,
        'name': 'Wing Taper Ratio',
        'unit': '',
        'nominal': wing_taper_dv.value,
        'offset': wing_taper_dv.value*0.5,
    },
    {
        'variable': tail_area_dv,
        'name': 'Tail Area',
        'unit': 'm²',
        'nominal': tail_area_dv.value,
        'offset': tail_area_dv.value*0.5,
    },
    {
        'variable': tail_ar_dv,
        'name': 'Tail Aspect Ratio',
        'unit': '',
        'nominal': tail_ar_dv.value,
        'offset': tail_ar_dv.value*0.5,
    },
    {
        'variable': tail_moment_arm_dv,
        'name': 'Tail Moment Arm',
        'unit': 'm',
        'nominal': tail_moment_arm_dv.value,
        'offset': tail_moment_arm_dv.value*0.5,
    },
    {
        'variable': tail_incidence_angle,
        'name': 'Tail Incidence Angle',
        'unit': '°',
        'nominal': tail_incidence_angle.value,
        'offset': 30.0,  # 10 degrees (tail.rotate uses units="degrees")
    },
    {
        'variable': propeller_radius_dv,
        'name': 'Propeller Radius',
        'unit': 'm',
        'nominal': propeller_radius_dv.value,
        'offset': propeller_radius_dv.value*0.5,
    },
]

def generate_sweep_values_with_hold(nominal, offset, num_osc_points=18, hold_points=3):
    nominal = float(np.squeeze(nominal))
    offset = float(np.squeeze(offset))
    hold_start = np.full(hold_points, nominal)
    osc = nominal + offset * np.sin(np.linspace(0, 2 * np.pi, num_osc_points, endpoint=False))
    hold_end = np.full(hold_points, nominal)
    return np.concatenate([hold_start, osc, hold_end])

print("Rendering video frames...")

# Opening hold position (8 frames = 0.4 sec)
for _ in range(8):
    render_frame("Baseline Solved Configuration", "All Variables at Nominal", "Nominal Hold Position")

# Nested loop: outer loop over variables, inner loop over perturbation values
for var_idx, var_info in enumerate(variables_to_sweep):
    target_var = var_info['variable']
    var_name = var_info['name']
    unit_str = var_info['unit']
    nominal_val = float(np.squeeze(var_info['nominal']))
    offset_val = float(np.squeeze(var_info['offset']))

    print(f"  [{var_idx+1}/{len(variables_to_sweep)}] Sweeping {var_name} (nominal = {nominal_val:.2f}{unit_str}, offset = ±{offset_val:.2f}{unit_str})...")

    sweep_values = generate_sweep_values_with_hold(nominal_val, offset_val, num_osc_points=18, hold_points=3)

    for val in sweep_values:
        sim[target_var] = np.array([val])
        sim.run()

        delta = val - nominal_val
        val_display = f"{val:.2f} {unit_str}".strip()
        delta_display = f"Δ = {delta:+.2f} {unit_str}".strip() if abs(delta) > 1e-4 else "Hold at Nominal"

        render_frame(var_name, val_display, delta_display)

    # Reset this variable back to nominal before advancing to next variable
    sim[target_var] = np.array([nominal_val])
    sim.run()

# Closing hold position (8 frames = 0.4 sec)
for _ in range(8):
    render_frame("Nominal Return", "All 9 Geometric DVs Verified", "Parameterization Complete")

plotter.close()
print(f"Successfully generated video: {video_path}")

# Copy video to artifact directory
artifact_dir = "/home/andrew/.gemini/antigravity/brain/bc376788-3f47-49d2-871b-76ff050ff703"
if os.path.exists(artifact_dir):
    dest_video = os.path.join(artifact_dir, "laser_powered_uav_perturbations.mp4")
    shutil.copy2(video_path, dest_video)
    print(f"Copied video to artifact directory: {dest_video}")
# endregion Perturbation Video Generation with JaxSimulator