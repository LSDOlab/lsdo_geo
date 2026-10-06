"""
Lift+Cruise (LPC) Geometric Parameterization
=============================================
Standard geometry parameterization example for the Lift+Cruise aircraft.

Key Features:
1. Wing & Tail design variables: Planform Area (S), Aspect Ratio (AR),
   and Taper Ratio (lambda) matching the direct ratio formulation.
2. Wing Tip Twist variable with root twist fixed explicitly to 0.0.
3. Cruise rotor (pusher prop) radius as an explicit design variable with FFD block.
4. 3-Section Fuselage parameterization with separate cabin length and
   tail moment arm variables, along with cabin width and height variables.
5. Configurable inequality constraint flag (`ENABLE_INEQUALITIES = True/False`)
   to enforce rotor-rotor and rotor-fuselage clearance constraints.
6. Interactive parameter sweep / perturbation animation generation.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time
import shutil
import numpy as np
import pyvista as pv

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo as lg
from lsdo_geo import (
    import_geometry,
    ParameterizationSolver,
    GeometricVariables,
    construct_ffd_block_around_entities,
    SectionalParameterization,
    SectionalParameters,
)

lfs.num_workers = 1

# Configuration flag (can be overridden via environment variable)
ENABLE_INEQUALITIES = os.environ.get("LPC_ENABLE_INEQUALITIES", "False").lower() in ("true", "1", "yes")

# ==============================================================================
# 1. Geometry Import and Component Declarations
# ==============================================================================
recorder = csdl.Recorder(inline=True)
recorder.start()

import_file_path = "examples/example_geometries/"
import_file = "lift_plus_cruise_final.stp"
geometry = import_geometry(import_file_path + import_file, parallelize=False)

# Main lifting surfaces and fuselage
wing = geometry.declare_component(function_search_names=["Wing"], name="wing")
h_tail = geometry.declare_component(function_search_names=["Tail_1"], name="h_tail")
v_tail = geometry.declare_component(function_search_names=["Tail_2"], name="v_tail")
fuselage = geometry.declare_component(function_search_names=["Fuselage_***.main"], name="fuselage")
nose_hub = geometry.declare_component(name="weird_nose_hub", function_search_names=["EngineGroup_10"])

# Cruise rotor (pusher prop) components
pp_disk = geometry.declare_component(name="pp_disk", function_search_names=["Rotor-9-disk"])
pp_blade_1 = geometry.declare_component(name="pp_blade_1", function_search_names=["Rotor_9_blades, 0"])
pp_blade_2 = geometry.declare_component(name="pp_blade_2", function_search_names=["Rotor_9_blades, 1"])
pp_blade_3 = geometry.declare_component(name="pp_blade_3", function_search_names=["Rotor_9_blades, 2"])
pp_blade_4 = geometry.declare_component(name="pp_blade_4", function_search_names=["Rotor_9_blades, 3"])
pp_hub = geometry.declare_component(name="pp_hub", function_search_names=["Rotor_9_Hub"])
pp_components = [pp_disk, pp_blade_1, pp_blade_2, pp_blade_3, pp_blade_4, pp_hub]

# 8 Lift Rotors and Booms
rotor_names = ["rlo", "rli", "rri", "rro", "flo", "fli", "fri", "fro"]
rotor_indices = [2, 4, 6, 8, 1, 3, 5, 7]
rotor_components = []
boom_components = []

for name, idx in zip(rotor_names, rotor_indices):
    disk = geometry.declare_component(name=f"{name}_disk", function_search_names=[f"Rotor_{idx}_disk"])
    b1 = geometry.declare_component(name=f"{name}_blade_1", function_search_names=[f"Rotor_{idx}_blades, 0"])
    b2 = geometry.declare_component(name=f"{name}_blade_2", function_search_names=[f"Rotor_{idx}_blades, 1"])
    hub = geometry.declare_component(name=f"{name}_hub", function_search_names=[f"Rotor_{idx}_Hub"])
    boom = geometry.declare_component(name=f"{name}_boom", function_search_names=[f"Rotor_{idx}_Support"])
    rotor_components.append([disk, b1, b2, hub])
    boom_components.append(boom)

# ==============================================================================
# 2. Key Projections for Parameterization & Constraints
# ==============================================================================
wing_te_right = wing.project(np.array([13.4, 25.250, 7.5]), plot=False)
wing_te_left = wing.project(np.array([13.4, -25.250, 7.5]), plot=False)
wing_te_center = wing.project(np.array([14.332, 0.0, 8.439]), plot=False)
wing_le_right = wing.project(np.array([12.356, 25.25, 7.618]), plot=False)
wing_le_left = wing.project(np.array([12.356, -25.25, 7.618]), plot=False)
wing_le_center = wing.project(np.array([8.892, 0.0, 8.633]), plot=False)
wing_qc = wing.project(np.array([10.25, 0.0, 8.5]), plot=False)

tail_te_right = h_tail.project(np.array([31.5, 6.75, 6.0]), plot=False)
tail_te_left = h_tail.project(np.array([31.5, -6.75, 6.0]), plot=False)
tail_le_right = h_tail.project(np.array([26.5, 6.75, 6.0]), plot=False)
tail_le_left = h_tail.project(np.array([26.5, -6.75, 6.0]), plot=False)
tail_te_center = h_tail.project(np.array([31.187, 0.0, 8.009]), plot=False)
tail_le_center = h_tail.project(np.array([27.428, 0.0, 8.009]), plot=False)
tail_qc = h_tail.project(np.array([24.15, 0.0, 8.0]), plot=False)

fuselage_wing_te_center = fuselage.project(np.array([14.332, 0.0, 8.439]), plot=False)
fuselage_tail_te_center = fuselage.project(np.array([31.187, 0.0, 8.009]), plot=False)
fuselage_nose_points_parametric = fuselage.project(np.array([2.464, 0.0, 5.113]), grid_search_density_parameter=20)
fuselage_rear_points_parametric = fuselage.project(np.array([31.889, 0.0, 7.798]))
fuselage_rear_point_on_pusher_disk_parametric = pp_disk.project(np.array([31.889, 0.0, 7.798]))

fuselage_top_parametric = fuselage.project(np.array([10.0, 0.0, 8.8]), plot=False)
fuselage_bottom_parametric = fuselage.project(np.array([10.0, 0.0, 2.3]), plot=False)
fuselage_right_parametric = fuselage.project(np.array([10.0, 3.1, 5.5]), plot=False)
fuselage_left_parametric = fuselage.project(np.array([10.0, -3.1, 5.5]), plot=False)

pp_disk_top = pp_disk.project(np.array([31.94, 0.00, 12.29]), plot=False)
pp_disk_bot = pp_disk.project(np.array([31.94, 0.00, 3.29]), plot=False)

rotor_pts = [
    (np.array([19.2, -13.75, 9.01]), np.array([19.2, -23.75, 9.01])),
    (np.array([18.76, -3.499, 9.996]), np.array([18.76, -13.401, 8.604])),
    (np.array([18.76, 13.401, 8.60]), np.array([18.76, 3.499, 9.996])),
    (np.array([19.2, 23.75, 9.01]), np.array([19.2, 13.75, 9.01])),
    (np.array([5.07, -13.75, 6.73]), np.array([5.07, -23.75, 6.73])),
    (np.array([4.63, -3.179, 7.736]), np.array([4.63, -13.081, 6.344])),
    (np.array([4.63, 13.081, 6.344]), np.array([4.63, 3.179, 7.736])),
    (np.array([5.07, 23.75, 6.73]), np.array([5.07, 13.75, 6.73])),
]
rotor_edges = []
for i, comp_set in enumerate(rotor_components):
    disk = comp_set[0]
    p1 = disk.project(rotor_pts[i][0])
    p2 = disk.project(rotor_pts[i][1])
    rotor_edges.append((p1, p2))

boom_pts = [
    np.array([12.000, -18.750, 7.613]),
    np.array([11.500, -8.250, 7.898]),
    np.array([11.500, 8.250, 7.898]),
    np.array([12.000, 18.750, 7.613]),
    np.array([12.200, -18.750, 7.615]),
    np.array([11.741, -8.250, 7.900]),
    np.array([11.741, 8.250, 7.900]),
    np.array([12.200, 18.750, 7.615]),
]
boom_projections = [boom.project(pt) for boom, pt in zip(boom_components, boom_pts)]
wing_boom_projections = [wing.project(pt) for pt in boom_pts]

wing_num_spanwise_quad = 23
tail_num_spanwise_quad = 11

wing_le_line_para = wing.project(
    np.linspace(np.array([8.356, -26.0, 7.618]), np.array([8.356, 26.0, 7.618]), wing_num_spanwise_quad),
    direction=np.array([0.0, 0.0, -1.0]),
)
wing_te_line_para = wing.project(
    np.linspace(np.array([15.4, -25.250, 7.5]), np.array([15.4, 25.250, 7.5]), wing_num_spanwise_quad),
    direction=np.array([0.0, 0.0, -1.0]),
)

ht_le_para = h_tail.project(
    np.linspace(np.array([26.5, -6.75, 6.0]), np.array([26.5, 6.75, 6.0]), tail_num_spanwise_quad),
    direction=np.array([0.0, 0.0, -1.0]),
)
ht_te_para = h_tail.project(
    np.linspace(np.array([31.5, -6.75, 6.0]), np.array([31.5, 6.75, 6.0]), tail_num_spanwise_quad),
    direction=np.array([0.0, 0.0, -1.0]),
)

if ENABLE_INEQUALITIES:
    fuselage_fri_collision_parametric = fuselage.project(geometry.evaluate(rotor_edges[6][1]).value)
    fuselage_fli_collision_parametric = fuselage.project(geometry.evaluate(rotor_edges[5][0]).value)

# ==============================================================================
# 3. Parameterization Solver & State Setup
# ==============================================================================
constant_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
linear_2_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
linear_3_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))

parameterization_solver = ParameterizationSolver()
geometric_variables = GeometricVariables()

# --- Wing Parameterization ---
wing_ffd_block = construct_ffd_block_around_entities(
    name="wing_ffd_block", entities=wing, num_coefficients=(2, 11, 2), degree=(1, 3, 1)
)
wing_sec = SectionalParameterization(
    name="wing_sec", parameterized_points=wing_ffd_block.coefficients, principal_parametric_dimension=1
)

wing_span_stretch = csdl.Variable(name="wing_span_stretch", value=0.0)
wing_root_chord_stretch = csdl.Variable(name="wing_root_chord_stretch", value=0.0)
wing_tip_chord_stretch = csdl.Variable(name="wing_tip_chord_stretch", value=0.0)
wing_translation_x = csdl.Variable(name="wing_translation_x", value=0.0)
wing_translation_z = csdl.Variable(name="wing_translation_z", value=0.0)

wing_tip_twist_dv = csdl.Variable(name="wing_tip_twist", value=0.0)  # in degrees
wing_tip_twist_rad = wing_tip_twist_dv * (np.pi / 180.0)

parameterization_solver.add_state(state=wing_span_stretch)
parameterization_solver.add_state(state=wing_root_chord_stretch)
parameterization_solver.add_state(state=wing_tip_chord_stretch)
parameterization_solver.add_state(state=wing_translation_x)
parameterization_solver.add_state(state=wing_translation_z)

wing_chord_bspline = lfs.Function(
    name="wing_chord_bspline",
    space=linear_3_space,
    coefficients=csdl.concatenate((wing_tip_chord_stretch, wing_root_chord_stretch, wing_tip_chord_stretch)),
)
wing_span_bspline = lfs.Function(
    name="wing_span_bspline",
    space=linear_2_space,
    coefficients=csdl.concatenate((-wing_span_stretch, wing_span_stretch)),
)
wing_tx_bspline = lfs.Function(name="wing_tx_bspline", space=constant_space, coefficients=wing_translation_x)
wing_tz_bspline = lfs.Function(name="wing_tz_bspline", space=constant_space, coefficients=wing_translation_z)

# Enforce root twist = 0.0 with tip twist at both extremities
wing_twist_bspline = lfs.Function(
    name="wing_twist_bspline",
    space=linear_3_space,
    coefficients=csdl.concatenate((wing_tip_twist_rad, csdl.Variable(value=0.0), wing_tip_twist_rad)),
)

sec_coords_11 = np.linspace(0.0, 1.0, wing_sec.num_sections).reshape((-1, 1))
wing_params = SectionalParameters()
wing_params.add_stretch(0, wing_chord_bspline.evaluate(sec_coords_11))
wing_params.add_translation(1, wing_span_bspline.evaluate(sec_coords_11))
wing_params.add_translation(0, wing_tx_bspline.evaluate(sec_coords_11))
wing_params.add_translation(2, wing_tz_bspline.evaluate(sec_coords_11))
wing_params.add_rotation(1, wing_twist_bspline.evaluate(sec_coords_11))

wing_ffd_coeffs = wing_sec.evaluate(wing_params, plot=False)
wing.set_coefficients(wing_ffd_block.evaluate_ffd(wing_ffd_coeffs, plot=False))

# --- Horizontal Tail Parameterization ---
h_tail_ffd_block = construct_ffd_block_around_entities(
    name="h_tail_ffd_block", entities=h_tail, num_coefficients=(2, 11, 2), degree=(1, 3, 1)
)
h_tail_sec = SectionalParameterization(
    name="h_tail_sec", parameterized_points=h_tail_ffd_block.coefficients, principal_parametric_dimension=1
)

h_tail_span_stretch = csdl.Variable(name="h_tail_span_stretch", value=0.0)
h_tail_root_chord_stretch = csdl.Variable(name="h_tail_root_chord_stretch", value=0.0)
h_tail_tip_chord_stretch = csdl.Variable(name="h_tail_tip_chord_stretch", value=0.0)
h_tail_translation_x = csdl.Variable(name="h_tail_translation_x", value=0.0)
h_tail_translation_z = csdl.Variable(name="h_tail_translation_z", value=0.0)

parameterization_solver.add_state(state=h_tail_span_stretch)
parameterization_solver.add_state(state=h_tail_root_chord_stretch)
parameterization_solver.add_state(state=h_tail_tip_chord_stretch)
parameterization_solver.add_state(state=h_tail_translation_x)
parameterization_solver.add_state(state=h_tail_translation_z)

h_tail_chord_bspline = lfs.Function(
    name="h_tail_chord_bspline",
    space=linear_3_space,
    coefficients=csdl.concatenate((h_tail_tip_chord_stretch, h_tail_root_chord_stretch, h_tail_tip_chord_stretch)),
)
h_tail_span_bspline = lfs.Function(
    name="h_tail_span_bspline",
    space=linear_2_space,
    coefficients=csdl.concatenate((-h_tail_span_stretch, h_tail_span_stretch)),
)
h_tail_tx_bspline = lfs.Function(name="h_tail_tx_bspline", space=constant_space, coefficients=h_tail_translation_x)
h_tail_tz_bspline = lfs.Function(name="h_tail_tz_bspline", space=constant_space, coefficients=h_tail_translation_z)

ht_params = SectionalParameters()
ht_params.add_stretch(0, h_tail_chord_bspline.evaluate(sec_coords_11))
ht_params.add_translation(1, h_tail_span_bspline.evaluate(sec_coords_11))
ht_params.add_translation(0, h_tail_tx_bspline.evaluate(sec_coords_11))
ht_params.add_translation(2, h_tail_tz_bspline.evaluate(sec_coords_11))

h_tail_ffd_coeffs = h_tail_sec.evaluate(ht_params, plot=False)
h_tail.set_coefficients(h_tail_ffd_block.evaluate_ffd(h_tail_ffd_coeffs, plot=False))

# --- Fuselage Parameterization ---
fuselage_ffd_block = construct_ffd_block_around_entities(
    name="fuselage_ffd_block", entities=[fuselage, nose_hub], num_coefficients=(3, 2, 2), degree=(1, 1, 1)
)
fuselage_sec = SectionalParameterization(
    name="fuselage_sec", parameterized_points=fuselage_ffd_block.coefficients, principal_parametric_dimension=0
)

cabin_stretch = csdl.Variable(name="cabin_stretch", value=0.0)
tail_boom_stretch = csdl.Variable(name="tail_boom_stretch", value=0.0)
cabin_width_stretch = csdl.Variable(name="cabin_width_stretch", value=0.0)
cabin_height_stretch = csdl.Variable(name="cabin_height_stretch", value=0.0)

parameterization_solver.add_state(state=cabin_stretch)
parameterization_solver.add_state(state=tail_boom_stretch)
parameterization_solver.add_state(state=cabin_width_stretch)
parameterization_solver.add_state(state=cabin_height_stretch)

fuse_tx = csdl.concatenate((csdl.Variable(value=0.0), cabin_stretch, cabin_stretch + tail_boom_stretch))
fuse_sy = csdl.concatenate((cabin_width_stretch, cabin_width_stretch, csdl.Variable(value=0.0)))
fuse_sz = csdl.concatenate((cabin_height_stretch, cabin_height_stretch, csdl.Variable(value=0.0)))

fuse_params = SectionalParameters()
fuse_params.add_translation(0, fuse_tx)
fuse_params.add_stretch(1, fuse_sy)
fuse_params.add_stretch(2, fuse_sz)

fuse_ffd_coeffs = fuselage_sec.evaluate(fuse_params, plot=False)
eval_fuse_coeffs = fuselage_ffd_block.evaluate_ffd(fuse_ffd_coeffs, plot=False)
fuselage.set_coefficients(eval_fuse_coeffs[0])
nose_hub.set_coefficients(eval_fuse_coeffs[1])

# --- Lift Rotors & Booms Parameterization ---
rotor_ffd_blocks = []
rotor_stretches = []
for i, comp_set in enumerate(rotor_components):
    r_ffd = construct_ffd_block_around_entities(
        name=f"{rotor_names[i]}_ffd", entities=comp_set, num_coefficients=(2, 2, 2), degree=(1, 1, 1)
    )
    r_sec = SectionalParameterization(
        name=f"{rotor_names[i]}_sec", parameterized_points=r_ffd.coefficients, principal_parametric_dimension=2
    )
    r_stretch = csdl.Variable(name=f"{rotor_names[i]}_stretch", value=0.0)
    parameterization_solver.add_state(state=r_stretch)
    r_bsp = lfs.Function(name=f"{rotor_names[i]}_bsp", space=constant_space, coefficients=r_stretch)

    r_params = SectionalParameters()
    sec_stretch = r_bsp.evaluate(np.linspace(0.0, 1.0, 2).reshape((-1, 1)))
    r_params.add_stretch(0, sec_stretch)
    r_params.add_stretch(1, sec_stretch)

    r_coeffs = r_sec.evaluate(r_params, plot=False)
    eval_r_coeffs = r_ffd.evaluate_ffd(r_coeffs, plot=False)
    for c_idx, comp in enumerate(comp_set):
        comp.set_coefficients(eval_r_coeffs[c_idx])

    # Rotor + boom rigid body translation (attached to wing)
    r_trans = csdl.Variable(name=f"{rotor_names[i]}_trans", shape=(3,), value=0.0)
    for comp in comp_set:
        for fn in comp.functions.values():
            fn.coefficients = fn.coefficients + csdl.expand(r_trans, fn.coefficients.shape, action="k->ijk")
    for fn in boom_components[i].functions.values():
        fn.coefficients = fn.coefficients + csdl.expand(r_trans, fn.coefficients.shape, action="k->ijk")
    parameterization_solver.add_state(state=r_trans)

# --- Cruise Rotor (Pusher Propeller) & Vertical Stabilizer Parameterization ---
pp_ffd = construct_ffd_block_around_entities(
    name="pp_ffd", entities=pp_components, num_coefficients=(2, 2, 2), degree=(1, 1, 1)
)
pp_sec = SectionalParameterization(
    name="pp_sec", parameterized_points=pp_ffd.coefficients, principal_parametric_dimension=0
)
cruise_rotor_stretch = csdl.Variable(name="cruise_rotor_stretch", value=0.0)
parameterization_solver.add_state(state=cruise_rotor_stretch)
pp_bsp = lfs.Function(name="pp_bsp", space=constant_space, coefficients=cruise_rotor_stretch)

pp_params = SectionalParameters()
sec_stretch_pp = pp_bsp.evaluate(np.linspace(0.0, 1.0, 2).reshape((-1, 1)))
pp_params.add_stretch(1, sec_stretch_pp)
pp_params.add_stretch(2, sec_stretch_pp)

pp_coeffs = pp_sec.evaluate(pp_params, plot=False)
eval_pp_coeffs = pp_ffd.evaluate_ffd(pp_coeffs, plot=False)
for c_idx, comp in enumerate(pp_components):
    comp.set_coefficients(eval_pp_coeffs[c_idx])

pp_trans = csdl.Variable(name="pp_trans", shape=(3,), value=0.0)
for comp in pp_components:
    for fn in comp.functions.values():
        fn.coefficients = fn.coefficients + csdl.expand(pp_trans, fn.coefficients.shape, action="k->ijk")
parameterization_solver.add_state(state=pp_trans)

vtail_trans = csdl.Variable(name="vtail_trans", shape=(3,), value=0.0)
for fn in v_tail.functions.values():
    fn.coefficients = fn.coefficients + csdl.expand(vtail_trans, fn.coefficients.shape, action="k->ijk")
parameterization_solver.add_state(state=vtail_trans)

# ==============================================================================
# 4. Computed Geometric Values & Target Design Variables
# ==============================================================================
# --- Wing Metrics ---
wing_span_comp = csdl.norm(geometry.evaluate(wing_le_right) - geometry.evaluate(wing_le_left))
wing_root_chord_comp = csdl.norm(geometry.evaluate(wing_te_center) - geometry.evaluate(wing_le_center))
wing_tip_chord_l_comp = csdl.norm(geometry.evaluate(wing_te_left) - geometry.evaluate(wing_le_left))
wing_tip_chord_r_comp = csdl.norm(geometry.evaluate(wing_te_right) - geometry.evaluate(wing_le_right))

wing_taper_comp = 0.5 * (wing_tip_chord_l_comp + wing_tip_chord_r_comp) / wing_root_chord_comp

w_le_eval = geometry.evaluate(wing_le_line_para)
w_te_eval = geometry.evaluate(wing_te_line_para)
chords_w = csdl.norm(w_te_eval - w_le_eval, axes=(1,))
mid_w = 0.5 * (w_le_eval + w_te_eval)
dl_w = csdl.norm(mid_w[1:] - mid_w[:-1], axes=(1,))
wing_area_comp = csdl.sum(0.5 * (chords_w[:-1] + chords_w[1:]) * dl_w)
wing_ar_comp = wing_span_comp**2 / wing_area_comp

# --- Tail Metrics ---
tail_span_comp = csdl.norm(geometry.evaluate(tail_le_right) - geometry.evaluate(tail_le_left))
tail_root_chord_comp = csdl.norm(geometry.evaluate(tail_te_center) - geometry.evaluate(tail_le_center))
tail_tip_chord_l_comp = csdl.norm(geometry.evaluate(tail_te_left) - geometry.evaluate(tail_le_left))
tail_tip_chord_r_comp = csdl.norm(geometry.evaluate(tail_te_right) - geometry.evaluate(tail_le_right))

tail_taper_comp = 0.5 * (tail_tip_chord_l_comp + tail_tip_chord_r_comp) / tail_root_chord_comp

t_le_eval = geometry.evaluate(ht_le_para)
t_te_eval = geometry.evaluate(ht_te_para)
chords_t = csdl.norm(t_te_eval - t_le_eval, axes=(1,))
mid_t = 0.5 * (t_le_eval + t_te_eval)
dl_t = csdl.norm(mid_t[1:] - mid_t[:-1], axes=(1,))
tail_area_comp = csdl.sum(0.5 * (chords_t[:-1] + chords_t[1:]) * dl_t)
tail_ar_comp = tail_span_comp**2 / tail_area_comp

# --- Fuselage Metrics ---
cabin_length_comp = csdl.norm(geometry.evaluate(fuselage_wing_te_center) - geometry.evaluate(fuselage_nose_points_parametric))
tail_moment_arm_comp = csdl.norm(geometry.evaluate(tail_qc) - geometry.evaluate(wing_qc))
cabin_height_comp = geometry.evaluate(fuselage_top_parametric)[2] - geometry.evaluate(fuselage_bottom_parametric)[2]
cabin_width_comp = geometry.evaluate(fuselage_right_parametric)[1] - geometry.evaluate(fuselage_left_parametric)[1]

# --- Cruise Rotor Metric ---
cruise_rotor_radius_comp = csdl.norm(geometry.evaluate(pp_disk_top) - geometry.evaluate(pp_disk_bot)) / 2.0

# --- Target Design Variables ---
wing_area_dv = csdl.Variable(name="wing_area", value=wing_area_comp.value)
wing_ar_dv = csdl.Variable(name="wing_aspect_ratio", value=wing_ar_comp.value)
wing_taper_dv = csdl.Variable(name="wing_taper_ratio", value=wing_taper_comp.value)

tail_area_dv = csdl.Variable(name="tail_area", value=tail_area_comp.value)
tail_ar_dv = csdl.Variable(name="tail_aspect_ratio", value=tail_ar_comp.value)
tail_taper_dv = csdl.Variable(name="tail_taper_ratio", value=tail_taper_comp.value)

cabin_length_dv = csdl.Variable(name="cabin_length", value=cabin_length_comp.value)
tail_moment_arm_dv = csdl.Variable(name="tail_moment_arm", value=tail_moment_arm_comp.value)
cabin_height_dv = csdl.Variable(name="cabin_height", value=cabin_height_comp.value)
cabin_width_dv = csdl.Variable(name="cabin_width", value=cabin_width_comp.value)
cruise_rotor_radius_dv = csdl.Variable(name="cruise_rotor_radius", value=cruise_rotor_radius_comp.value)

# Register Targets
geometric_variables.add_variable(wing_area_comp, wing_area_dv)
geometric_variables.add_variable(wing_ar_comp, wing_ar_dv)
geometric_variables.add_variable(wing_taper_comp, wing_taper_dv)

geometric_variables.add_variable(tail_area_comp, tail_area_dv)
geometric_variables.add_variable(tail_ar_comp, tail_ar_dv)
geometric_variables.add_variable(tail_taper_comp, tail_taper_dv)

geometric_variables.add_variable(cabin_length_comp, cabin_length_dv)
geometric_variables.add_variable(tail_moment_arm_comp, tail_moment_arm_dv)
geometric_variables.add_variable(cabin_height_comp, cabin_height_dv)
geometric_variables.add_variable(cabin_width_comp, cabin_width_dv)
geometric_variables.add_variable(cruise_rotor_radius_comp, cruise_rotor_radius_dv)

# --- Connection Invariants ---
wing_fuse_conn = geometry.evaluate(wing_te_center) - geometry.evaluate(fuselage_wing_te_center)
geometric_variables.add_variable(wing_fuse_conn[0], wing_fuse_conn.value[0])
geometric_variables.add_variable(wing_fuse_conn[2], wing_fuse_conn.value[2])

tail_fuse_conn = geometry.evaluate(tail_te_center) - geometry.evaluate(fuselage_tail_te_center)
geometric_variables.add_variable(tail_fuse_conn[0], tail_fuse_conn.value[0])
geometric_variables.add_variable(tail_fuse_conn[2], tail_fuse_conn.value[2])

vtail_fuse_pt = geometry.evaluate(v_tail.project(np.array([30.543, 0.0, 8.231])))
vtail_fuse_conn = geometry.evaluate(fuselage_rear_points_parametric) - vtail_fuse_pt
geometric_variables.add_variable(vtail_fuse_conn, vtail_fuse_conn.value)

pusher_fuse_conn = geometry.evaluate(fuselage_rear_points_parametric) - geometry.evaluate(fuselage_rear_point_on_pusher_disk_parametric)
geometric_variables.add_variable(pusher_fuse_conn, pusher_fuse_conn.value)

# --- Lift Rotor Radii and Boom Connections ---
rotor_radii_comp = []
rotor_radii_dv = []
for i in range(8):
    p1, p2 = rotor_edges[i]
    r_comp = csdl.norm(geometry.evaluate(p1) - geometry.evaluate(p2)) / 2.0
    r_dv = csdl.Variable(name=f"{rotor_names[i]}_radius", value=r_comp.value)
    geometric_variables.add_variable(r_comp, r_dv)
    rotor_radii_comp.append(r_comp)
    rotor_radii_dv.append(r_dv)

    b_conn = geometry.evaluate(boom_projections[i]) - geometry.evaluate(wing_boom_projections[i])
    geometric_variables.add_variable(b_conn, b_conn.value)

# --- Optional Inequality Constraints ---
if ENABLE_INEQUALITIES:
    inner_outer_min_dist = 0.4
    inner_fuse_min_dist = 0.4

    r_front_left_dist = geometry.evaluate(rotor_edges[5][1])[1] - geometry.evaluate(rotor_edges[4][0])[1]
    r_front_right_dist = geometry.evaluate(rotor_edges[7][1])[1] - geometry.evaluate(rotor_edges[6][0])[1]
    r_rear_left_dist = geometry.evaluate(rotor_edges[1][1])[1] - geometry.evaluate(rotor_edges[0][0])[1]
    r_rear_right_dist = geometry.evaluate(rotor_edges[3][1])[1] - geometry.evaluate(rotor_edges[2][0])[1]

    for dist, name in [
        (r_front_left_dist, "ineq_fl_rotor_rotor"),
        (r_front_right_dist, "ineq_fr_rotor_rotor"),
        (r_rear_left_dist, "ineq_rl_rotor_rotor"),
        (r_rear_right_dist, "ineq_rr_rotor_rotor"),
    ]:
        constraint = -(dist - inner_outer_min_dist)
        constraint.add_name(name)
        parameterization_solver.add_inequality_constraint(constraint=constraint, quadratic_penalty_factor=1.0e3, linear_penalty_factor=0.0)

    fri_fuse_dist = geometry.evaluate(rotor_edges[6][1])[1] - geometry.evaluate(fuselage_fri_collision_parametric)[1]
    fli_fuse_dist = geometry.evaluate(fuselage_fli_collision_parametric)[1] - geometry.evaluate(rotor_edges[5][0])[1]

    for dist, name in [(fri_fuse_dist, "ineq_fri_fuselage"), (fli_fuse_dist, "ineq_fli_fuselage")]:
        constraint = -(dist - inner_fuse_min_dist)
        constraint.add_name(name)
        parameterization_solver.add_inequality_constraint(constraint=constraint, quadratic_penalty_factor=1.0e3, linear_penalty_factor=0.0)

print("=== Parameterization Model Diagnostics ===")
num_states = sum(s.state.size for s in parameterization_solver.states)
num_constraints = sum(v.size for v in geometric_variables.computed_value)
print(f"Number of parameterization states:      {num_states}")
print(f"Number of equality constraints:         {num_constraints}")
print(f"Inequality constraints enabled:         {ENABLE_INEQUALITIES}")
print(f"Wing: Area = {wing_area_comp.value[0]:.2f} m², AR = {wing_ar_comp.value[0]:.2f}, Taper = {wing_taper_comp.value[0]:.4f}, Tip Twist = {wing_tip_twist_dv.value[0]:.1f}°")
print(f"Tail: Area = {tail_area_comp.value[0]:.2f} m², AR = {tail_ar_comp.value[0]:.2f}, Taper = {tail_taper_comp.value[0]:.4f}")
print(f"Fuselage: Cabin Length = {cabin_length_comp.value[0]:.2f} m, Tail Moment Arm = {tail_moment_arm_comp.value[0]:.2f} m")
print(f"Fuselage: Cabin Width = {cabin_width_comp.value[0]:.2f} m, Cabin Height = {cabin_height_comp.value[0]:.2f} m")
print(f"Cruise Rotor Radius: {cruise_rotor_radius_comp.value[0]:.2f} m")
print("=" * 45 + "\n")

# Baseline solve evaluation
t_eval_start = time.time()
parameterization_solver.evaluate(geometric_variables)
t_eval_end = time.time()
print(f"Initial parameterization solve completed in {t_eval_end - t_eval_start:.2f} seconds.\n")

# ==============================================================================
# 5. Shared Interface for Evaluation, Perturbations, and Benchmarking
# ==============================================================================
jax_inputs = [
    wing_area_dv,
    wing_ar_dv,
    wing_taper_dv,
    wing_tip_twist_dv,
    tail_area_dv,
    tail_ar_dv,
    tail_taper_dv,
    cabin_length_dv,
    tail_moment_arm_dv,
    cabin_height_dv,
    cabin_width_dv,
    cruise_rotor_radius_dv,
] + rotor_radii_dv

variable_names = [
    "wing_area",
    "wing_aspect_ratio",
    "wing_taper_ratio",
    "wing_tip_twist",
    "tail_area",
    "tail_aspect_ratio",
    "tail_taper_ratio",
    "cabin_length",
    "tail_moment_arm",
    "cabin_height",
    "cabin_width",
    "cruise_rotor_radius",
] + [f"{name}_radius" for name in rotor_names]

geometry_coefficients = [fn.coefficients for fn in geometry.functions.values()]
computed_metrics = [
    wing_area_comp,
    wing_ar_comp,
    wing_taper_comp,
    wing_tip_twist_dv,
    tail_area_comp,
    tail_ar_comp,
    tail_taper_comp,
    cabin_length_comp,
    tail_moment_arm_comp,
    cabin_height_comp,
    cabin_width_comp,
    cruise_rotor_radius_comp,
] + rotor_radii_comp

jax_outputs = geometry_coefficients + computed_metrics

video_components = [
    (wing, "#3498db", "Wing"),
    (h_tail, "#2ecc71", "Horizontal Tail"),
    (v_tail, "#1abc9c", "Vertical Tail"),
    (fuselage, "#95a5a6", "Fuselage"),
    (nose_hub, "#7f8c8d", "Nose Hub"),
]
for p in pp_components:
    video_components.append((p, "#e74c3c", "Pusher Propeller"))
for comp_set in rotor_components:
    for comp in comp_set:
        video_components.append((comp, "#e67e22", "Lift Rotor"))
for boom in boom_components:
    video_components.append((boom, "#f39c12", "Boom"))

# ==============================================================================
# 6. Perturbation Video Generation
# ==============================================================================
def generate_perturbation_video(sweep_paired_rotors=True):
    print("=== Starting Perturbation Video Generation Looping Over Variables ===")
    print("Compiling JAX Simulator for video generation...")
    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=jax_inputs,
        additional_outputs=jax_outputs,
        gpu=False,
    )
    sim.run()
    print("JAX Simulator compiled successfully!\n")

    output_dir = "examples/showcase_examples/lift_plus_cruise"
    os.makedirs(output_dir, exist_ok=True)
    video_filename = (
        "lift_plus_cruise_perturbations_with_inequalities.mp4"
        if ENABLE_INEQUALITIES
        else "lift_plus_cruise_perturbations.mp4"
    )
    video_path = os.path.join(output_dir, video_filename)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1088])
    fps = 10
    plotter.open_movie(video_path, framerate=fps)

    camera_pos = [(-50, -50, 40), (15, 0, 5), (0, 0, 1)]

    # Initial ghost reference meshes
    initial_meshes = []
    for comp, _, _ in video_components:
        comp_meshes = [
            elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem
            for elem in comp.plot(show=False)
        ]
        for m in comp_meshes:
            if m is not None:
                initial_meshes.append(m.copy())
    valid_ghost_meshes = [m for m in initial_meshes if m is not None]

    def render_frame(title_str, val_str, delta_str=""):
        plotter.clear()

        # 1. Reference ghost geometry overlay
        for gm in valid_ghost_meshes:
            plotter.add_mesh(
                gm,
                color="#B6B1A9",
                opacity=0.25,
                smooth_shading=True,
                show_edges=False,
            )

        # 2. Perturbed geometry
        for comp, col, name in video_components:
            comp_meshes = [
                elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem
                for elem in comp.plot(show=False)
            ]
            for m in comp_meshes:
                if m is not None:
                    plotter.add_mesh(
                        m,
                        color=col,
                        smooth_shading=True,
                        specular=0.5,
                        specular_power=20,
                        ambient=0.3,
                        diffuse=0.7,
                        show_edges=False,
                    )

        plotter.enable_lightkit()
        plotter.set_background("#12151c", top="#1e2330")

        hud_text = (
            f"LSDO_GEO: Lift+Cruise (LPC) Parameterization\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Design Variable: {title_str}\n"
            f"Current Value:   {val_str}\n"
            f"Perturbation:    {delta_str}\n"
            f"Ghost Reference: Initial CAD Geometry (gray, 0.25 opacity)"
        )
        plotter.add_text(
            hud_text,
            position="upper_left",
            font_size=12,
            color="white",
            font="courier",
            shadow=True,
        )

        plotter.camera_position = camera_pos
        plotter.write_frame()

    variables_to_sweep = [
        {"variables": [wing_area_dv], "name": "Wing Area", "unit": "m²", "nominal": wing_area_comp.value, "offset": wing_area_comp.value * 0.20},
        {"variables": [wing_ar_dv], "name": "Wing Aspect Ratio", "unit": "", "nominal": wing_ar_comp.value, "offset": wing_ar_comp.value * 0.20},
        {"variables": [wing_taper_dv], "name": "Wing Taper Ratio", "unit": "", "nominal": wing_taper_comp.value, "offset": wing_taper_comp.value * 0.35},
        {"variables": [wing_tip_twist_dv], "name": "Wing Tip Twist", "unit": "°", "nominal": 0.0, "offset": 4.0},
        {"variables": [tail_area_dv], "name": "Horizontal Tail Area", "unit": "m²", "nominal": tail_area_comp.value, "offset": tail_area_comp.value * 0.25},
        {"variables": [tail_ar_dv], "name": "Horizontal Tail Aspect Ratio", "unit": "", "nominal": tail_ar_comp.value, "offset": tail_ar_comp.value * 0.25},
        {"variables": [tail_taper_dv], "name": "Horizontal Tail Taper Ratio", "unit": "", "nominal": tail_taper_comp.value, "offset": tail_taper_comp.value * 0.25},
        {"variables": [cabin_length_dv], "name": "Fuselage Cabin Length", "unit": "m", "nominal": cabin_length_comp.value, "offset": cabin_length_comp.value * 0.15},
        {"variables": [tail_moment_arm_dv], "name": "Fuselage Tail Moment Arm", "unit": "m", "nominal": tail_moment_arm_comp.value, "offset": tail_moment_arm_comp.value * 0.20},
        {"variables": [cabin_height_dv], "name": "Fuselage Cabin Height", "unit": "m", "nominal": cabin_height_comp.value, "offset": cabin_height_comp.value * 0.15},
        {"variables": [cabin_width_dv], "name": "Fuselage Cabin Width", "unit": "m", "nominal": cabin_width_comp.value, "offset": cabin_width_comp.value * 0.15},
        {"variables": [cruise_rotor_radius_dv], "name": "Cruise Rotor Radius", "unit": "m", "nominal": cruise_rotor_radius_comp.value, "offset": cruise_rotor_radius_comp.value * 0.15},
    ]

    if sweep_paired_rotors:
        rotor_pairs = [
            ([rotor_radii_dv[4], rotor_radii_dv[7]], "Front Outer Rotor Radii", "flo & fro"),
            ([rotor_radii_dv[5], rotor_radii_dv[6]], "Front Inner Rotor Radii", "fli & fri"),
            ([rotor_radii_dv[1], rotor_radii_dv[2]], "Rear Inner Rotor Radii", "rli & rri"),
            ([rotor_radii_dv[0], rotor_radii_dv[3]], "Rear Outer Rotor Radii", "rlo & rro"),
        ]
        for vars_list, label, _ in rotor_pairs:
            nom = vars_list[0].value
            variables_to_sweep.append({
                "variables": vars_list,
                "name": label,
                "unit": "m",
                "nominal": nom,
                "offset": nom * 0.12,
            })
    else:
        for i, name in enumerate(rotor_names):
            nom = rotor_radii_dv[i].value
            variables_to_sweep.append({
                "variables": [rotor_radii_dv[i]],
                "name": f"Rotor Radius ({name.upper()})",
                "unit": "m",
                "nominal": nom,
                "offset": nom * 0.12,
            })

    def generate_sweep_values_with_hold(nominal, offset, num_osc_points=14, hold_points=2):
        nominal = float(np.squeeze(nominal))
        offset = float(np.squeeze(offset))
        hold_start = np.full(hold_points, nominal)
        osc = nominal + offset * np.sin(np.linspace(0, 2 * np.pi, num_osc_points, endpoint=False))
        hold_end = np.full(hold_points, nominal)
        return np.concatenate([hold_start, osc, hold_end])

    print(f"Total variables to sweep: {len(variables_to_sweep)}")
    print(f"Video output target: {video_path}")
    print("Rendering video frames...")

    for _ in range(6):
        render_frame("Baseline Solved Configuration", "All Variables at Nominal", "Nominal Hold Position")

    for var_idx, var_info in enumerate(variables_to_sweep):
        target_vars = var_info["variables"]
        var_name = var_info["name"]
        unit_str = var_info["unit"]
        nominal_val = float(np.squeeze(var_info["nominal"]))
        offset_val = float(np.squeeze(var_info["offset"]))

        print(f"  [{var_idx+1:2d}/{len(variables_to_sweep)}] Sweeping {var_name} (nominal = {nominal_val:.2f}{unit_str}, offset = ±{offset_val:.2f}{unit_str})...")

        sweep_values = generate_sweep_values_with_hold(nominal_val, offset_val, num_osc_points=14, hold_points=2)

        for val in sweep_values:
            for v in target_vars:
                sim[v] = np.array([val])
            sim.run()

            delta = val - nominal_val
            val_display = f"{val:.2f} {unit_str}".strip()
            delta_display = f"Δ = {delta:+.2f} {unit_str}".strip() if abs(delta) > 1e-4 else "Hold at Nominal"
            render_frame(var_name, val_display, delta_display)

        for v in target_vars:
            sim[v] = np.array([nominal_val])
        sim.run()

    for _ in range(6):
        render_frame("Nominal Return", "All Variables Verified", "Parameterization Complete")

    plotter.close()
    print(f"\nSuccessfully generated video: {video_path}")

    artifact_dir = "/home/andrewfletcher/.gemini/antigravity/brain/eeb769c4-0006-447e-a9ad-42b4822002c6"
    if os.path.exists(artifact_dir):
        dest_video = os.path.join(artifact_dir, video_filename)
        shutil.copy2(video_path, dest_video)
        print(f"Copied video to artifact directory: {dest_video}")

if __name__ == '__main__':
    generate_perturbation_video()
