import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time
import gc
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
from lsdo_geo import ParameterizationSolver, GeometricVariables
import pyvista as pv

print("================================================================================")
print("     LSDO_GEO: Quadruped Dimensionless Ratio Parameterization Showcase          ")
print("================================================================================")

# ==============================================================================
# 1. Start CSDL Recorder and Import Geometry
# ==============================================================================
recorder = csdl.Recorder(inline=True)
recorder.start()

cad_file = "examples/example_geometries/robot_dog.stp"
print(f"Importing {cad_file}...")
t0 = time.time()
geometry = lg.import_geometry(cad_file, parallelize=False)
print(f"Successfully imported {len(geometry.functions)} surfaces in {time.time() - t0:.2f}s.\n")


# Save initial CAD geometry plot meshes for reference ghost overlay in perturbation video
initial_meshes = [
    elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem
    for elem in geometry.plot(show=False)
]
valid_ghost_meshes = [m for m in initial_meshes if m is not None]

# ==============================================================================
# 2. Declare Components
# ==============================================================================
# In the imported STEP model, side 0 corresponds to Left (+y) and side 1 corresponds to Right (-y).
body = geometry.declare_component(function_search_names=['body'], name='body')

front_left_thigh = geometry.declare_component(function_search_names=['front_thighs, 0'], name='front_left_thigh')
front_right_thigh = geometry.declare_component(function_search_names=['front_thighs, 1'], name='front_right_thigh')
rear_left_thigh = geometry.declare_component(function_search_names=['back_thighs, 0'], name='back_left_thigh')
rear_right_thigh = geometry.declare_component(function_search_names=['back_thighs, 1'], name='back_right_thigh')

front_left_shank = geometry.declare_component(function_search_names=['front_calves, 0'], name='front_left_shank')
front_right_shank = geometry.declare_component(function_search_names=['front_calves, 1'], name='front_right_shank')
rear_left_shank = geometry.declare_component(function_search_names=['back_calves, 0'], name='back_left_shank')
rear_right_shank = geometry.declare_component(function_search_names=['back_calves, 1'], name='back_right_shank')

# Combined leg components for initial posture hip rotations
front_left_leg = geometry.declare_component(function_search_names=['front_thighs, 0', 'front_calves, 0'], name='front_left_leg')
front_right_leg = geometry.declare_component(function_search_names=['front_thighs, 1', 'front_calves, 1'], name='front_right_leg')
rear_left_leg = geometry.declare_component(function_search_names=['back_thighs, 0', 'back_calves, 0'], name='back_left_leg')
rear_right_leg = geometry.declare_component(function_search_names=['back_thighs, 1', 'back_calves, 1'], name='back_right_leg')

print("Declared components:")
print(f"  Body             : {len(body.functions)} surfaces")
print(f"  Front Thighs     : {len(front_left_thigh.functions) + len(front_right_thigh.functions)} surfaces (FL, FR)")
print(f"  Rear Thighs      : {len(rear_left_thigh.functions) + len(rear_right_thigh.functions)} surfaces (RL, RR)")
print(f"  Front Shanks     : {len(front_left_shank.functions) + len(front_right_shank.functions)} surfaces (FL, FR)")
print(f"  Rear Shanks      : {len(rear_left_shank.functions) + len(rear_right_shank.functions)} surfaces (RL, RR)")
print(f"  Total Surfaces   : {len(geometry.functions)}\n")

# ==============================================================================
# 3. Reference and Contact Point Projections
# ==============================================================================
print("Projecting reference joint and contact points...")

# Documented reference coordinates
p_hip_fl = np.array([0.43789617, 0.15250035, 0.65])
p_hip_fr = np.array([0.43789617, -0.15250035, 0.65])
p_hip_rl = np.array([-0.46300000, 0.15250035, 0.65])
p_hip_rr = np.array([-0.46300000, -0.15250035, 0.65])

p_knee_fl = np.array([0.43789617, 0.18800000, 0.35])
p_knee_fr = np.array([0.43789617, -0.18800000, 0.35])
p_knee_rl = np.array([-0.46300000, 0.18800000, 0.35])
p_knee_rr = np.array([-0.46300000, -0.18800000, 0.35])

p_foot_fl = np.array([0.43789617, 0.18800000, 0.00])
p_foot_fr = np.array([0.43789617, -0.18800000, 0.00])
p_foot_rl = np.array([-0.46300000, 0.18800000, 0.00])
p_foot_rr = np.array([-0.46300000, -0.18800000, 0.00])

p_body_front = np.array([0.55, 0.00, 0.50])
p_body_rear = np.array([-0.55, 0.00, 0.50])

# Projections onto body
para_body_front = body.project(p_body_front, projection_tolerance=1.e-6, plot=False)
para_body_rear = body.project(p_body_rear, projection_tolerance=1.e-6, plot=False)

para_body_hip_fl = body.project(p_hip_fl, projection_tolerance=1.e-2, plot=False)
para_body_hip_fr = body.project(p_hip_fr, projection_tolerance=1.e-2, plot=False)
para_body_hip_rl = body.project(p_hip_rl, projection_tolerance=1.e-2, plot=False)
para_body_hip_rr = body.project(p_hip_rr, projection_tolerance=1.e-2, plot=False)

# Projections onto thighs
para_thigh_hip_fl = front_left_thigh.project(p_hip_fl, projection_tolerance=1.e-6, plot=False)
para_thigh_hip_fr = front_right_thigh.project(p_hip_fr, projection_tolerance=1.e-6, plot=False)
para_thigh_hip_rl = rear_left_thigh.project(p_hip_rl, projection_tolerance=1.e-6, plot=False)
para_thigh_hip_rr = rear_right_thigh.project(p_hip_rr, projection_tolerance=1.e-6, plot=False)

para_thigh_knee_fl = front_left_thigh.project(p_knee_fl, projection_tolerance=1.e-6, plot=False)
para_thigh_knee_fr = front_right_thigh.project(p_knee_fr, projection_tolerance=1.e-6, plot=False)
para_thigh_knee_rl = rear_left_thigh.project(p_knee_rl, projection_tolerance=1.e-6, plot=False)
para_thigh_knee_rr = rear_right_thigh.project(p_knee_rr, projection_tolerance=1.e-6, plot=False)

# Projections onto shanks
para_shank_knee_fl = front_left_shank.project(p_knee_fl, projection_tolerance=1.e-6, plot=False)
para_shank_knee_fr = front_right_shank.project(p_knee_fr, projection_tolerance=1.e-6, plot=False)
para_shank_knee_rl = rear_left_shank.project(p_knee_rl, projection_tolerance=1.e-6, plot=False)
para_shank_knee_rr = rear_right_shank.project(p_knee_rr, projection_tolerance=1.e-6, plot=False)

para_shank_foot_fl = front_left_shank.project(p_foot_fl, projection_tolerance=1.e-6, plot=False)
para_shank_foot_fr = front_right_shank.project(p_foot_fr, projection_tolerance=1.e-6, plot=False)
para_shank_foot_rl = rear_left_shank.project(p_foot_rl, projection_tolerance=1.e-6, plot=False)
para_shank_foot_rr = rear_right_shank.project(p_foot_rr, projection_tolerance=1.e-6, plot=False)

# ==============================================================================
# 4. Measure Baseline Nominal Quantities and Capture Connection Vectors
# ==============================================================================
nom_body_length = float(np.squeeze(body.evaluate(para_body_front)[0].value - body.evaluate(para_body_rear)[0].value))
nom_scale_length = nom_body_length

nom_front_thigh_length = float(np.squeeze(np.linalg.norm(front_left_thigh.evaluate(para_thigh_hip_fl).value - front_left_thigh.evaluate(para_thigh_knee_fl).value)))
nom_rear_thigh_length = float(np.squeeze(np.linalg.norm(rear_left_thigh.evaluate(para_thigh_hip_rl).value - rear_left_thigh.evaluate(para_thigh_knee_rl).value)))
nom_front_shank_length = float(np.squeeze(np.linalg.norm(front_left_shank.evaluate(para_shank_knee_fl).value - front_left_shank.evaluate(para_shank_foot_fl).value)))
nom_rear_shank_length = float(np.squeeze(np.linalg.norm(rear_left_shank.evaluate(para_shank_knee_rl).value - rear_left_shank.evaluate(para_shank_foot_rl).value)))

nom_leg_length = nom_front_thigh_length + nom_front_shank_length
nom_leg_aspect_ratio = nom_front_thigh_length / nom_front_shank_length
nom_leg_to_body_ratio = nom_leg_length / nom_scale_length

nom_chassis_width = float(np.squeeze(body.evaluate(para_body_hip_fl).value[1] - body.evaluate(para_body_hip_fr).value[1]))
nom_chassis_aspect_ratio = nom_chassis_width / nom_scale_length

# Capture baseline connection vectors from imported CAD
baseline_fl_hip_conn = front_left_thigh.evaluate(para_thigh_hip_fl).value - body.evaluate(para_body_hip_fl).value
baseline_fr_hip_conn = front_right_thigh.evaluate(para_thigh_hip_fr).value - body.evaluate(para_body_hip_fr).value
baseline_rl_hip_conn = rear_left_thigh.evaluate(para_thigh_hip_rl).value - body.evaluate(para_body_hip_rl).value
baseline_rr_hip_conn = rear_right_thigh.evaluate(para_thigh_hip_rr).value - body.evaluate(para_body_hip_rr).value

baseline_fl_knee_conn = front_left_shank.evaluate(para_shank_knee_fl).value - front_left_thigh.evaluate(para_thigh_knee_fl).value
baseline_fr_knee_conn = front_right_shank.evaluate(para_shank_knee_fr).value - front_right_thigh.evaluate(para_thigh_knee_fr).value
baseline_rl_knee_conn = rear_left_shank.evaluate(para_shank_knee_rl).value - rear_left_thigh.evaluate(para_thigh_knee_rl).value
baseline_rr_knee_conn = rear_right_shank.evaluate(para_shank_knee_rr).value - rear_right_thigh.evaluate(para_thigh_knee_rr).value

nom_abduction_offset = float(np.squeeze(abs(baseline_fl_hip_conn[1])))
nom_abduction_offset_ratio = nom_abduction_offset / nom_leg_length

nom_initial_hip_angle_deg = 45.0
nom_initial_knee_angle_deg = -90.0

print("Imported CAD Nominal & Dimensionless Values:")
print(f"  Scale Length (Lb)             : {nom_scale_length:.6f} m")
print(f"  Leg Aspect Ratio (rho_leg)    : {nom_leg_aspect_ratio:.6f}  (L_thigh / L_shank)")
print(f"  Leg-to-Body Ratio (rho_body)  : {nom_leg_to_body_ratio:.6f}  ((L2 + L3) / Lb)")
print(f"  Chassis Aspect Ratio (alpha)  : {nom_chassis_aspect_ratio:.6f}  (Wb / Lb)")
print(f"  Abduction Offset Ratio (rho_ab: {nom_abduction_offset_ratio:.6f}  (L_ab / (L2 + L3))")
print(f"  Initial Hip Angle (theta_hip) : {nom_initial_hip_angle_deg:.2f}°")
print(f"  Initial Knee Angle (theta_knee: {nom_initial_knee_angle_deg:.2f}°\n")

# ==============================================================================
# 5. Seven Named Dimensionless Design Inputs
# ==============================================================================
scale_length_dv = csdl.Variable(value=nom_scale_length, name='scale_length')
leg_aspect_ratio_dv = csdl.Variable(value=nom_leg_aspect_ratio, name='leg_aspect_ratio')
leg_to_body_ratio_dv = csdl.Variable(value=nom_leg_to_body_ratio, name='leg_to_body_ratio')
chassis_aspect_ratio_dv = csdl.Variable(value=nom_chassis_aspect_ratio, name='chassis_aspect_ratio')
abduction_offset_ratio_dv = csdl.Variable(value=nom_abduction_offset_ratio, name='abduction_offset_ratio')
initial_hip_angle_dv = csdl.Variable(value=nom_initial_hip_angle_deg, name='initial_hip_angle')
initial_knee_angle_dv = csdl.Variable(value=nom_initial_knee_angle_deg, name='initial_knee_angle')

# Convert angle inputs from degrees to radians for geometric rotation operations
hip_angle_rad = initial_hip_angle_dv * (np.pi / 180.0)
knee_angle_rad = initial_knee_angle_dv * (np.pi / 180.0)

# ==============================================================================
# 6. FFD Blocks and Geometry Parameterization Maps
# ==============================================================================
# Trivariate 2x2x2 linear FFD blocks around body and each limb component
body_ffd = lg.construct_ffd_block_around_entities(body, num_coefficients=(2, 2, 2))

fl_thigh_ffd = lg.construct_ffd_block_around_entities(front_left_thigh, num_coefficients=(2, 2, 2))
fr_thigh_ffd = lg.construct_ffd_block_around_entities(front_right_thigh, num_coefficients=(2, 2, 2))
rl_thigh_ffd = lg.construct_ffd_block_around_entities(rear_left_thigh, num_coefficients=(2, 2, 2))
rr_thigh_ffd = lg.construct_ffd_block_around_entities(rear_right_thigh, num_coefficients=(2, 2, 2))

fl_shank_ffd = lg.construct_ffd_block_around_entities(front_left_shank, num_coefficients=(2, 2, 2))
fr_shank_ffd = lg.construct_ffd_block_around_entities(front_right_shank, num_coefficients=(2, 2, 2))
rl_shank_ffd = lg.construct_ffd_block_around_entities(rear_left_shank, num_coefficients=(2, 2, 2))
rr_shank_ffd = lg.construct_ffd_block_around_entities(rear_right_shank, num_coefficients=(2, 2, 2))

# FFD stretch states
body_stretch_x = csdl.Variable(value=0.0, name='body_stretch_x')
body_stretch_y = csdl.Variable(value=0.0, name='body_stretch_y')

fl_thigh_stretch = csdl.Variable(value=0.0, name='fl_thigh_stretch')
fr_thigh_stretch = csdl.Variable(value=0.0, name='fr_thigh_stretch')
rl_thigh_stretch = csdl.Variable(value=0.0, name='rl_thigh_stretch')
rr_thigh_stretch = csdl.Variable(value=0.0, name='rr_thigh_stretch')

fl_shank_stretch = csdl.Variable(value=0.0, name='fl_shank_stretch')
fr_shank_stretch = csdl.Variable(value=0.0, name='fr_shank_stretch')
rl_shank_stretch = csdl.Variable(value=0.0, name='rl_shank_stretch')
rr_shank_stretch = csdl.Variable(value=0.0, name='rr_shank_stretch')

# Body longitudinal x deformation: front (+x) moves +stretch_x/2, rear (-x) moves -stretch_x/2
# Body lateral y deformation: left (+y) moves +stretch_y/2, right (-y) moves -stretch_y/2
body_coeff = body_ffd.coefficients.set(csdl.slice[1, :, :, 0], body_ffd.coefficients[1, :, :, 0] + body_stretch_x / 2.0)
body_coeff = body_coeff.set(csdl.slice[0, :, :, 0], body_coeff[0, :, :, 0] - body_stretch_x / 2.0)
body_coeff = body_coeff.set(csdl.slice[:, 1, :, 1], body_coeff[:, 1, :, 1] + body_stretch_y / 2.0)
body_coeff = body_coeff.set(csdl.slice[:, 0, :, 1], body_coeff[:, 0, :, 1] - body_stretch_y / 2.0)
body.set_coefficients(body_ffd.evaluate_ffd(body_coeff))

# Thigh longitudinal deformation: move knee joint end (z=0, index 0 in dim 2) along -z
fl_th_coeff = fl_thigh_ffd.coefficients.set(csdl.slice[:, :, 0, 2], fl_thigh_ffd.coefficients[:, :, 0, 2] - fl_thigh_stretch)
front_left_thigh.set_coefficients(fl_thigh_ffd.evaluate_ffd(fl_th_coeff))

fr_th_coeff = fr_thigh_ffd.coefficients.set(csdl.slice[:, :, 0, 2], fr_thigh_ffd.coefficients[:, :, 0, 2] - fr_thigh_stretch)
front_right_thigh.set_coefficients(fr_thigh_ffd.evaluate_ffd(fr_th_coeff))

rl_th_coeff = rl_thigh_ffd.coefficients.set(csdl.slice[:, :, 0, 2], rl_thigh_ffd.coefficients[:, :, 0, 2] - rl_thigh_stretch)
rear_left_thigh.set_coefficients(rl_thigh_ffd.evaluate_ffd(rl_th_coeff))

rr_th_coeff = rr_thigh_ffd.coefficients.set(csdl.slice[:, :, 0, 2], rr_thigh_ffd.coefficients[:, :, 0, 2] - rr_thigh_stretch)
rear_right_thigh.set_coefficients(rr_thigh_ffd.evaluate_ffd(rr_th_coeff))

# shank longitudinal deformation: move foot contact end (z=0, index 0 in dim 2) along -z
fl_sh_coeff = fl_shank_ffd.coefficients.set(csdl.slice[:, :, 0, 2], fl_shank_ffd.coefficients[:, :, 0, 2] - fl_shank_stretch)
front_left_shank.set_coefficients(fl_shank_ffd.evaluate_ffd(fl_sh_coeff))

fr_sh_coeff = fr_shank_ffd.coefficients.set(csdl.slice[:, :, 0, 2], fr_shank_ffd.coefficients[:, :, 0, 2] - fr_shank_stretch)
front_right_shank.set_coefficients(fr_shank_ffd.evaluate_ffd(fr_sh_coeff))

rl_sh_coeff = rl_shank_ffd.coefficients.set(csdl.slice[:, :, 0, 2], rl_shank_ffd.coefficients[:, :, 0, 2] - rl_shank_stretch)
rear_left_shank.set_coefficients(rl_shank_ffd.evaluate_ffd(rl_sh_coeff))

rr_sh_coeff = rr_shank_ffd.coefficients.set(csdl.slice[:, :, 0, 2], rr_shank_ffd.coefficients[:, :, 0, 2] - rr_shank_stretch)
rear_right_shank.set_coefficients(rr_shank_ffd.evaluate_ffd(rr_sh_coeff))

# ==============================================================================
# 7. Kinematic Posture Rotations
# ==============================================================================
# Rotate each complete leg assembly about its own hip joint
rot_axis_y = np.array([0.0, 1.0, 0.0])

fl_hip_origin = front_left_leg.evaluate(para_thigh_hip_fl)
fr_hip_origin = front_right_leg.evaluate(para_thigh_hip_fr)
rl_hip_origin = rear_left_leg.evaluate(para_thigh_hip_rl)
rr_hip_origin = rear_right_leg.evaluate(para_thigh_hip_rr)

front_left_leg.rotate(rotation_origin=fl_hip_origin, axis_vector=rot_axis_y, angles=hip_angle_rad)
front_right_leg.rotate(rotation_origin=fr_hip_origin, axis_vector=rot_axis_y, angles=hip_angle_rad)
rear_left_leg.rotate(rotation_origin=rl_hip_origin, axis_vector=rot_axis_y, angles=hip_angle_rad)
rear_right_leg.rotate(rotation_origin=rr_hip_origin, axis_vector=rot_axis_y, angles=hip_angle_rad)

# Evaluate rotated knee points on thighs/legs after hip rotation
fl_knee_origin = front_left_thigh.evaluate(para_thigh_knee_fl)
fr_knee_origin = front_right_thigh.evaluate(para_thigh_knee_fr)
rl_knee_origin = rear_left_thigh.evaluate(para_thigh_knee_rl)
rr_knee_origin = rear_right_thigh.evaluate(para_thigh_knee_rr)

# Rotate shanks about their rotated knee joints
front_left_shank.rotate(rotation_origin=fl_knee_origin, axis_vector=rot_axis_y, angles=knee_angle_rad)
front_right_shank.rotate(rotation_origin=fr_knee_origin, axis_vector=rot_axis_y, angles=knee_angle_rad)
rear_left_shank.rotate(rotation_origin=rl_knee_origin, axis_vector=rot_axis_y, angles=knee_angle_rad)
rear_right_shank.rotate(rotation_origin=rr_knee_origin, axis_vector=rot_axis_y, angles=knee_angle_rad)

# ==============================================================================
# 8. Rigid Body Corrections
# ==============================================================================
# Body pitch correction about lateral y axis
body_pitch = csdl.Variable(value=0.0, name='body_pitch')
body.rotate(rotation_origin=np.array([0.0, 0.0, 0.0]), axis_vector=rot_axis_y, angles=body_pitch)

# Body vertical translation (heave) to position feet at z = 0
body_trans_z = csdl.Variable(value=-0.19, name='body_trans_z')
body_trans = csdl.concatenate([csdl.Variable(value=0.0), csdl.Variable(value=0.0), body_trans_z])
for f in body.functions.values():
    f.coefficients += csdl.expand(body_trans, f.coefficients.shape, 'i->jki')

# Component-level 3D translations for thighs to preserve hip connections
fl_th_trans = csdl.Variable(value=np.zeros(3), name='fl_th_trans')
fr_th_trans = csdl.Variable(value=np.zeros(3), name='fr_th_trans')
rl_th_trans = csdl.Variable(value=np.zeros(3), name='rl_th_trans')
rr_th_trans = csdl.Variable(value=np.zeros(3), name='rr_th_trans')

for f in front_left_thigh.functions.values(): f.coefficients += csdl.expand(fl_th_trans, f.coefficients.shape, 'i->jki')
for f in front_right_thigh.functions.values(): f.coefficients += csdl.expand(fr_th_trans, f.coefficients.shape, 'i->jki')
for f in rear_left_thigh.functions.values(): f.coefficients += csdl.expand(rl_th_trans, f.coefficients.shape, 'i->jki')
for f in rear_right_thigh.functions.values(): f.coefficients += csdl.expand(rr_th_trans, f.coefficients.shape, 'i->jki')

# Component-level 3D translations for shanks to preserve knee connections
fl_sh_trans = csdl.Variable(value=np.zeros(3), name='fl_sh_trans')
fr_sh_trans = csdl.Variable(value=np.zeros(3), name='fr_sh_trans')
rl_sh_trans = csdl.Variable(value=np.zeros(3), name='rl_sh_trans')
rr_sh_trans = csdl.Variable(value=np.zeros(3), name='rr_sh_trans')

for f in front_left_shank.functions.values(): f.coefficients += csdl.expand(fl_sh_trans, f.coefficients.shape, 'i->jki')
for f in front_right_shank.functions.values(): f.coefficients += csdl.expand(fr_sh_trans, f.coefficients.shape, 'i->jki')
for f in rear_left_shank.functions.values(): f.coefficients += csdl.expand(rl_sh_trans, f.coefficients.shape, 'i->jki')
for f in rear_right_shank.functions.values(): f.coefficients += csdl.expand(rr_sh_trans, f.coefficients.shape, 'i->jki')

# ==============================================================================
# 9. Geometric Output Metrics and Connection Quantities
# ==============================================================================
# Evaluated points
fl_th_h_pt = front_left_thigh.evaluate(para_thigh_hip_fl)
fl_th_k_pt = front_left_thigh.evaluate(para_thigh_knee_fl)
fr_th_h_pt = front_right_thigh.evaluate(para_thigh_hip_fr)
fr_th_k_pt = front_right_thigh.evaluate(para_thigh_knee_fr)

rl_th_h_pt = rear_left_thigh.evaluate(para_thigh_hip_rl)
rl_th_k_pt = rear_left_thigh.evaluate(para_thigh_knee_rl)
rr_th_h_pt = rear_right_thigh.evaluate(para_thigh_hip_rr)
rr_th_k_pt = rear_right_thigh.evaluate(para_thigh_knee_rr)

fl_sh_k_pt = front_left_shank.evaluate(para_shank_knee_fl)
fl_sh_f_pt = front_left_shank.evaluate(para_shank_foot_fl)
fr_sh_k_pt = front_right_shank.evaluate(para_shank_knee_fr)
fr_sh_f_pt = front_right_shank.evaluate(para_shank_foot_fr)

rl_sh_k_pt = rear_left_shank.evaluate(para_shank_knee_rl)
rl_sh_f_pt = rear_left_shank.evaluate(para_shank_foot_rl)
rr_sh_k_pt = rear_right_shank.evaluate(para_shank_knee_rr)
rr_sh_f_pt = rear_right_shank.evaluate(para_shank_foot_rr)

# Length and width metrics
body_len_comp = body.evaluate(para_body_front)[0] - body.evaluate(para_body_rear)[0]
chassis_width_comp = body.evaluate(para_body_hip_fl)[1] - body.evaluate(para_body_hip_fr)[1]

fl_thigh_len_comp = csdl.norm(fl_th_h_pt - fl_th_k_pt)
fr_thigh_len_comp = csdl.norm(fr_th_h_pt - fr_th_k_pt)
rl_thigh_len_comp = csdl.norm(rl_th_h_pt - rl_th_k_pt)
rr_thigh_len_comp = csdl.norm(rr_th_h_pt - rr_th_k_pt)

fl_shank_len_comp = csdl.norm(fl_sh_k_pt - fl_sh_f_pt)
fr_shank_len_comp = csdl.norm(fr_sh_k_pt - fr_sh_f_pt)
rl_shank_len_comp = csdl.norm(rl_sh_k_pt - rl_sh_f_pt)
rr_shank_len_comp = csdl.norm(rr_sh_k_pt - rr_sh_f_pt)

# Dimensionless ratio metrics computed directly from geometry
chassis_aspect_ratio_comp = chassis_width_comp / body_len_comp

fl_leg_aspect_ratio_comp = fl_thigh_len_comp / fl_shank_len_comp
fr_leg_aspect_ratio_comp = fr_thigh_len_comp / fr_shank_len_comp
rl_leg_aspect_ratio_comp = rl_thigh_len_comp / rl_shank_len_comp
rr_leg_aspect_ratio_comp = rr_thigh_len_comp / rr_shank_len_comp

fl_leg_to_body_ratio_comp = (fl_thigh_len_comp + fl_shank_len_comp) / body_len_comp
fr_leg_to_body_ratio_comp = (fr_thigh_len_comp + fr_shank_len_comp) / body_len_comp
rl_leg_to_body_ratio_comp = (rl_thigh_len_comp + rl_shank_len_comp) / body_len_comp
rr_leg_to_body_ratio_comp = (rr_thigh_len_comp + rr_shank_len_comp) / body_len_comp

# Joint connection vectors
conn_fl_hip = fl_th_h_pt - body.evaluate(para_body_hip_fl)
conn_fr_hip = fr_th_h_pt - body.evaluate(para_body_hip_fr)
conn_rl_hip = rl_th_h_pt - body.evaluate(para_body_hip_rl)
conn_rr_hip = rr_th_h_pt - body.evaluate(para_body_hip_rr)

conn_fl_knee = fl_sh_k_pt - fl_th_k_pt
conn_fr_knee = fr_sh_k_pt - fr_th_k_pt
conn_rl_knee = rl_sh_k_pt - rl_th_k_pt
conn_rr_knee = rr_sh_k_pt - rr_th_k_pt

fl_abduction_offset_ratio_comp = conn_fl_hip[1] / (fl_thigh_len_comp + fl_shank_len_comp)
fr_abduction_offset_ratio_comp = (-conn_fr_hip[1]) / (fr_thigh_len_comp + fr_shank_len_comp)
rl_abduction_offset_ratio_comp = conn_rl_hip[1] / (rl_thigh_len_comp + rl_shank_len_comp)
rr_abduction_offset_ratio_comp = (-conn_rr_hip[1]) / (rr_thigh_len_comp + rr_shank_len_comp)

# Foot heights
foot_fl_z = fl_sh_f_pt[2]
foot_fr_z = fr_sh_f_pt[2]
foot_rl_z = rl_sh_f_pt[2]
foot_rr_z = rr_sh_f_pt[2]

# ==============================================================================
# 10. Parameterization Solver Setup and Nominal Solve
# ==============================================================================
print("Configuring ParameterizationSolver...")
solver = ParameterizationSolver()

# Solver states
solver.add_state(body_stretch_x)
solver.add_state(body_stretch_y)
solver.add_state(fl_thigh_stretch)
solver.add_state(fr_thigh_stretch)
solver.add_state(rl_thigh_stretch)
solver.add_state(rr_thigh_stretch)
solver.add_state(fl_shank_stretch)
solver.add_state(fr_shank_stretch)
solver.add_state(rl_shank_stretch)
solver.add_state(rr_shank_stretch)

solver.add_state(body_pitch)
solver.add_state(body_trans_z)

solver.add_state(fl_th_trans)
solver.add_state(fr_th_trans)
solver.add_state(rl_th_trans)
solver.add_state(rr_th_trans)

solver.add_state(fl_sh_trans)
solver.add_state(fr_sh_trans)
solver.add_state(rl_sh_trans)
solver.add_state(rr_sh_trans)

# Hip connection alignment in X (longitudinal) and Z (vertical)
solver.add_equality_constraint(csdl.concatenate([conn_fl_hip[0], conn_fl_hip[2]]), np.array([baseline_fl_hip_conn[0], baseline_fl_hip_conn[2]]))
solver.add_equality_constraint(csdl.concatenate([conn_fr_hip[0], conn_fr_hip[2]]), np.array([baseline_fr_hip_conn[0], baseline_fr_hip_conn[2]]))
solver.add_equality_constraint(csdl.concatenate([conn_rl_hip[0], conn_rl_hip[2]]), np.array([baseline_rl_hip_conn[0], baseline_rl_hip_conn[2]]))
solver.add_equality_constraint(csdl.concatenate([conn_rr_hip[0], conn_rr_hip[2]]), np.array([baseline_rr_hip_conn[0], baseline_rr_hip_conn[2]]))

# Equality constraints for knee assembly
solver.add_equality_constraint(conn_fl_knee, baseline_fl_knee_conn)
solver.add_equality_constraint(conn_fr_knee, baseline_fr_knee_conn)
solver.add_equality_constraint(conn_rl_knee, baseline_rl_knee_conn)
solver.add_equality_constraint(conn_rr_knee, baseline_rr_knee_conn)

# Ground constraints
solver.add_equality_constraint(foot_fl_z, 0.0)
solver.add_equality_constraint(foot_rl_z, 0.0)

# Geometric design variables directly enforced implicitly by the solver
geom_vars = GeometricVariables()
geom_vars.add_variable(body_len_comp, scale_length_dv)
geom_vars.add_variable(chassis_aspect_ratio_comp, chassis_aspect_ratio_dv)

geom_vars.add_variable(fl_leg_aspect_ratio_comp, leg_aspect_ratio_dv)
geom_vars.add_variable(fr_leg_aspect_ratio_comp, leg_aspect_ratio_dv)
geom_vars.add_variable(rl_leg_aspect_ratio_comp, leg_aspect_ratio_dv)
geom_vars.add_variable(rr_leg_aspect_ratio_comp, leg_aspect_ratio_dv)

geom_vars.add_variable(fl_leg_to_body_ratio_comp, leg_to_body_ratio_dv)
geom_vars.add_variable(fr_leg_to_body_ratio_comp, leg_to_body_ratio_dv)
geom_vars.add_variable(rl_leg_to_body_ratio_comp, leg_to_body_ratio_dv)
geom_vars.add_variable(rr_leg_to_body_ratio_comp, leg_to_body_ratio_dv)

geom_vars.add_variable(fl_abduction_offset_ratio_comp, abduction_offset_ratio_dv)
geom_vars.add_variable(fr_abduction_offset_ratio_comp, abduction_offset_ratio_dv)
geom_vars.add_variable(rl_abduction_offset_ratio_comp, abduction_offset_ratio_dv)
geom_vars.add_variable(rr_abduction_offset_ratio_comp, abduction_offset_ratio_dv)

print("Solving baseline geometry parameterization...")
t_solv0 = time.time()
solver.evaluate(geom_vars)
print(f"Parameterization solver converged in {time.time() - t_solv0:.2f}s.\n")

# Diagnostics report
res_scale = abs(float(np.squeeze(body_len_comp.value)) - float(np.squeeze(scale_length_dv.value)))
res_chassis = abs(float(np.squeeze(chassis_aspect_ratio_comp.value)) - float(np.squeeze(chassis_aspect_ratio_dv.value)))

res_leg_ar_fl = abs(float(np.squeeze(fl_leg_aspect_ratio_comp.value)) - float(np.squeeze(leg_aspect_ratio_dv.value)))
res_leg_ar_fr = abs(float(np.squeeze(fr_leg_aspect_ratio_comp.value)) - float(np.squeeze(leg_aspect_ratio_dv.value)))
res_leg_ar_rl = abs(float(np.squeeze(rl_leg_aspect_ratio_comp.value)) - float(np.squeeze(leg_aspect_ratio_dv.value)))
res_leg_ar_rr = abs(float(np.squeeze(rr_leg_aspect_ratio_comp.value)) - float(np.squeeze(leg_aspect_ratio_dv.value)))

res_leg_body_fl = abs(float(np.squeeze(fl_leg_to_body_ratio_comp.value)) - float(np.squeeze(leg_to_body_ratio_dv.value)))
res_leg_body_fr = abs(float(np.squeeze(fr_leg_to_body_ratio_comp.value)) - float(np.squeeze(leg_to_body_ratio_dv.value)))
res_leg_body_rl = abs(float(np.squeeze(rl_leg_to_body_ratio_comp.value)) - float(np.squeeze(leg_to_body_ratio_dv.value)))
res_leg_body_rr = abs(float(np.squeeze(rr_leg_to_body_ratio_comp.value)) - float(np.squeeze(leg_to_body_ratio_dv.value)))

res_abduct_fl = abs(float(np.squeeze(fl_abduction_offset_ratio_comp.value)) - float(np.squeeze(abduction_offset_ratio_dv.value)))
res_abduct_fr = abs(float(np.squeeze(fr_abduction_offset_ratio_comp.value)) - float(np.squeeze(abduction_offset_ratio_dv.value)))
res_abduct_rl = abs(float(np.squeeze(rl_abduction_offset_ratio_comp.value)) - float(np.squeeze(abduction_offset_ratio_dv.value)))
res_abduct_rr = abs(float(np.squeeze(rr_abduction_offset_ratio_comp.value)) - float(np.squeeze(abduction_offset_ratio_dv.value)))

max_hip_res = max([
    np.linalg.norm((conn_fl_hip.value - baseline_fl_hip_conn)[[0, 2]]),
    np.linalg.norm((conn_fr_hip.value - baseline_fr_hip_conn)[[0, 2]]),
    np.linalg.norm((conn_rl_hip.value - baseline_rl_hip_conn)[[0, 2]]),
    np.linalg.norm((conn_rr_hip.value - baseline_rr_hip_conn)[[0, 2]]),
])

max_knee_res = max([
    np.linalg.norm(conn_fl_knee.value - baseline_fl_knee_conn),
    np.linalg.norm(conn_fr_knee.value - baseline_fr_knee_conn),
    np.linalg.norm(conn_rl_knee.value - baseline_rl_knee_conn),
    np.linalg.norm(conn_rr_knee.value - baseline_rr_knee_conn),
])

foot_heights = [abs(float(np.squeeze(v.value))) for v in [foot_fl_z, foot_fr_z, foot_rl_z, foot_rr_z]]

print("--------------------------------------------------------------------------------")
print("                     Baseline Parameterization Verification                     ")
print("--------------------------------------------------------------------------------")
print(f"  Scale Length (Lb)          : {float(np.squeeze(body_len_comp.value)):.6f} m   (Target: {float(np.squeeze(scale_length_dv.value)):.6f} m, Residual: {res_scale:.2e} m)")
print(f"  Chassis Aspect Ratio (alph): {float(np.squeeze(chassis_aspect_ratio_comp.value)):.6f}     (Target: {float(np.squeeze(chassis_aspect_ratio_dv.value)):.6f}, Residual: {res_chassis:.2e})")
print(f"  Leg Aspect Ratio FL/FR/RL/RR: [{res_leg_ar_fl:.1e}, {res_leg_ar_fr:.1e}, {res_leg_ar_rl:.1e}, {res_leg_ar_rr:.1e}] (Target: {float(np.squeeze(leg_aspect_ratio_dv.value)):.4f})")
print(f"  Leg-to-Body Ratio FL/FR/RL/R: [{res_leg_body_fl:.1e}, {res_leg_body_fr:.1e}, {res_leg_body_rl:.1e}, {res_leg_body_rr:.1e}] (Target: {float(np.squeeze(leg_to_body_ratio_dv.value)):.4f})")
print(f"  Abduction Offset Ratio FL..R: [{res_abduct_fl:.1e}, {res_abduct_fr:.1e}, {res_abduct_rl:.1e}, {res_abduct_rr:.1e}] (Target: {float(np.squeeze(abduction_offset_ratio_dv.value)):.4f})")
print(f"  Max Hip Alignment Error    : {max_hip_res:.2e} m")
print(f"  Max Knee Joint Error       : {max_knee_res:.2e} m")
print(f"  Foot Heights (FL,FR,RL,RR) : [{foot_heights[0]:.2e}, {foot_heights[1]:.2e}, {foot_heights[2]:.2e}, {foot_heights[3]:.2e}] m")
print("--------------------------------------------------------------------------------\n")

tol = 1e-5
assert res_scale <= tol, f"Scale length residual {res_scale:.2e} exceeds tolerance {tol}"
assert res_chassis <= tol, f"Chassis aspect ratio residual {res_chassis:.2e} exceeds tolerance {tol}"
assert max(res_leg_ar_fl, res_leg_ar_fr, res_leg_ar_rl, res_leg_ar_rr) <= tol, "Leg aspect ratio residual exceeds tolerance"
assert max(res_leg_body_fl, res_leg_body_fr, res_leg_body_rl, res_leg_body_rr) <= tol, "Leg-to-body ratio residual exceeds tolerance"
assert max(res_abduct_fl, res_abduct_fr, res_abduct_rl, res_abduct_rr) <= tol, "Abduction offset ratio residual exceeds tolerance"
assert max_hip_res <= tol, f"Hip connection residual {max_hip_res:.2e} exceeds tolerance {tol}"
assert max_knee_res <= tol, f"Knee connection residual {max_knee_res:.2e} exceeds tolerance {tol}"
assert max(foot_heights) <= tol, f"Foot height residual {max(foot_heights):.2e} exceeds tolerance {tol}"

# ==============================================================================
# 11. Module-Level Benchmark and Evaluation Interface
# ==============================================================================
jax_inputs = [
    scale_length_dv,
    leg_aspect_ratio_dv,
    leg_to_body_ratio_dv,
    chassis_aspect_ratio_dv,
    abduction_offset_ratio_dv,
    initial_hip_angle_dv,
    initial_knee_angle_dv,
]

variable_names = [
    'scale_length',
    'leg_aspect_ratio',
    'leg_to_body_ratio',
    'chassis_aspect_ratio',
    'abduction_offset_ratio',
    'initial_hip_angle',
    'initial_knee_angle',
]

units = [
    'm',
    '-',
    '-',
    '-',
    '-',
    '°',
    '°',
]

target_comp_pairs = [
    (scale_length_dv, body_len_comp),
    (chassis_aspect_ratio_dv, chassis_aspect_ratio_comp),
    (leg_aspect_ratio_dv, fl_leg_aspect_ratio_comp),
    (leg_aspect_ratio_dv, fr_leg_aspect_ratio_comp),
    (leg_aspect_ratio_dv, rl_leg_aspect_ratio_comp),
    (leg_aspect_ratio_dv, rr_leg_aspect_ratio_comp),
    (leg_to_body_ratio_dv, fl_leg_to_body_ratio_comp),
    (leg_to_body_ratio_dv, fr_leg_to_body_ratio_comp),
    (leg_to_body_ratio_dv, rl_leg_to_body_ratio_comp),
    (leg_to_body_ratio_dv, rr_leg_to_body_ratio_comp),
    (abduction_offset_ratio_dv, fl_abduction_offset_ratio_comp),
    (abduction_offset_ratio_dv, fr_abduction_offset_ratio_comp),
    (abduction_offset_ratio_dv, rl_abduction_offset_ratio_comp),
    (abduction_offset_ratio_dv, rr_abduction_offset_ratio_comp),
]

tracking_outputs = [
    conn_fl_hip,
    conn_fr_hip,
    conn_rl_hip,
    conn_rr_hip,
    conn_fl_knee,
    conn_fr_knee,
    conn_rl_knee,
    conn_rr_knee,
    foot_fl_z,
    foot_fr_z,
    foot_rl_z,
    foot_rr_z,
]

jax_outputs = (
    [f.coefficients for f in geometry.functions.values()]
    + [pair[0] for pair in target_comp_pairs]
    + [pair[1] for pair in target_comp_pairs]
    + tracking_outputs
)


# ==============================================================================
# 12. Perturbation Video Generation
# ==============================================================================
def generate_perturbation_video():
    print("=== Setting up JaxSimulator for Dimensionless Parameter Perturbations ===")
    print("Compiling JaxSimulator (XLA JIT graph compilation)...")
    t_jit0 = time.time()
    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=jax_inputs,
        additional_outputs=jax_outputs,
        gpu=False,
    )
    sim.run()
    print(f"JaxSimulator compiled and verified in {time.time() - t_jit0:.2f}s!\n")

    output_dir = "examples/showcase_examples/robot_dog"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "robot_dog_perturbations.mp4")

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1088])
    fps = 10
    plotter.open_movie(video_path, framerate=fps)

    # Oblique perspective camera framing the entire quadruped and ground plane
    camera_pos = (-1.80, -2.10, 1.30)
    focal_pt = (0.00, 0.00, 0.35)
    view_up = (0.0, 0.0, 1.0)

    # Component styling
    video_components = [
        (body, "#3498db", "Body"),
        (front_left_thigh, "#e67e22", "FL Thigh"),
        (front_right_thigh, "#e67e22", "FR Thigh"),
        (rear_left_thigh, "#d35400", "RL Thigh"),
        (rear_right_thigh, "#d35400", "RR Thigh"),
        (front_left_shank, "#2ecc71", "FL Shank"),
        (front_right_shank, "#2ecc71", "FR Shank"),
        (rear_left_shank, "#27ae60", "RL Shank"),
        (rear_right_shank, "#27ae60", "RR Shank"),
    ]

    # Ground plane mesh at z = 0
    ground_plane = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=2.4, j_size=1.8)

    def render_frame(title_str, val_str, delta_str=""):
        plotter.clear()

        # Ground plane at z = 0
        plotter.add_mesh(
            ground_plane,
            color="#222834",
            opacity=0.75,
            show_edges=True,
            edge_color="#364053",
            smooth_shading=False,
        )

        # 1. Initial CAD geometry reference ghost overlay (#B6B1A9 with 0.25 opacity)
        for gm in valid_ghost_meshes:
            plotter.add_mesh(
                gm,
                color="#B6B1A9",
                opacity=0.25,
                smooth_shading=True,
                show_edges=False,
            )

        # Update geometry function coefficients from simulator state
        for f in geometry.functions.values():
            f.coefficients.value = np.array(sim[f.coefficients])

        # 2. Current perturbed geometry with vivid component colors
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

        # Telemetry HUD card
        hud_text = (
            "LSDO_GEO: Quadruped Dimensionless Ratio Parameterization\n"
            "------------------------------------------------------------\n"
            f"Design Input:    {title_str}\n"
            f"Current Value:   {val_str}\n"
            f"Perturbation:    {delta_str}\n"
            "Hip Joints:      Preserved (Residual < 1e-5 m)\n"
            "Knee Joints:     Preserved (Residual < 1e-5 m)\n"
            "Ground Contact:  Locked at z = 0 (All 4 Feet)\n"
            "Ghost Reference: Initial CAD Geometry (gray, 0.25 opacity)"
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
        gc.collect()

    def evaluate_feasibility(target_dv, target_val):
        sim[target_dv] = target_val
        sim.run()

        # Check finiteness across all outputs
        for out in jax_outputs:
            v = np.array(sim[out])
            if not np.all(np.isfinite(v)):
                return False, "non-finite outputs"

        # Check tolerances
        tol_check = 1e-5
        r_scale = abs(float(np.squeeze(sim[body_len_comp])) - float(np.squeeze(sim[scale_length_dv])))
        r_chassis = abs(float(np.squeeze(sim[chassis_aspect_ratio_comp])) - float(np.squeeze(sim[chassis_aspect_ratio_dv])))

        r_leg_ar = max([
            abs(float(np.squeeze(sim[comp])) - float(np.squeeze(sim[leg_aspect_ratio_dv])))
            for comp in [fl_leg_aspect_ratio_comp, fr_leg_aspect_ratio_comp, rl_leg_aspect_ratio_comp, rr_leg_aspect_ratio_comp]
        ])
        r_leg_body = max([
            abs(float(np.squeeze(sim[comp])) - float(np.squeeze(sim[leg_to_body_ratio_dv])))
            for comp in [fl_leg_to_body_ratio_comp, fr_leg_to_body_ratio_comp, rl_leg_to_body_ratio_comp, rr_leg_to_body_ratio_comp]
        ])
        r_abduct = max([
            abs(float(np.squeeze(sim[comp])) - float(np.squeeze(sim[abduction_offset_ratio_dv])))
            for comp in [fl_abduction_offset_ratio_comp, fr_abduction_offset_ratio_comp, rl_abduction_offset_ratio_comp, rr_abduction_offset_ratio_comp]
        ])

        r_hip = max([
            np.linalg.norm((np.array(sim[conn_fl_hip]) - baseline_fl_hip_conn)[[0, 2]]),
            np.linalg.norm((np.array(sim[conn_fr_hip]) - baseline_fr_hip_conn)[[0, 2]]),
            np.linalg.norm((np.array(sim[conn_rl_hip]) - baseline_rl_hip_conn)[[0, 2]]),
            np.linalg.norm((np.array(sim[conn_rr_hip]) - baseline_rr_hip_conn)[[0, 2]]),
        ])

        r_knee = max([
            np.linalg.norm(np.array(sim[conn_fl_knee]) - baseline_fl_knee_conn),
            np.linalg.norm(np.array(sim[conn_fr_knee]) - baseline_fr_knee_conn),
            np.linalg.norm(np.array(sim[conn_rl_knee]) - baseline_rl_knee_conn),
            np.linalg.norm(np.array(sim[conn_rr_knee]) - baseline_rr_knee_conn),
        ])

        r_feet = max([abs(float(np.squeeze(sim[f]))) for f in [foot_fl_z, foot_fr_z, foot_rl_z, foot_rr_z]])

        max_res = max(r_scale, r_chassis, r_leg_ar, r_leg_body, r_abduct, r_hip, r_knee, r_feet)
        if max_res > tol_check:
            return False, f"residual {max_res:.2e} exceeds tolerance {tol_check:.2e}"

        return True, "feasible"

    sweep_definitions = [
        {
            'dv': scale_length_dv,
            'name': 'Scale Length (Lb)',
            'unit': 'm',
            'nominal': nom_scale_length,
            'is_angle': False,
            'candidates': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        },
        {
            'dv': leg_aspect_ratio_dv,
            'name': 'Leg Aspect Ratio (rho_leg)',
            'unit': '-',
            'nominal': nom_leg_aspect_ratio,
            'is_angle': False,
            'candidates': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        },
        {
            'dv': leg_to_body_ratio_dv,
            'name': 'Leg-to-Body Ratio (rho_body)',
            'unit': '-',
            'nominal': nom_leg_to_body_ratio,
            'is_angle': False,
            'candidates': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        },
        {
            'dv': chassis_aspect_ratio_dv,
            'name': 'Chassis Aspect Ratio (alpha)',
            'unit': '-',
            'nominal': nom_chassis_aspect_ratio,
            'is_angle': False,
            'candidates': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        },
        {
            'dv': abduction_offset_ratio_dv,
            'name': 'Abduction Offset Ratio (rho_ab)',
            'unit': '-',
            'nominal': nom_abduction_offset_ratio,
            'is_angle': False,
            'candidates': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        },
        {
            'dv': initial_hip_angle_dv,
            'name': 'Initial Hip Angle (theta_hip)',
            'unit': '°',
            'nominal': nom_initial_hip_angle_deg,
            'is_angle': True,
            'candidates': [5.0, 10.0, 15.0, 20.0, 25.0],
        },
        {
            'dv': initial_knee_angle_dv,
            'name': 'Initial Knee Angle (theta_knee)',
            'unit': '°',
            'nominal': nom_initial_knee_angle_deg,
            'is_angle': True,
            'candidates': [5.0, 10.0, 15.0, 20.0, 25.0],
        },
    ]

    print("Probing candidate perturbation ranges for all 7 design inputs...")
    accepted_ranges = []
    for sweep_info in sweep_definitions:
        var_dv = sweep_info['dv']
        var_name = sweep_info['name']
        nom = sweep_info['nominal']
        is_angle = sweep_info['is_angle']
        candidates = sweep_info['candidates']

        accepted_cand = None
        for cand in candidates:
            amp = cand * nom if not is_angle else cand
            ok_pos, msg_pos = evaluate_feasibility(var_dv, nom + amp)
            ok_neg, msg_neg = evaluate_feasibility(var_dv, nom - amp)

            # Reset back to nominal
            sim[var_dv] = nom
            sim.run()

            if ok_pos and ok_neg:
                accepted_cand = cand
            else:
                # Stop extending after the first failing candidate
                break

        if accepted_cand is None:
            raise RuntimeError(
                f"Candidate probing failed: smallest candidate for {var_name} could not be satisfied ({msg_pos if not ok_pos else msg_neg})."
            )

        max_amp = accepted_cand * nom if not is_angle else accepted_cand
        pct_str = f"{accepted_cand*100:.0f}%" if not is_angle else f"+/-{accepted_cand:.1f} deg"
        unit_label = sweep_info['unit'] if sweep_info['unit'] != '-' else ''
        print(f"  {var_name:32s}: accepted candidate = {pct_str} (±{max_amp:.4f} {unit_label})")
        accepted_ranges.append(max_amp)

    print("\nRendering perturbation video frames...")

    # Opening baseline hold (6 frames = 0.6s)
    for _ in range(6):
        render_frame("Baseline Configuration", "All 7 Variables at Nominal", "Nominal Hold Position")

    # Sweep each input independently while restoring nominal between sweeps
    for idx, sweep_info in enumerate(sweep_definitions):
        var_dv = sweep_info['dv']
        var_name = sweep_info['name']
        unit_str = sweep_info['unit']
        unit_display = f" {unit_str}" if unit_str != '-' else ""
        nom = sweep_info['nominal']
        amp = accepted_ranges[idx]

        print(f"  [{idx+1}/7] Rendering sweep for {var_name} (nominal = {nom:.4f}{unit_display}, amplitude = ±{amp:.4f}{unit_display})...")

        # Smooth sinusoidal cycle: nominal hold -> sine oscillation -> nominal hold
        hold_frames = 3
        osc_frames = 20
        t_arr = np.linspace(0.0, 2.0 * np.pi, osc_frames, endpoint=False)
        osc_values = nom + amp * np.sin(t_arr)

        sweep_vals = np.concatenate([
            np.full(hold_frames, nom),
            osc_values,
            np.full(hold_frames, nom),
        ])

        for val in sweep_vals:
            sim[var_dv] = val
            sim.run()

            delta = val - nom
            val_display = f"{val:.4f}{unit_display}".strip()
            delta_display = f"Offset = {delta:+.4f}{unit_display}".strip() if abs(delta) > 1e-5 else "Hold at Nominal"

            render_frame(var_name, val_display, delta_display)

        # Restore nominal value
        sim[var_dv] = nom
        sim.run()

    # Closing nominal hold (6 frames = 0.6s)
    for _ in range(6):
        render_frame("Nominal Return", "All 7 Inputs Reverified", "Parameterization Complete")

    plotter.close()
    print(f"\nPerturbation video successfully generated: {video_path}\n")

    # Final nominal solve verification
    print("Rechecking nominal solve at end...")
    for var_dv, sweep_info in zip(jax_inputs, sweep_definitions):
        sim[var_dv] = sweep_info['nominal']
    sim.run()

    ok_final, msg_final = evaluate_feasibility(jax_inputs[0], sweep_definitions[0]['nominal'])
    assert ok_final, f"Final nominal verification failed: {msg_final}"
    print("Final nominal configuration verified within tolerance 1e-5 m!\n")


if __name__ == "__main__":
    generate_perturbation_video()
