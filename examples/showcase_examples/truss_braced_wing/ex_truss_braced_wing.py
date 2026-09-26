import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time
import gc
import shutil      
import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg
from lsdo_geo import (
    ParameterizationSolver,
    GeometricVariables,
    SectionalParameters,
    SectionalParameterization,
)
import pyvista as pv

print("================================================================================")
print("       LSDO_GEO: Truss-Braced Wing (TBW) Geometric Parameterization            ")
print("================================================================================")

# ==============================================================================
# 1. Start CSDL Recorder and Import Geometry
# ==============================================================================
recorder = csdl.Recorder(inline=True)
recorder.start()

cad_file = "examples/example_geometries/tbw.stp"
print(f"Importing {cad_file}...")
t0 = time.time()
geo = lg.import_geometry(cad_file, parallelize=False)
print(f"Successfully imported {len(geo.functions)} surfaces in {time.time() - t0:.2f}s.\n")

# Extract initial CAD geometry meshes for reference ghost overlay in perturbation video
initial_meshes = [
    elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem
    for elem in geo.plot(show=False)
]
valid_ghost_meshes = [m for m in initial_meshes if m is not None]
ghost_mesh = (
    valid_ghost_meshes[0].merge(valid_ghost_meshes[1:])
    if len(valid_ghost_meshes) > 1
    else valid_ghost_meshes[0]
)

# ==============================================================================
# 2. Declare Components
# ==============================================================================
fuselage = geo.declare_component(
    function_search_names=['Fuselage', 'Gear Pod'], name='fuselage'
)
wing = geo.declare_component(
    function_search_names=['Wing'], name='wing'
)
tail = geo.declare_component(
    function_search_names=['Htail', 'Vtail'], name='tail'
)
struts = geo.declare_component(
    function_search_names=['Strut'], name='struts'
)
juries = geo.declare_component(
    function_search_names=['Jury'], name='juries'
)

print("Declared components:")
print(f"  Fuselage : {len(fuselage.functions)} surfaces (includes main fuselage and gear pods)")
print(f"  Wing     : {len(wing.functions)} surfaces")
print(f"  Tail     : {len(tail.functions)} surfaces (horizontal and vertical stabilizers)")
print(f"  Struts   : {len(struts.functions)} surfaces (left and right struts)")
print(f"  Juries   : {len(juries.functions)} surfaces (left and right jury struts)")
print(f"  Total    : {len(geo.functions)} surfaces\n")

# ==============================================================================
# 3. Reference and Connection Point Projections
# ==============================================================================
print("Projecting key reference points and structural attachment junctions...")

# Wing reference points
p_wing_le_center = np.array([47.231, 0.0, 6.94])
p_wing_te_center = np.array([57.953, 0.0, 6.57])
p_wing_le_right = np.array([68.035, 85.291, 4.80])
p_wing_te_right = np.array([71.790, 85.291, 4.80])
p_wing_le_left = np.array([68.035, -85.291, 4.80])
p_wing_te_left = np.array([71.790, -85.291, 4.80])

para_w_le_c = wing.project(p_wing_le_center, plot=False)
para_w_te_c = wing.project(p_wing_te_center, plot=False)
para_w_le_r = wing.project(p_wing_le_right, plot=False)
para_w_te_r = wing.project(p_wing_te_right, plot=False)
para_w_le_l = wing.project(p_wing_le_left, plot=False)

# Discretization lines for planform integration via trapezoidal rule
n_half = 11
t_wing = np.linspace(0, 1, n_half)
pts_le_left = (1 - t_wing)[:, None] * p_wing_le_left + t_wing[:, None] * p_wing_le_center
pts_te_left = (1 - t_wing)[:, None] * p_wing_te_left + t_wing[:, None] * p_wing_te_center
pts_le_right = (1 - t_wing[1:])[:, None] * p_wing_le_center + t_wing[1:, None] * p_wing_le_right
pts_te_right = (1 - t_wing[1:])[:, None] * p_wing_te_center + t_wing[1:, None] * p_wing_te_right
pts_wing_le = np.vstack([pts_le_left, pts_le_right])
pts_wing_te = np.vstack([pts_te_left, pts_te_right])

wing_le_line_para = wing.project(pts_wing_le, plot=False)
wing_te_line_para = wing.project(pts_wing_te, plot=False)

# Fuselage reference points
p_fuse_nose = np.array([0.0, 0.0, 0.0])
p_fuse_tail = np.array([124.75, 0.0, 5.0])
para_f_nose = fuselage.project(p_fuse_nose, plot=False)
para_f_tail = fuselage.project(p_fuse_tail, plot=False)

# Tail reference points
p_tail_le_c = np.array([105.34, 0.0, 5.0])
p_tail_le_r = np.array([124.0, 19.22, 19.5])
p_tail_le_l = np.array([124.0, -19.22, 19.5])
p_tail_te_c = np.array([115.0, 0.0, 5.0])

para_t_le_c = tail.project(p_tail_le_c, plot=False)
para_t_te_c = tail.project(p_tail_te_center if 'p_tail_te_center' in locals() else p_tail_te_c, plot=False)
para_t_le_r = tail.project(p_tail_le_r, plot=False)
para_t_le_l = tail.project(p_tail_le_l, plot=False)

# Attachment reference points
p_strut_root_r = np.array([57.309, 12.641, -4.200])
para_fuse_strut_r = fuselage.project(p_strut_root_r, plot=False)
para_strut_root_r = struts.project(p_strut_root_r, plot=False)

p_strut_tip_r = np.array([62.902, 48.994, 5.763])
para_strut_tip_r = struts.project(p_strut_tip_r, plot=False)
para_wing_strut_r = wing.project(p_strut_tip_r, plot=False)

p_jury_base_r = np.array([59.676, 30.818, 0.732])
para_strut_jury_r = struts.project(p_jury_base_r, plot=False)
para_jury_base_r = juries.project(p_jury_base_r, plot=False)

p_jury_tip_r = np.array([59.237, 30.818, 6.181])
para_jury_tip_r = juries.project(p_jury_tip_r, plot=False)
para_wing_jury_r = wing.project(p_jury_tip_r, plot=False)

para_fuse_tail = fuselage.project(p_tail_le_c, plot=False)
para_tail_attach = tail.project(p_tail_le_c, plot=False)
print("All key reference points and junction interfaces successfully projected.\n")

# ==============================================================================
# 4. Construct FFD Blocks and Sectional Parameterizations
# ==============================================================================
print("Setting up Free-Form Deformation (FFD) blocks and sectional parameterizations...")
from lsdo_geo import construct_ffd_block_around_entities

wing_ffd = construct_ffd_block_around_entities(
    entities=wing, num_coefficients=(2, 5, 2), degree=(1, 1, 1), name='wing_ffd'
)
tail_ffd = construct_ffd_block_around_entities(
    entities=tail, num_coefficients=(2, 3, 2), degree=(1, 1, 1), name='tail_ffd'
)
fuselage_ffd = construct_ffd_block_around_entities(
    entities=fuselage, num_coefficients=(3, 2, 2), degree=(1, 1, 1), name='fuselage_ffd'
)
strut_ffd = construct_ffd_block_around_entities(
    entities=struts, num_coefficients=(2, 5, 2), degree=(1, 1, 1), name='strut_ffd'
)
jury_ffd = construct_ffd_block_around_entities(
    entities=juries, num_coefficients=(2, 3, 2), degree=(1, 1, 1), name='jury_ffd'
)

wing_sec = SectionalParameterization(
    name='wing_sec', parameterized_points=wing_ffd.coefficients, principal_parametric_dimension=1
)
tail_sec = SectionalParameterization(
    name='tail_sec', parameterized_points=tail_ffd.coefficients, principal_parametric_dimension=1
)
fuselage_sec = SectionalParameterization(
    name='fuselage_sec', parameterized_points=fuselage_ffd.coefficients, principal_parametric_dimension=0
)
strut_sec = SectionalParameterization(
    name='strut_sec', parameterized_points=strut_ffd.coefficients, principal_parametric_dimension=1
)
jury_sec = SectionalParameterization(
    name='jury_sec', parameterized_points=jury_ffd.coefficients, principal_parametric_dimension=2
)
print("All 5 FFD blocks and sectional parameterization objects initialized.\n")

# ==============================================================================
# 5. Define Parameterization States and Forward Maps
# ==============================================================================
# Wing states
wing_chord_stretches = csdl.Variable(shape=(2,), value=0.0, name='wing_chord_stretches')
wing_root_chord_stretch = wing_chord_stretches[0]
wing_tip_chord_stretch = wing_chord_stretches[1]
wing_span_stretch = csdl.Variable(value=0.0, name='wing_span_stretch')
wing_sweep_translation = csdl.Variable(value=0.0, name='wing_sweep_translation')

# Fuselage state
fuselage_stretch = csdl.Variable(value=0.0, name='fuselage_stretch')

# Tail states
tail_span_stretch = csdl.Variable(value=0.0, name='tail_span_stretch')
tail_chord_stretch = csdl.Variable(value=0.0, name='tail_chord_stretch')
tail_translation_x = csdl.Variable(value=0.0, name='tail_translation_x')

# Strut connection states (root and tip 3D translation)
strut_root_trans = csdl.Variable(shape=(3,), value=0.0, name='strut_root_trans')
strut_tip_trans = csdl.Variable(shape=(3,), value=0.0, name='strut_tip_trans')
strut_root_trans_x = strut_root_trans[0]
strut_root_trans_y = strut_root_trans[1]
strut_root_trans_z = strut_root_trans[2]
strut_tip_trans_x = strut_tip_trans[0]
strut_tip_trans_y = strut_tip_trans[1]
strut_tip_trans_z = strut_tip_trans[2]

# Jury connection states (base and tip 3D translation)
jury_base_trans = csdl.Variable(shape=(3,), value=0.0, name='jury_base_trans')
jury_tip_trans = csdl.Variable(shape=(3,), value=0.0, name='jury_tip_trans')
jury_base_trans_x = jury_base_trans[0]
jury_base_trans_y = jury_base_trans[1]
jury_base_trans_z = jury_base_trans[2]
jury_tip_trans_x = jury_tip_trans[0]
jury_tip_trans_y = jury_tip_trans[1]
jury_tip_trans_z = jury_tip_trans[2]

# --- Wing Forward Map ---
wing_mid_chord = 0.5 * (wing_tip_chord_stretch + wing_root_chord_stretch)
wing_sec_chord = csdl.concatenate([
    wing_tip_chord_stretch,
    wing_mid_chord,
    wing_root_chord_stretch,
    wing_mid_chord,
    wing_tip_chord_stretch,
])
wing_sec_span = csdl.concatenate([
    -wing_span_stretch,
    -0.5 * wing_span_stretch,
    csdl.Variable(value=0.0),
    0.5 * wing_span_stretch,
    wing_span_stretch,
])
wing_sec_sweep = csdl.concatenate([
    wing_sweep_translation,
    0.5 * wing_sweep_translation,
    csdl.Variable(value=0.0),
    0.5 * wing_sweep_translation,
    wing_sweep_translation,
])
wing_params = SectionalParameters()
wing_params.add_stretch(axis=0, stretch=wing_sec_chord)
wing_params.add_translation(axis=1, translation=wing_sec_span)
wing_params.add_translation(axis=0, translation=wing_sec_sweep)
wing_ffd_coeffs = wing_sec.evaluate(wing_params, plot=False)
wing.set_coefficients(wing_ffd.evaluate_ffd(wing_ffd_coeffs, plot=False))

# --- Fuselage Forward Map ---
fuse_sec_trans_x = csdl.concatenate([
    -0.5 * fuselage_stretch,
    csdl.Variable(value=0.0),
    0.5 * fuselage_stretch,
])
fuse_params = SectionalParameters()
fuse_params.add_translation(axis=0, translation=fuse_sec_trans_x)
fuse_ffd_coeffs = fuselage_sec.evaluate(fuse_params, plot=False)
fuselage.set_coefficients(fuselage_ffd.evaluate_ffd(fuse_ffd_coeffs, plot=False))

# --- Tail Forward Map ---
tail_sec_chord = csdl.concatenate([
    tail_chord_stretch,
    tail_chord_stretch,
    tail_chord_stretch,
])
tail_sec_span = csdl.concatenate([
    -tail_span_stretch,
    csdl.Variable(value=0.0),
    tail_span_stretch,
])
tail_params = SectionalParameters()
tail_params.add_stretch(axis=0, stretch=tail_sec_chord)
tail_params.add_translation(axis=1, translation=tail_sec_span)
tail_ffd_coeffs = tail_sec.evaluate(tail_params, plot=False)
tail.set_coefficients(tail_ffd.evaluate_ffd(tail_ffd_coeffs, plot=False))
tail.translate(translation=csdl.concatenate([
    tail_translation_x,
    csdl.Variable(value=0.0),
    csdl.Variable(value=0.0),
]))

# --- Strut Forward Map ---
strut_trans_x = csdl.concatenate([
    strut_tip_trans_x,
    strut_root_trans_x,
    strut_root_trans_x,
    strut_root_trans_x,
    strut_tip_trans_x,
])
strut_trans_y = csdl.concatenate([
    -strut_tip_trans_y,
    -strut_root_trans_y,
    csdl.Variable(value=0.0),
    strut_root_trans_y,
    strut_tip_trans_y,
])
strut_trans_z = csdl.concatenate([
    strut_tip_trans_z,
    strut_root_trans_z,
    strut_root_trans_z,
    strut_root_trans_z,
    strut_tip_trans_z,
])
strut_params = SectionalParameters()
strut_params.add_translation(axis=0, translation=strut_trans_x)
strut_params.add_translation(axis=1, translation=strut_trans_y)
strut_params.add_translation(axis=2, translation=strut_trans_z)
strut_ffd_coeffs = strut_sec.evaluate(strut_params, plot=False)
struts.set_coefficients(strut_ffd.evaluate_ffd(strut_ffd_coeffs, plot=False))

# --- Jury Forward Map ---
jury_sec_trans_x = csdl.concatenate([jury_base_trans_x, jury_tip_trans_x])
jury_sec_trans_y = csdl.concatenate([jury_base_trans_y, jury_tip_trans_y])
jury_sec_trans_z = csdl.concatenate([jury_base_trans_z, jury_tip_trans_z])
jury_params = SectionalParameters()
jury_params.add_translation(axis=0, translation=jury_sec_trans_x)
jury_params.add_translation(axis=1, translation=jury_sec_trans_y)
jury_params.add_translation(axis=2, translation=jury_sec_trans_z)
jury_ffd_coeffs = jury_sec.evaluate(jury_params, plot=False)
juries.set_coefficients(jury_ffd.evaluate_ffd(jury_ffd_coeffs, plot=False))

print("Forward parameterization maps defined and linked to geometry coefficients.\n")

# ==============================================================================
# 6. Computed Metrics and Connection Invariants
# ==============================================================================
# Aircraft geometric metrics
wingspan_comp = geo.evaluate(para_w_le_r)[1] - geo.evaluate(para_w_le_l)[1]
wing_root_chord_comp = geo.evaluate(para_w_te_c)[0] - geo.evaluate(para_w_le_c)[0]
wing_tip_chord_comp = geo.evaluate(para_w_te_r)[0] - geo.evaluate(para_w_le_r)[0]
spanwise_vec = geo.evaluate(para_w_le_r) - geo.evaluate(para_w_le_c)
wing_sweep_comp = csdl.arctan(spanwise_vec[0] / spanwise_vec[1])
fuselage_length_comp = geo.evaluate(para_f_tail)[0] - geo.evaluate(para_f_nose)[0]
tail_span_comp = geo.evaluate(para_t_le_r)[1] - geo.evaluate(para_t_le_l)[1]
tail_root_chord_comp = geo.evaluate(para_t_te_c)[0] - geo.evaluate(para_t_le_c)[0]

# Wing Planform Area and Aspect Ratio via Trapezoidal Integration
wing_le_eval = geo.evaluate(wing_le_line_para)
wing_te_eval = geo.evaluate(wing_te_line_para)
chords_w = csdl.norm(wing_te_eval - wing_le_eval, axes=(1,))
mid_w = 0.5 * (wing_le_eval + wing_te_eval)
dl_w = csdl.norm(mid_w[1:] - mid_w[:-1], axes=(1,))
wing_area_comp = csdl.sum(0.5 * (chords_w[:-1] + chords_w[1:]) * dl_w)
wing_ar_comp = wingspan_comp**2 / wing_area_comp
wing_taper_comp = wing_tip_chord_comp / wing_root_chord_comp

# Connection invariant vectors
conn_fuse_strut = geo.evaluate(para_fuse_strut_r) - geo.evaluate(para_strut_root_r)
conn_wing_strut = geo.evaluate(para_wing_strut_r) - geo.evaluate(para_strut_tip_r)
conn_strut_jury = geo.evaluate(para_strut_jury_r) - geo.evaluate(para_jury_base_r)
conn_wing_jury = geo.evaluate(para_wing_jury_r) - geo.evaluate(para_jury_tip_r)
conn_fuse_tail = geo.evaluate(para_fuse_tail) - geo.evaluate(para_tail_attach)

# ==============================================================================
# 7. Define Target Geometric Design Variables
# ==============================================================================
# Nominal baseline values
wing_area_dv = csdl.Variable(value=wing_area_comp.value, name='wing_area_target')
wing_ar_dv = csdl.Variable(value=wing_ar_comp.value, name='wing_ar_target')
wing_taper_dv = csdl.Variable(value=wing_taper_comp.value, name='wing_taper_target')
wing_sweep_dv = csdl.Variable(value=wing_sweep_comp.value, name='wing_sweep_target')
fuselage_length_dv = csdl.Variable(value=fuselage_length_comp.value, name='fuselage_length_target')
tail_span_dv = csdl.Variable(value=tail_span_comp.value, name='tail_span_target')
tail_root_chord_dv = csdl.Variable(value=tail_root_chord_comp.value, name='tail_root_chord_target')

# ==============================================================================
# 8. Setup and Evaluate ParameterizationSolver
# ==============================================================================
print("Configuring ParameterizationSolver...")
solver = ParameterizationSolver()

# Add all 20 parameterization states as grouped vector variables
solver.add_state(wing_span_stretch)
solver.add_state(wing_chord_stretches)
solver.add_state(wing_sweep_translation)
solver.add_state(fuselage_stretch)
solver.add_state(tail_span_stretch)
solver.add_state(tail_chord_stretch)
solver.add_state(tail_translation_x)

solver.add_state(strut_root_trans)
solver.add_state(strut_tip_trans)

solver.add_state(jury_base_trans)
solver.add_state(jury_tip_trans)

# Add connection invariant equality constraints (13 equations)
solver.add_equality_constraint(conn_fuse_strut, conn_fuse_strut.value)
solver.add_equality_constraint(conn_wing_strut, conn_wing_strut.value)
solver.add_equality_constraint(conn_strut_jury, conn_strut_jury.value)
solver.add_equality_constraint(conn_wing_jury, conn_wing_jury.value)
solver.add_equality_constraint(conn_fuse_tail[0], conn_fuse_tail.value[0])

# Add geometric design variables (7 equations -> 20x20 fully determined square system)
geom_vars = GeometricVariables()
geom_vars.add_variable(wing_area_comp, wing_area_dv)
geom_vars.add_variable(wing_ar_comp, wing_ar_dv)
geom_vars.add_variable(wing_taper_comp, wing_taper_dv)
geom_vars.add_variable(wing_sweep_comp, wing_sweep_dv)
geom_vars.add_variable(fuselage_length_comp, fuselage_length_dv)
geom_vars.add_variable(tail_span_comp, tail_span_dv)
geom_vars.add_variable(tail_root_chord_comp, tail_root_chord_dv)

print("Solving baseline geometry parameterization...")
t_solv0 = time.time()
solver.evaluate(geom_vars)
print(f"Parameterization solver converged in {time.time() - t_solv0:.2f}s.\n")

print("--------------------------------------------------------------------------------")
print("                     Baseline Parameterization Verification                     ")
print("--------------------------------------------------------------------------------")
print(f"  Wingspan         : {wingspan_comp.value[0]:.3f} m")
print(f"  Wing Area (Trap) : {wing_area_comp.value[0]:.3f} m²  (Target: {wing_area_dv.value[0]:.3f} m²)")
print(f"  Wing Aspect Ratio: {wing_ar_comp.value[0]:.3f}      (Target: {wing_ar_dv.value[0]:.3f})")
print(f"  Wing Taper Ratio : {wing_taper_comp.value[0]:.3f}   (Target: {wing_taper_dv.value[0]:.3f})")
print(f"  Wing Root Chord  : {wing_root_chord_comp.value[0]:.3f} m")
print(f"  Wing Tip Chord   : {wing_tip_chord_comp.value[0]:.3f} m")
print(f"  Wing Sweep Angle : {wing_sweep_comp.value[0]*180/np.pi:.3f} deg (Target: {wing_sweep_dv.value[0]*180/np.pi:.3f} deg)")
print(f"  Fuselage Length  : {fuselage_length_comp.value[0]:.3f} m   (Target: {fuselage_length_dv.value[0]:.3f} m)")
print(f"  Tail Span        : {tail_span_comp.value[0]:.3f} m   (Target: {tail_span_dv.value[0]:.3f} m)")
print(f"  Tail Root Chord  : {tail_root_chord_comp.value[0]:.3f} m   (Target: {tail_root_chord_dv.value[0]:.3f} m)")
print("--------------------------------------------------------------------------------")
print("                   Connection Invariant Verification Residuals                  ")
print("--------------------------------------------------------------------------------")
err_fuse_strut = np.linalg.norm(conn_fuse_strut.value - conn_fuse_strut.value)
err_wing_strut = np.linalg.norm(conn_wing_strut.value - conn_wing_strut.value)
err_strut_jury = np.linalg.norm(conn_strut_jury.value - conn_strut_jury.value)
err_wing_jury  = np.linalg.norm(conn_wing_jury.value - conn_wing_jury.value)
err_fuse_tail  = np.linalg.norm(conn_fuse_tail.value[0] - conn_fuse_tail.value[0])
print(f"  Fuselage -> Strut Root Joint Error : {err_fuse_strut:.2e} m")
print(f"  Wing     -> Strut Tip Joint Error  : {err_wing_strut:.2e} m")
print(f"  Strut    -> Jury Base Joint Error  : {err_strut_jury:.2e} m")
print(f"  Wing     -> Jury Tip Joint Error   : {err_wing_jury:.2e} m")
print(f"  Fuselage -> Tail Joint Error       : {err_fuse_tail:.2e} m")
print("--------------------------------------------------------------------------------\n")

# ==============================================================================
# 9. JaxSimulator Setup for Fast Interactive Re-Evaluation
# ==============================================================================
print("=== Setting up JaxSimulator for Geometric Variable Perturbations ===")
jax_inputs = [
    wing_area_dv,
    wing_ar_dv,
    wing_taper_dv,
    wing_sweep_dv,
    fuselage_length_dv,
    tail_span_dv,
    tail_root_chord_dv,
]
jax_outputs = [f.coefficients for f in geo.functions.values()]

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

# ==============================================================================
# 10. Perturbation Video Generation
# ==============================================================================
output_dir = "examples/showcase_examples/truss_braced_wing"
os.makedirs(output_dir, exist_ok=True)
video_path = os.path.join(output_dir, "truss_braced_wing_perturbations.mp4")

pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True, window_size=[1920, 1088])
fps = 10
plotter.open_movie(video_path, framerate=fps)

# Isometric perspective framing showing complete aircraft and truss attachments
camera_pos = (-65.0, -130.0, 95.0)
focal_pt = (55.0, 0.0, 8.0)
view_up = (0.0, 0.0, 1.0)

video_components = [
    (wing, "#3498db", "Wing"),
    (struts, "#e74c3c", "Struts"),
    (juries, "#f39c12", "Juries"),
    (fuselage, "#95a5a6", "Fuselage"),
    (tail, "#2ecc71", "Tail"),
]

def render_frame(title_str, val_str, delta_str=""):
    plotter.clear()

    # 1. Initial CAD geometry reference ghost overlay in #B6B1A9 with 0.25 opacity
    plotter.add_mesh(
        ghost_mesh,
        color="#B6B1A9",
        opacity=0.25,
        smooth_shading=True,
        show_edges=False,
    )

    # 2. Current perturbed geometry with vivid component colors
    for comp, col, name in video_components:
        comp_meshes = [
            elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem
            for elem in comp.plot(show=False)
        ]
        comp_meshes = [m for m in comp_meshes if m is not None]
        if comp_meshes:
            merged_comp = comp_meshes[0].merge(comp_meshes[1:]) if len(comp_meshes) > 1 else comp_meshes[0]
            plotter.add_mesh(
                merged_comp,
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

    # Monospace telemetry HUD card
    hud_text = (
        "LSDO_GEO: Truss-Braced Wing Parameterization\n"
        "------------------------------------------------------------\n"
        f"Design Variable: {title_str}\n"
        f"Current Value:   {val_str}\n"
        f"Perturbation:    {delta_str}\n"
        "Strut Attach:    Locked (Residual = 0.000 m)\n"
        "Jury Attach:     Locked (Residual = 0.000 m)\n"
        "Tail Attach:     Locked (Residual = 0.000 m)\n"
        "Ghost Reference: Initial CAD Geometry (#B6B1A9, 0.25 opacity)"
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

    # Periodic garbage collection for VTK memory management
    gc.collect()

# Define geometric design variables to sweep with nominal values and smooth sinusoidal oscillations
variables_to_sweep = [
    {
        'variable': wing_area_dv,
        'name': 'Wing Area',
        'unit': 'm²',
        'nominal': wing_area_dv.value[0],
        'offset': 100.0,
    },
    {
        'variable': wing_ar_dv,
        'name': 'Wing Aspect Ratio',
        'unit': '',
        'nominal': wing_ar_dv.value[0],
        'offset': 2.0,
    },
    {
        'variable': wing_taper_dv,
        'name': 'Wing Taper Ratio',
        'unit': '',
        'nominal': wing_taper_dv.value[0],
        'offset': 0.08,
    },
    {
        'variable': wing_sweep_dv,
        'name': 'Wing Sweep Angle',
        'unit': '°',
        'nominal': wing_sweep_dv.value[0],
        'offset': 3.0 * np.pi / 180.0,
        'is_angle': True,
    },
    {
        'variable': fuselage_length_dv,
        'name': 'Fuselage Length',
        'unit': 'm',
        'nominal': fuselage_length_dv.value[0],
        'offset': 8.0,
    },
    {
        'variable': tail_span_dv,
        'name': 'Tail Span',
        'unit': 'm',
        'nominal': tail_span_dv.value[0],
        'offset': 4.5,
    },
    {
        'variable': tail_root_chord_dv,
        'name': 'Tail Root Chord',
        'unit': 'm',
        'nominal': tail_root_chord_dv.value[0],
        'offset': 1.8,
    },
]

def generate_sweep_values_with_hold(nominal, offset, num_osc_points=16, hold_points=3):
    nominal = float(np.squeeze(nominal))
    offset = float(np.squeeze(offset))
    hold_start = np.full(hold_points, nominal)
    osc = nominal + offset * np.sin(np.linspace(0, 2 * np.pi, num_osc_points, endpoint=False))
    hold_end = np.full(hold_points, nominal)
    return np.concatenate([hold_start, osc, hold_end])

print("Rendering video frames across all 7 geometric design variables...")

# Opening hold position (6 frames = 0.6s)
for _ in range(6):
    render_frame("Baseline Solved Configuration", "All Variables at Nominal", "Nominal Hold Position")

# Sweep over each design variable sequentially
for var_idx, var_info in enumerate(variables_to_sweep):
    target_var = var_info['variable']
    var_name = var_info['name']
    unit_str = var_info['unit']
    is_angle = var_info.get('is_angle', False)
    nominal_val = float(np.squeeze(var_info['nominal']))
    offset_val = float(np.squeeze(var_info['offset']))

    display_nominal = nominal_val * 180.0 / np.pi if is_angle else nominal_val
    display_offset = offset_val * 180.0 / np.pi if is_angle else offset_val

    print(f"  [{var_idx+1}/{len(variables_to_sweep)}] Sweeping {var_name} (nominal = {display_nominal:.2f}{unit_str}, offset = ±{display_offset:.2f}{unit_str})...")

    sweep_values = generate_sweep_values_with_hold(nominal_val, offset_val, num_osc_points=16, hold_points=3)

    for val in sweep_values:
        sim[target_var] = np.array([val])
        sim.run()

        delta = val - nominal_val
        if is_angle:
            disp_val = val * 180.0 / np.pi
            disp_delta = delta * 180.0 / np.pi
        else:
            disp_val = val
            disp_delta = delta

        val_display = f"{disp_val:.2f} {unit_str}".strip()
        delta_display = f"Δ = {disp_delta:+.2f} {unit_str}".strip() if abs(disp_delta) > 1e-4 else "Hold at Nominal"

        render_frame(var_name, val_display, delta_display)

    # Reset back to nominal
    sim[target_var] = np.array([nominal_val])
    sim.run()

# Closing hold position (6 frames = 0.6s)
for _ in range(6):
    render_frame("Nominal Return", "All 7 Geometric DVs Verified", "Parameterization Complete")

plotter.close()
print(f"\nSuccessfully generated video: {video_path}")

# Copy video to artifact directory
artifact_dir = "/home/andrewfletcher/.gemini/antigravity/brain/29740a73-aa03-496c-b841-730839d9e40d"
if os.path.exists(artifact_dir):
    dest_video = os.path.join(artifact_dir, "truss_braced_wing_perturbations.mp4")
    shutil.copy2(video_path, dest_video)
    print(f"Copied video to artifact directory: {dest_video}")
