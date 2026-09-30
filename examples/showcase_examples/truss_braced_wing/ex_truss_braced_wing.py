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
p_wing_le_mid_right = np.array([57.575, 49.280, 5.65])
p_wing_te_mid_right = np.array([67.088, 49.280, 5.51])
p_wing_le_mid_left = np.array([57.575, -49.280, 5.65])
p_wing_te_mid_left = np.array([67.088, -49.280, 5.51])

para_w_le_c = wing.project(p_wing_le_center, plot=False)
para_w_te_c = wing.project(p_wing_te_center, plot=False)
para_w_le_r = wing.project(p_wing_le_right, plot=False)
para_w_te_r = wing.project(p_wing_te_right, plot=False)
para_w_le_l = wing.project(p_wing_le_left, plot=False)
para_w_le_m_r = wing.project(p_wing_le_mid_right, plot=False)
para_w_te_m_r = wing.project(p_wing_te_mid_right, plot=False)

# Discretization lines for planform integration via trapezoidal rule (passing through Yehoudi break)
n_seg = 6
t_seg1 = np.linspace(0, 1, n_seg)
t_seg2 = np.linspace(0, 1, n_seg)[1:]

# Left half: tip -> yehoudi -> center
pts_le_l1 = (1 - t_seg1)[:, None] * p_wing_le_left + t_seg1[:, None] * p_wing_le_mid_left
pts_le_l2 = (1 - t_seg2)[:, None] * p_wing_le_mid_left + t_seg2[:, None] * p_wing_le_center
pts_le_left = np.vstack([pts_le_l1, pts_le_l2])

pts_te_l1 = (1 - t_seg1)[:, None] * p_wing_te_left + t_seg1[:, None] * p_wing_te_mid_left
pts_te_l2 = (1 - t_seg2)[:, None] * p_wing_te_mid_left + t_seg2[:, None] * p_wing_te_center
pts_te_left = np.vstack([pts_te_l1, pts_te_l2])

# Right half: center -> yehoudi -> tip
pts_le_r1 = (1 - t_seg1[1:])[:, None] * p_wing_le_center + t_seg1[1:, None] * p_wing_le_mid_right
pts_le_r2 = (1 - t_seg2)[:, None] * p_wing_le_mid_right + t_seg2[:, None] * p_wing_le_right
pts_le_right = np.vstack([pts_le_r1, pts_le_r2])

pts_te_r1 = (1 - t_seg1[1:])[:, None] * p_wing_te_center + t_seg1[1:, None] * p_wing_te_mid_right
pts_te_r2 = (1 - t_seg2)[:, None] * p_wing_te_mid_right + t_seg2[:, None] * p_wing_te_right
pts_te_right = np.vstack([pts_te_r1, pts_te_r2])

pts_wing_le = np.vstack([pts_le_left, pts_le_right])
pts_wing_te = np.vstack([pts_te_left, pts_te_right])

wing_le_line_para = wing.project(pts_wing_le, plot=False)
wing_te_line_para = wing.project(pts_wing_te, plot=False)

# Fuselage reference points
p_fuse_nose = np.array([0.0, 0.0, 0.0])
p_cabin_start = np.array([21.2075, 6.195835, 0.0])
p_cabin_end = np.array([95.289, 5.364, 0.300])
p_fuse_side_r = np.array([60.0, 6.19585, 0.0])
p_fuse_side_l = np.array([60.0, -6.19585, 0.0])

para_f_nose = fuselage.project(p_fuse_nose, plot=False)
para_f_cabin_start = fuselage.project(p_cabin_start, plot=False)
para_f_cabin_end = fuselage.project(p_cabin_end, plot=False)
para_f_side_r = fuselage.project(p_fuse_side_r, plot=False)
para_f_side_l = fuselage.project(p_fuse_side_l, plot=False)

# Tail reference points (rear point of vertical stabilizer interface with fuselage)
p_tail_fuse_attach = np.array([124.75, 0.0, 6.3])
p_tail_le_c = np.array([122.9735, 0.0, 19.8853])
p_tail_te_c = np.array([134.2873, 0.0, 19.9971])
p_tail_le_r = np.array([132.0029, 19.2166, 18.9998])
p_tail_te_r = np.array([135.9923, 19.2166, 18.9930])
p_tail_le_l = np.array([132.0029, -19.2166, 18.9998])

para_fuse_tail = fuselage.project(p_tail_fuse_attach, plot=False)
para_tail_attach = tail.project(p_tail_fuse_attach, plot=False)
para_t_le_c = tail.project(p_tail_le_c, plot=False)
para_t_te_c = tail.project(p_tail_te_c, plot=False)
para_t_le_r = tail.project(p_tail_le_r, plot=False)
para_t_te_r = tail.project(p_tail_te_r, plot=False)
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

print("All key reference points and junction interfaces successfully projected.\n")

# ==============================================================================
# 4. Construct FFD Blocks and Sectional Parameterizations
# ==============================================================================
print("Setting up Free-Form Deformation (FFD) blocks and sectional parameterizations...")
from lsdo_geo import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.ffd_block import FFDBlock
import lsdo_function_spaces as lfs

# Wing FFD with mid-span sections coinciding with the Yehoudi break
y_yehudi = 49.27960026
wing_pts = np.vstack([f.coefficients.value.reshape(-1, 3) for f in wing.functions.values()])
mins = np.min(wing_pts, axis=0)
maxs = np.max(wing_pts, axis=0)

y_sections = np.array([mins[1], -y_yehudi, 0.0, y_yehudi, maxs[1]])
v_sections = (y_sections - mins[1]) / (maxs[1] - mins[1])

wing_ffd_init_coeffs = np.zeros((2, 5, 2, 3))
for i, x in enumerate([mins[0], maxs[0]]):
    for j, y in enumerate(y_sections):
        for k, z in enumerate([mins[2], maxs[2]]):
            wing_ffd_init_coeffs[i, j, k, :] = [x, y, z]

knots_y = np.array([0.0, 0.0, v_sections[1], 0.5, v_sections[3], 1.0, 1.0])
knots_x = np.array([0.0, 0.0, 1.0, 1.0])
knots_z = np.array([0.0, 0.0, 1.0, 1.0])
knots = (knots_x, knots_y, knots_z)

wing_ffd_space = lfs.BSplineSpace(
    num_parametric_dimensions=3, degree=(1, 1, 1), coefficients_shape=(2, 5, 2), knots=knots
)
# wing_ffd_b_spline = lfs.Function(
#     space=wing_ffd_space, coefficients=wing_ffd_init_coeffs, name='wing_ffd_coefficients'
# )
wing_ffd = FFDBlock(
    space=wing_ffd_space, coefficients=wing_ffd_init_coeffs, name='wing_ffd', embedded_entities=[wing]
)

# Fuselage FFD with 4 sections: nose (0.0), wing junction (55.0), cabin end (95.289), tail tip (max)
fuse_pts = np.vstack([f.coefficients.value.reshape(-1, 3) for f in fuselage.functions.values()])
mins_f = np.min(fuse_pts, axis=0)
maxs_f = np.max(fuse_pts, axis=0)
x_cabin_end = 95.289
fuse_x_sections = np.array([mins_f[0], 55.0, x_cabin_end, maxs_f[0]])
fuse_u_sections = (fuse_x_sections - mins_f[0]) / (maxs_f[0] - mins_f[0])
fuse_ffd_init_coeffs = np.zeros((4, 2, 2, 3))
for i, x in enumerate(fuse_x_sections):
    for j, y in enumerate([mins_f[1], maxs_f[1]]):
        for k, z in enumerate([mins_f[2], maxs_f[2]]):
            fuse_ffd_init_coeffs[i, j, k, :] = [x, y, z]
f_knots_x = np.array([0.0, 0.0, fuse_u_sections[1], fuse_u_sections[2], 1.0, 1.0])
f_knots_y = np.array([0.0, 0.0, 1.0, 1.0])
f_knots_z = np.array([0.0, 0.0, 1.0, 1.0])
fuse_ffd_space = lfs.BSplineSpace(
    num_parametric_dimensions=3, degree=(1, 1, 1), coefficients_shape=(4, 2, 2),
    knots=(f_knots_x, f_knots_y, f_knots_z)
)
fuselage_ffd = FFDBlock(
    space=fuse_ffd_space, coefficients=fuse_ffd_init_coeffs, name='fuselage_ffd', embedded_entities=[fuselage]
)

tail_ffd = construct_ffd_block_around_entities(
    entities=tail, num_coefficients=(2, 3, 2), degree=(1, 1, 1), name='tail_ffd'
)
# Strut FFD with sections at root and tip for uniform linear stretching: [-y_tip, -y_root, 0, y_root, y_tip]
s_pts = np.vstack([f.coefficients.value.reshape(-1, 3) for f in struts.functions.values()])
mins_s = np.min(s_pts, axis=0)
maxs_s = np.max(s_pts, axis=0)
y_strut_root = 12.641
y_sections_s = np.array([mins_s[1], -y_strut_root, 0.0, y_strut_root, maxs_s[1]])
v_sections_s = (y_sections_s - mins_s[1]) / (maxs_s[1] - mins_s[1])

strut_ffd_init_coeffs = np.zeros((2, 5, 2, 3))
for i, x in enumerate([mins_s[0], maxs_s[0]]):
    for j, y in enumerate(y_sections_s):
        for k, z in enumerate([mins_s[2], maxs_s[2]]):
            strut_ffd_init_coeffs[i, j, k, :] = [x, y, z]

knots_y_s = np.array([0.0, 0.0, v_sections_s[1], 0.5, v_sections_s[3], 1.0, 1.0])
strut_space = lfs.BSplineSpace(num_parametric_dimensions=3, degree=(1, 1, 1), coefficients_shape=(2, 5, 2), knots=(knots_x, knots_y_s, knots_z))
strut_ffd = FFDBlock(space=strut_space, coefficients=strut_ffd_init_coeffs, name='strut_ffd', embedded_entities=[struts])
jury_ffd = construct_ffd_block_around_entities(
    entities=juries, num_coefficients=(2, 2, 2), degree=(1, 1, 1), name='jury_ffd'
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
    name='jury_sec', parameterized_points=jury_ffd.coefficients, principal_parametric_dimension=1
)
print("All 5 FFD blocks and sectional parameterization objects initialized.\n")

# ==============================================================================
# 5. Define Parameterization States and Forward Maps
# ==============================================================================
# Wing states
wing_chord_stretches = csdl.Variable(shape=(3,), value=0.0, name='wing_chord_stretches')
wing_root_chord_stretch = wing_chord_stretches[0]
wing_mid_chord_stretch = wing_chord_stretches[1]
wing_tip_chord_stretch = wing_chord_stretches[2]
wing_span_stretch = csdl.Variable(value=0.0, name='wing_span_stretch')
wing_inboard_sweep_trans = csdl.Variable(value=0.0, name='wing_inboard_sweep_trans')
wing_outboard_sweep_trans = csdl.Variable(value=0.0, name='wing_outboard_sweep_trans')

# Fuselage states
fuse_cabin_stretch = csdl.Variable(value=0.0, name='fuse_cabin_stretch')
fuse_tail_stretch = csdl.Variable(value=0.0, name='fuse_tail_stretch')
fuse_radius_stretch = csdl.Variable(value=0.0, name='fuse_radius_stretch')

# Tail states (matching wing style: span stretch, root/tip chord stretches, and x/z translations)
tail_chord_stretches = csdl.Variable(shape=(2,), value=0.0, name='tail_chord_stretches')
tail_root_chord_stretch = tail_chord_stretches[0]
tail_tip_chord_stretch = tail_chord_stretches[1]
tail_span_stretch = csdl.Variable(value=0.0, name='tail_span_stretch')
tail_trans_xz = csdl.Variable(shape=(2,), value=0.0, name='tail_trans_xz')
tail_translation_x = tail_trans_xz[0]
tail_translation_z = tail_trans_xz[1]

# Strut connection states (root and tip 3D translation)
strut_root_trans = csdl.Variable(shape=(3,), value=0.0, name='strut_root_trans')
strut_tip_trans = csdl.Variable(shape=(3,), value=0.0, name='strut_tip_trans')
strut_root_trans_x = strut_root_trans[0]
strut_root_trans_y = strut_root_trans[1]
strut_root_trans_z = strut_root_trans[2]
strut_tip_trans_x = strut_tip_trans[0]
strut_tip_trans_y = strut_tip_trans[1]
strut_tip_trans_z = strut_tip_trans[2]

# Jury connection states (fixed to strut: entire jury translates rigidly with strut attachment)
jury_trans = csdl.Variable(shape=(3,), value=0.0, name='jury_trans')

# --- Wing Forward Map ---
eta_mid = float(y_yehudi / maxs[1])
wing_sec_chord = csdl.concatenate([
    wing_tip_chord_stretch,
    wing_mid_chord_stretch,
    wing_root_chord_stretch,
    wing_mid_chord_stretch,
    wing_tip_chord_stretch,
])
wing_sec_span = csdl.concatenate([
    -wing_span_stretch,
    -eta_mid * wing_span_stretch,
    csdl.Variable(value=0.0),
    eta_mid * wing_span_stretch,
    wing_span_stretch,
])
wing_sec_sweep = csdl.concatenate([
    wing_inboard_sweep_trans + wing_outboard_sweep_trans,
    wing_inboard_sweep_trans,
    csdl.Variable(value=0.0),
    wing_inboard_sweep_trans,
    wing_inboard_sweep_trans + wing_outboard_sweep_trans,
])
wing_params = SectionalParameters()
wing_params.add_stretch(axis=0, stretch=wing_sec_chord)
wing_params.add_translation(axis=1, translation=wing_sec_span)
wing_params.add_translation(axis=0, translation=wing_sec_sweep)
wing_ffd_coeffs = wing_sec.evaluate(wing_params, plot=False)
wing.set_coefficients(wing_ffd.evaluate_ffd(wing_ffd_coeffs, plot=False))

# --- Fuselage Forward Map ---
fuse_sec_trans_x = csdl.concatenate([
    csdl.Variable(value=0.0),
    csdl.Variable(value=0.0),
    fuse_cabin_stretch,
    fuse_cabin_stretch + fuse_tail_stretch,
])
fuse_sec_radius = csdl.concatenate([
    fuse_radius_stretch,
    fuse_radius_stretch,
    fuse_radius_stretch,
    fuse_radius_stretch,
])
fuse_params = SectionalParameters()
fuse_params.add_translation(axis=0, translation=fuse_sec_trans_x)
fuse_params.add_stretch(axis=1, stretch=fuse_sec_radius)
fuse_params.add_stretch(axis=2, stretch=fuse_sec_radius)
fuse_ffd_coeffs = fuselage_sec.evaluate(fuse_params, plot=False)
fuselage.set_coefficients(fuselage_ffd.evaluate_ffd(fuse_ffd_coeffs, plot=False))

# --- Tail Forward Map ---
tail_sec_chord = csdl.concatenate([
    tail_tip_chord_stretch,
    tail_root_chord_stretch,
    tail_tip_chord_stretch,
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
    tail_translation_z,
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
# Jury is completely fixed to the strut (translates rigidly with the strut attachment point)
jury_sec_trans_x = csdl.concatenate([jury_trans[0], jury_trans[0]])
jury_sec_trans_y = csdl.concatenate([-jury_trans[1], jury_trans[1]])
jury_sec_trans_z = csdl.concatenate([jury_trans[2], jury_trans[2]])
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
wing_mid_chord_comp = geo.evaluate(para_w_te_m_r)[0] - geo.evaluate(para_w_le_m_r)[0]
wing_tip_chord_comp = geo.evaluate(para_w_te_r)[0] - geo.evaluate(para_w_le_r)[0]

inboard_spanwise_vec = geo.evaluate(para_w_le_m_r) - geo.evaluate(para_w_le_c)
wing_inboard_sweep_comp = csdl.arctan(inboard_spanwise_vec[0] / inboard_spanwise_vec[1])
outboard_spanwise_vec = geo.evaluate(para_w_le_r) - geo.evaluate(para_w_le_m_r)
wing_outboard_sweep_comp = csdl.arctan(outboard_spanwise_vec[0] / outboard_spanwise_vec[1])

# Fuselage geometric metrics
cabin_length_comp = geo.evaluate(para_f_cabin_end)[0] - geo.evaluate(para_f_cabin_start)[0]
tail_moment_arm_comp = geo.evaluate(para_t_le_c)[0] - geo.evaluate(para_w_le_c)[0]
fuselage_radius_comp = 0.5 * (geo.evaluate(para_f_side_r)[1] - geo.evaluate(para_f_side_l)[1])

# Tail geometric metrics
tail_span_comp = geo.evaluate(para_t_le_r)[1] - geo.evaluate(para_t_le_l)[1]
tail_root_chord_comp = geo.evaluate(para_t_te_c)[0] - geo.evaluate(para_t_le_c)[0]
tail_tip_chord_comp = geo.evaluate(para_t_te_r)[0] - geo.evaluate(para_t_le_r)[0]
tail_taper_comp = tail_tip_chord_comp / tail_root_chord_comp
tail_area_comp = 0.5 * (tail_root_chord_comp + tail_tip_chord_comp) * tail_span_comp
tail_ar_comp = tail_span_comp**2 / tail_area_comp

# Wing Planform Area and Aspect Ratio via Trapezoidal Integration
wing_le_eval = geo.evaluate(wing_le_line_para)
wing_te_eval = geo.evaluate(wing_te_line_para)
chords_w = csdl.norm(wing_te_eval - wing_le_eval, axes=(1,))
mid_w = 0.5 * (wing_le_eval + wing_te_eval)
dl_w = csdl.norm(mid_w[1:] - mid_w[:-1], axes=(1,))
wing_area_comp = csdl.sum(0.5 * (chords_w[:-1] + chords_w[1:]) * dl_w)
wing_ar_comp = wingspan_comp**2 / wing_area_comp

# Wing taper ratios defining the planform (relative to root chord)
wing_mid_taper_comp = wing_mid_chord_comp / wing_root_chord_comp
wing_tip_taper_comp = wing_tip_chord_comp / wing_root_chord_comp

# Connection invariant vectors
conn_fuse_strut = geo.evaluate(para_fuse_strut_r) - geo.evaluate(para_strut_root_r)
conn_wing_strut = geo.evaluate(para_wing_strut_r) - geo.evaluate(para_strut_tip_r)
conn_strut_jury = geo.evaluate(para_strut_jury_r) - geo.evaluate(para_jury_base_r)
conn_wing_jury = geo.evaluate(para_wing_jury_r) - geo.evaluate(para_jury_tip_r)
conn_fuse_tail = geo.evaluate(para_fuse_tail) - geo.evaluate(para_tail_attach)

target_conn_fuse_strut = conn_fuse_strut.value.copy()
target_conn_wing_strut = conn_wing_strut.value.copy()
target_conn_strut_jury = conn_strut_jury.value.copy()
target_conn_fuse_tail = conn_fuse_tail.value[[0, 2]].copy()

# ==============================================================================
# 7. Define Target Geometric Design Variables
# ==============================================================================
# Nominal baseline values across all 12 geometric design variables
wing_area_dv = csdl.Variable(value=wing_area_comp.value, name='wing_area_target')
wing_ar_dv = csdl.Variable(value=wing_ar_comp.value, name='wing_ar_target')
wing_mid_taper_dv = csdl.Variable(value=wing_mid_taper_comp.value, name='wing_mid_taper_target')
wing_tip_taper_dv = csdl.Variable(value=wing_tip_taper_comp.value, name='wing_tip_taper_target')
wing_inboard_sweep_dv = csdl.Variable(value=wing_inboard_sweep_comp.value, name='wing_inboard_sweep_target')
wing_outboard_sweep_dv = csdl.Variable(value=wing_outboard_sweep_comp.value, name='wing_outboard_sweep_target')

cabin_length_dv = csdl.Variable(value=cabin_length_comp.value, name='cabin_length_target')
tail_moment_arm_dv = csdl.Variable(value=tail_moment_arm_comp.value, name='tail_moment_arm_target')
fuselage_radius_dv = csdl.Variable(value=fuselage_radius_comp.value, name='fuselage_radius_target')

tail_area_dv = csdl.Variable(value=tail_area_comp.value, name='tail_area_target')
tail_ar_dv = csdl.Variable(value=tail_ar_comp.value, name='tail_ar_target')
tail_taper_dv = csdl.Variable(value=tail_taper_comp.value, name='tail_taper_target')

# ==============================================================================
# 8. Setup and Evaluate ParameterizationSolver
# ==============================================================================
print("Configuring ParameterizationSolver...")
solver = ParameterizationSolver()

# Add all 24 parameterization states as grouped vector variables
solver.add_state(wing_span_stretch)
solver.add_state(wing_chord_stretches)
solver.add_state(wing_inboard_sweep_trans)
solver.add_state(wing_outboard_sweep_trans)

solver.add_state(fuse_cabin_stretch)
solver.add_state(fuse_tail_stretch)
solver.add_state(fuse_radius_stretch)

solver.add_state(tail_span_stretch)
solver.add_state(tail_chord_stretches)
solver.add_state(tail_trans_xz)

solver.add_state(strut_root_trans)
solver.add_state(strut_tip_trans)

solver.add_state(jury_trans)

# Add connection invariant equality constraints (11 equations)
solver.add_equality_constraint(conn_fuse_strut, target_conn_fuse_strut)
solver.add_equality_constraint(conn_wing_strut, target_conn_wing_strut)
solver.add_equality_constraint(conn_strut_jury, target_conn_strut_jury)
solver.add_equality_constraint(conn_fuse_tail[[0, 2]], target_conn_fuse_tail)

# Add geometric design variables (12 equations -> 23x23 fully determined square system)
geom_vars = GeometricVariables()
geom_vars.add_variable(wing_area_comp, wing_area_dv)
geom_vars.add_variable(wing_ar_comp, wing_ar_dv)
geom_vars.add_variable(wing_mid_taper_comp, wing_mid_taper_dv)
geom_vars.add_variable(wing_tip_taper_comp, wing_tip_taper_dv)
geom_vars.add_variable(wing_inboard_sweep_comp, wing_inboard_sweep_dv)
geom_vars.add_variable(wing_outboard_sweep_comp, wing_outboard_sweep_dv)

geom_vars.add_variable(cabin_length_comp, cabin_length_dv)
geom_vars.add_variable(tail_moment_arm_comp, tail_moment_arm_dv)
geom_vars.add_variable(fuselage_radius_comp, fuselage_radius_dv)

geom_vars.add_variable(tail_area_comp, tail_area_dv)
geom_vars.add_variable(tail_ar_comp, tail_ar_dv)
geom_vars.add_variable(tail_taper_comp, tail_taper_dv)

print("Solving baseline geometry parameterization...")
t_solv0 = time.time()
solver.evaluate(geom_vars)
print(f"Parameterization solver converged in {time.time() - t_solv0:.2f}s.\n")

print("--------------------------------------------------------------------------------")
print("                     Baseline Parameterization Verification                     ")
print("--------------------------------------------------------------------------------")
print(f"  Wingspan             : {wingspan_comp.value[0]:.3f} m")
print(f"  Wing Area (Trap)     : {wing_area_comp.value[0]:.3f} m²  (Target: {wing_area_dv.value[0]:.3f} m²)")
print(f"  Wing Aspect Ratio    : {wing_ar_comp.value[0]:.3f}      (Target: {wing_ar_dv.value[0]:.3f})")
print(f"  Wing Mid Taper       : {wing_mid_taper_comp.value[0]:.3f}   (Target: {wing_mid_taper_dv.value[0]:.3f})")
print(f"  Wing Tip Taper       : {wing_tip_taper_comp.value[0]:.3f}   (Target: {wing_tip_taper_dv.value[0]:.3f})")
print(f"  Wing Root Chord      : {wing_root_chord_comp.value[0]:.3f} m")
print(f"  Wing Mid Chord       : {wing_mid_chord_comp.value[0]:.3f} m")
print(f"  Wing Tip Chord       : {wing_tip_chord_comp.value[0]:.3f} m")
print(f"  Wing Inboard Sweep   : {wing_inboard_sweep_comp.value[0]*180/np.pi:.3f} deg (Target: {wing_inboard_sweep_dv.value[0]*180/np.pi:.3f} deg)")
print(f"  Wing Outboard Sweep  : {wing_outboard_sweep_comp.value[0]*180/np.pi:.3f} deg (Target: {wing_outboard_sweep_dv.value[0]*180/np.pi:.3f} deg)")
print(f"  Cabin Length         : {cabin_length_comp.value[0]:.3f} m   (Target: {cabin_length_dv.value[0]:.3f} m)")
print(f"  Tail Moment Arm      : {tail_moment_arm_comp.value[0]:.3f} m   (Target: {tail_moment_arm_dv.value[0]:.3f} m)")
print(f"  Fuselage Radius      : {fuselage_radius_comp.value[0]:.3f} m   (Target: {fuselage_radius_dv.value[0]:.3f} m)")
print(f"  Tail Span            : {tail_span_comp.value[0]:.3f} m")
print(f"  Tail Root Chord      : {tail_root_chord_comp.value[0]:.3f} m")
print(f"  Tail Tip Chord       : {tail_tip_chord_comp.value[0]:.3f} m")
print(f"  Tail Area            : {tail_area_comp.value[0]:.3f} m²  (Target: {tail_area_dv.value[0]:.3f} m²)")
print(f"  Tail Aspect Ratio    : {tail_ar_comp.value[0]:.3f}      (Target: {tail_ar_dv.value[0]:.3f})")
print(f"  Tail Taper Ratio     : {tail_taper_comp.value[0]:.3f}   (Target: {tail_taper_dv.value[0]:.3f})")
print("--------------------------------------------------------------------------------")
print("                   Connection Invariant Verification Residuals                  ")
print("--------------------------------------------------------------------------------")
err_fuse_strut = np.linalg.norm(conn_fuse_strut.value - target_conn_fuse_strut)
err_wing_strut = np.linalg.norm(conn_wing_strut.value - target_conn_wing_strut)
err_strut_jury = np.linalg.norm(conn_strut_jury.value - target_conn_strut_jury)
err_fuse_tail  = np.linalg.norm(conn_fuse_tail.value[[0, 2]] - target_conn_fuse_tail)
print(f"  Fuselage -> Strut Root Joint Error : {err_fuse_strut:.2e} m")
print(f"  Wing     -> Strut Tip Joint Error  : {err_wing_strut:.2e} m")
print(f"  Strut    -> Jury Base Joint Error  : {err_strut_jury:.2e} m")
print(f"  Fuselage -> Tail Joint Error (XZ)  : {err_fuse_tail:.2e} m")
print("--------------------------------------------------------------------------------\n")

# ==============================================================================
# 9. JaxSimulator Setup for Fast Interactive Re-Evaluation
# ==============================================================================
print("=== Setting up JaxSimulator for Geometric Variable Perturbations ===")
jax_inputs = [
    wing_area_dv,
    wing_ar_dv,
    wing_mid_taper_dv,
    wing_tip_taper_dv,
    wing_inboard_sweep_dv,
    wing_outboard_sweep_dv,
    cabin_length_dv,
    tail_moment_arm_dv,
    fuselage_radius_dv,
    tail_area_dv,
    tail_ar_dv,
    tail_taper_dv,
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
    for gm in valid_ghost_meshes:
        plotter.add_mesh(
            gm,
            color="#B6B1A9",
            opacity=0.25,
            smooth_shading=True,
            show_edges=False,
        )

    # 2. Current perturbed geometry with vivid component colors (rendered without merging to preserve smooth normals)
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

# Define geometric design variables to sweep with nominal values and large visible perturbations (~50%)
variables_to_sweep = [
    {
        'variable': wing_area_dv,
        'name': 'Wing Area',
        'unit': 'm²',
        'nominal': wing_area_dv.value[0],
        'offset': 0.50 * float(np.squeeze(wing_area_dv.value[0])),
    },
    {
        'variable': wing_ar_dv,
        'name': 'Wing Aspect Ratio',
        'unit': '',
        'nominal': wing_ar_dv.value[0],
        'offset': 0.45 * float(np.squeeze(wing_ar_dv.value[0])),
    },
    {
        'variable': wing_mid_taper_dv,
        'name': 'Wing Mid Taper Ratio',
        'unit': '',
        'nominal': wing_mid_taper_dv.value[0],
        'offset': 0.45 * float(np.squeeze(wing_mid_taper_dv.value[0])),
    },
    {
        'variable': wing_tip_taper_dv,
        'name': 'Wing Tip Taper Ratio',
        'unit': '',
        'nominal': wing_tip_taper_dv.value[0],
        'offset': 0.45 * float(np.squeeze(wing_tip_taper_dv.value[0])),
    },
    {
        'variable': wing_inboard_sweep_dv,
        'name': 'Wing Inboard Sweep Angle',
        'unit': '°',
        'nominal': wing_inboard_sweep_dv.value[0],
        'offset': 0.50 * float(np.squeeze(wing_inboard_sweep_dv.value[0])),
        'is_angle': True,
    },
    {
        'variable': wing_outboard_sweep_dv,
        'name': 'Wing Outboard Sweep Angle',
        'unit': '°',
        'nominal': wing_outboard_sweep_dv.value[0],
        'offset': 0.50 * float(np.squeeze(wing_outboard_sweep_dv.value[0])),
        'is_angle': True,
    },
    {
        'variable': cabin_length_dv,
        'name': 'Cabin Length',
        'unit': 'm',
        'nominal': cabin_length_dv.value[0],
        'offset': 0.35 * float(np.squeeze(cabin_length_dv.value[0])),
        'offset_neg': 0.15 * float(np.squeeze(cabin_length_dv.value[0])),
    },
    {
        'variable': tail_moment_arm_dv,
        'name': 'Tail Moment Arm',
        'unit': 'm',
        'nominal': tail_moment_arm_dv.value[0],
        'offset': 0.30 * float(np.squeeze(tail_moment_arm_dv.value[0])),
        'offset_neg': 0.12 * float(np.squeeze(tail_moment_arm_dv.value[0])),
    },
    {
        'variable': fuselage_radius_dv,
        'name': 'Fuselage Radius',
        'unit': 'm',
        'nominal': fuselage_radius_dv.value[0],
        'offset': 0.35 * float(np.squeeze(fuselage_radius_dv.value[0])),
    },
    {
        'variable': tail_area_dv,
        'name': 'Tail Area',
        'unit': 'm²',
        'nominal': tail_area_dv.value[0],
        'offset': 0.50 * float(np.squeeze(tail_area_dv.value[0])),
    },
    {
        'variable': tail_ar_dv,
        'name': 'Tail Aspect Ratio',
        'unit': '',
        'nominal': tail_ar_dv.value[0],
        'offset': 0.45 * float(np.squeeze(tail_ar_dv.value[0])),
    },
    {
        'variable': tail_taper_dv,
        'name': 'Tail Taper Ratio',
        'unit': '',
        'nominal': tail_taper_dv.value[0],
        'offset': 0.45 * float(np.squeeze(tail_taper_dv.value[0])),
    },
]

def generate_sweep_values_with_hold(nominal, offset, offset_neg=None, num_osc_points=16, hold_points=3):
    nominal = float(np.squeeze(nominal))
    offset = float(np.squeeze(offset))
    if offset_neg is None:
        offset_neg = offset
    else:
        offset_neg = float(np.squeeze(offset_neg))
    hold_start = np.full(hold_points, nominal)
    s = np.sin(np.linspace(0, 2 * np.pi, num_osc_points, endpoint=False))
    osc = nominal + np.where(s >= 0, offset * s, offset_neg * s)
    hold_end = np.full(hold_points, nominal)
    return np.concatenate([hold_start, osc, hold_end])

print("Rendering video frames across all 12 geometric design variables...")

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
    offset_neg_val = var_info.get('offset_neg', None)

    display_nominal = nominal_val * 180.0 / np.pi if is_angle else nominal_val
    display_offset = offset_val * 180.0 / np.pi if is_angle else offset_val

    print(f"  [{var_idx+1}/{len(variables_to_sweep)}] Sweeping {var_name} (nominal = {display_nominal:.2f}{unit_str}, offset = ±{display_offset:.2f}{unit_str})...")

    sweep_values = generate_sweep_values_with_hold(
        nominal_val, offset_val, offset_neg=offset_neg_val, num_osc_points=16, hold_points=3
    )

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
    render_frame("Nominal Return", "All 11 Geometric DVs Verified", "Parameterization Complete")

plotter.close()
print(f"\nSuccessfully generated video: {video_path}")

# Copy video to artifact directory
artifact_dir = "/home/andrewfletcher/.gemini/antigravity/brain/29740a73-aa03-496c-b841-730839d9e40d"
if os.path.exists(artifact_dir):
    dest_video = os.path.join(artifact_dir, "truss_braced_wing_perturbations.mp4")
    shutil.copy2(video_path, dest_video)
    print(f"Copied video to artifact directory: {dest_video}")

