"""
Lift-Plus-Cruise Design Space Exploration Figure Generator
===========================================================
Generates a publication-quality multi-panel figure for the ASME Journal of
Mechanical Design (JMD) paper showing the NASA Lift-Plus-Cruise geometry
benchmark under various design configurations:
  (a) Baseline Configuration (Nominal design)
  (b) Extended Tail Moment Arm (+35% moment arm stretch)
  (c) Increased Wing Aspect Ratio (+32% span stretch)
  (d) Decreased Wing AR (Without Inequalities) -> Rotor overlap & fuselage clash
  (e) Decreased Wing AR (With Inequalities)    -> Clearance enforced, rotors shrink
  (f) Latin Hypercube Sample (Representative point from 100-sample benchmark)

Output:
  - figures/lift_plus_cruise_design_points.pdf
  - figures/lift_plus_cruise_design_points.png (high-DPI)
"""

import os
import sys

# Auto-reexec with system libstdc++ if miniconda's older libstdc++ is shadowing it
# This ensures full hardware-accelerated OpenGL / GLX 3D rendering and shading in VTK
sys_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(sys_lib) and sys_lib not in os.environ.get("LD_PRELOAD", ""):
    env = os.environ.copy()
    current_preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{sys_lib}:{current_preload}" if current_preload else sys_lib
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import argparse
import numpy as np
from PIL import Image
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Compatibility check for OpenVSP / lsdo_geo
LSDO_GEO_PATH = "/home/andrew/optimization/lsdo_geo"
LPC_DIR = os.path.join(LSDO_GEO_PATH, "examples/showcase_examples/lift_plus_cruise")
if LPC_DIR not in sys.path:
    sys.path.insert(0, LPC_DIR)

# Ensure JAX runs on CPU
os.environ["JAX_PLATFORMS"] = "cpu"

def load_lpc_model():
    """
    Loads the Lift-Plus-Cruise geometry model, FFD blocks, and sectional
    parameterizations from lsdo_geo's canonical example up to the solver definition.
    """
    cad_script = os.path.join(LPC_DIR, "ex_lift_plus_cruise.py")
    if not os.path.exists(cad_script):
        raise FileNotFoundError(f"Cannot find LPC script at {cad_script}")

    with open(cad_script, "r") as f:
        # Read the geometric parameterization definitions (lines 1 to 410)
        code = f.readlines()[:410]

    env = {"__file__": cad_script, "__name__": "__main__"}
    print("Initializing Lift-Plus-Cruise CAD model and FFD volumes...")
    exec("".join(code), env)
    print("LPC geometry model loaded successfully.")
    return env


def build_component_lists(env):
    """Assembles all components and their original B-spline coefficients."""
    wing = env["wing"]
    fuselage = env["fuselage"]
    nose_hub = env["nose_hub"]
    h_tail = env["h_tail"]
    v_tail = env["v_tail"]
    pp_components = env["pp_components"]
    rotor_components = env["rotor_components"]
    boom_components = env["boom_components"]

    # Component list with publication colors
    components = [
        (wing, "#3498db", "Wing"),
        (h_tail, "#2ecc71", "Horizontal Tail"),
        (v_tail, "#1abc9c", "Vertical Tail"),
        (fuselage, "#95a5a6", "Fuselage"),
        (nose_hub, "#7f8c8d", "Nose Hub"),
    ]
    for p_comp in pp_components:
        components.append((p_comp, "#e74c3c", "Pusher Propeller"))
    for comp_set in rotor_components:
        for comp in comp_set:
            components.append((comp, "#e67e22", "Lift Rotor"))
    for boom in boom_components:
        components.append((boom, "#f39c12", "Boom"))

    # Save original (baseline) function coefficients
    all_comps = [wing, fuselage, nose_hub, h_tail, v_tail] + pp_components + [c for r in rotor_components for c in r] + boom_components
    orig_coeffs = {c.name: {k: v.coefficients.value.copy() for k, v in c.functions.items()} for c in all_comps}
    orig_coeffs["wing_ffd"] = env["wing_ffd_block"].coefficients.value.copy()

    return components, orig_coeffs


def reset_to_baseline(components, orig_coeffs):
    """Resets all component B-spline coefficients to their pristine baseline."""
    for comp, _, _ in components:
        if comp.name in orig_coeffs:
            for k, fn in comp.functions.items():
                if k in orig_coeffs[comp.name]:
                    fn.coefficients = orig_coeffs[comp.name][k].copy()


def apply_deformation(env, orig_coeffs,
                      tail_ext=0.0,
                      cabin_stretch=0.0,
                      cabin_width=0.0,
                      cabin_height=0.0,
                      inner_span_stretch=0.0,
                      outer_span_stretch=0.0,
                      rotor_radius_stretch=0.0,
                      chord_stretch=0.0):
    """
    Applies geometric state deformations through the hierarchical FFD and
    sectional parameterization models of lsdo_geo.
    """
    from lsdo_geo import SectionalParameters

    wing = env["wing"]
    fuselage = env["fuselage"]
    nose_hub = env["nose_hub"]
    h_tail = env["h_tail"]
    v_tail = env["v_tail"]
    pp_components = env["pp_components"]
    rotor_components = env["rotor_components"]
    boom_components = env["boom_components"]
    rotor_ffd_blocks = env["rotor_ffd_blocks"]

    # 1. Fuselage Sectional Deformation
    fuse_sec = env["fuselage_sec"]
    fuse_ffd = env["fuselage_ffd_block"]
    p_fuse = SectionalParameters()
    # Sections: cabin (index 1), tail cone (index 2)
    fuse_tx = np.array([0.0, cabin_stretch, cabin_stretch + tail_ext])
    fuse_sy = np.array([cabin_width, cabin_width, 0.0])
    fuse_sz = np.array([cabin_height, cabin_height, 0.0])
    p_fuse.add_translation(0, fuse_tx)
    if abs(cabin_width) > 1e-6:
        p_fuse.add_stretch(1, fuse_sy)
    if abs(cabin_height) > 1e-6:
        p_fuse.add_stretch(2, fuse_sz)

    eval_f = fuse_ffd.evaluate_ffd(fuse_sec.evaluate(p_fuse, plot=False), plot=False)
    fuselage.set_coefficients(eval_f[0])
    nose_hub.set_coefficients(eval_f[1])

    # 2. Horizontal Tail, Vertical Tail, Pusher Propeller Rigid Translation with Tail Cone
    dx_tail = cabin_stretch + tail_ext
    for k, fn in h_tail.functions.items():
        fn.coefficients = orig_coeffs[h_tail.name][k] + np.array([dx_tail, 0.0, 0.0])
    for k, fn in v_tail.functions.items():
        fn.coefficients = orig_coeffs[v_tail.name][k] + np.array([dx_tail, 0.0, 0.0])
    for comp in pp_components:
        for k, fn in comp.functions.items():
            fn.coefficients = orig_coeffs[comp.name][k] + np.array([dx_tail, 0.0, 0.0])

    # 3. Wing Sectional Span & Chord Deformation
    wing_sec = env["wing_sec"]
    wing_ffd = env["wing_ffd_block"]
    a_coeffs = env["a_coeffs"]
    b_coeffs = env["b_coeffs"]

    p_wing = SectionalParameters()
    # Note: axis 1 in wing_sec points along [0, -1, 0]. To stretch wing in +y direction,
    # the translation parameter must be negative of the desired displacement.
    sec_span_trans = -(
        inner_span_stretch * a_coeffs.reshape((-1, 1)) +
        outer_span_stretch * b_coeffs.reshape((-1, 1))
    )
    p_wing.add_translation(1, sec_span_trans)
    if abs(chord_stretch) > 1e-6:
        p_wing.add_stretch(0, np.full((7, 1), chord_stretch))

    w_eval_ffd = wing_sec.evaluate(p_wing, plot=False)
    eval_w = wing_ffd.evaluate_ffd(w_eval_ffd, plot=False)
    wing.set_coefficients(eval_w)

    # 4. Lift Booms and Rotors: compute exact wing junction displacement so booms are locked to wing
    orig_w_ffd = orig_coeffs["wing_ffd"]
    dy_inner = np.mean(w_eval_ffd.value[:, 4, :, 1] - orig_w_ffd[:, 4, :, 1])
    dy_outer = np.mean(w_eval_ffd.value[:, 5, :, 1] - orig_w_ffd[:, 5, :, 1])

    # Left rotors/booms (indices 0, 1, 4, 5): -y direction
    # Right rotors/booms (indices 2, 3, 6, 7): +y direction
    boom_dy = [
        -dy_outer,  # 0: rlo (rear left outer)
        -dy_inner,  # 1: rli (rear left inner)
        +dy_inner,  # 2: rri (rear right inner)
        +dy_outer,  # 3: rro (rear right outer)
        -dy_outer,  # 4: flo (front left outer)
        -dy_inner,  # 5: fli (front left inner)
        +dy_inner,  # 6: fri (front right inner)
        +dy_outer,  # 7: fro (front right outer)
    ]

    for i in range(8):
        dy = boom_dy[i]
        # Translate boom
        for k, fn in boom_components[i].functions.items():
            fn.coefficients = orig_coeffs[boom_components[i].name][k] + np.array([0.0, dy, 0.0])

        # Rotor scaling & Translation relative to rotor hub center
        comp_set = rotor_components[i]
        hub_comp = [c for c in comp_set if "hub" in c.name][0]
        hub_pts = np.vstack([orig_coeffs[hub_comp.name][k].reshape(-1, 3) for k in hub_comp.functions.keys()])
        hub_center = np.mean(hub_pts, axis=0)

        scale = 1.0 + rotor_radius_stretch
        for comp in comp_set:
            for k, fn in comp.functions.items():
                pts = orig_coeffs[comp.name][k].copy()
                if abs(rotor_radius_stretch) > 1e-6:
                    pts[:, :, 0] = (pts[:, :, 0] - hub_center[0]) * scale + hub_center[0]
                    pts[:, :, 1] = (pts[:, :, 1] - hub_center[1]) * scale + hub_center[1]
                pts[:, :, 1] += dy
                fn.coefficients = pts


UCSD_BLUE = "#0072CE"          # Vibrant Aerospace Cobalt Blue (Pantone 300C, high-contrast & professional)
UCSD_COASTAL_BLUE = "#0072CE"  # Default high-clarity blue
UCSD_GRAY = "#B6B1A9"          # UCSD Sand / Gray baseline ghost

def render_configuration(camera_pos, comp_ghost_meshes, comp_deformed_meshes, components,
                         output_image_path,
                         show_ghost=True,
                         deformed_color=UCSD_COASTAL_BLUE,
                         ghost_color=UCSD_GRAY,
                         ghost_opacity=0.3,
                         ambient=0.38,
                         diffuse=0.82,
                         specular=0.55):
    """
    Renders an off-screen high-resolution view of the aircraft with full 3D lighting,
    specular highlights, and smooth shading. Each panel uses an isolated fresh VTK
    Plotter instance to ensure pristine hardware-accelerated LightKit shading.
    The foreground geometry is rendered 100% opaque, while the nominal baseline is
    rendered underneath in 0.3 opacity UCSD Gray.
    """
    plotter = pv.Plotter(off_screen=True, window_size=[2560, 1600])
    plotter.set_background("white")

    # 1. Baseline Ghost Geometry Overlay (0.3 opacity UCSD Gray)
    # Smart Ghosting: only render patches where the geometry actually displaced
    # (displacement > 0.05m) to completely eliminate co-planar z-fighting artifacts.
    if show_ghost and comp_ghost_meshes is not None:
        for comp, col, name in components:
            gms = comp_ghost_meshes.get(comp.name, [])
            dms = comp_deformed_meshes.get(comp.name, [])
            for gm, dm in zip(gms, dms):
                diff = np.max(np.linalg.norm(gm.points - dm.points, axis=1))
                if diff > 0.05:
                    plotter.add_mesh(
                        gm,
                        color=ghost_color,
                        opacity=ghost_opacity,
                        smooth_shading=True,
                        show_edges=False,
                    )

    # 2. Deformed Aircraft Surface with 3D Rendering & Lighting
    # 100% OPAQUE solid foreground with specular reflection and diffuse contours
    for comp, col, name in components:
        for m in comp_deformed_meshes.get(comp.name, []):
            plotter.add_mesh(
                m,
                color=deformed_color,
                opacity=1.0,
                smooth_shading=True,
                specular=specular,
                specular_power=25,
                ambient=ambient,
                diffuse=diffuse,
                show_edges=False,
            )

    # Standard VTK LightKit and SSAA anti-aliasing on the scene
    plotter.enable_lightkit()
    plotter.enable_anti_aliasing("ssaa")
    plotter.camera_position = camera_pos
    plotter.screenshot(output_image_path)
    plotter.close()


def generate_panels(grid_mode="2x3", cache_dir="figures/lpc_panels", force_rerun=False,
                    deformed_color=UCSD_COASTAL_BLUE, ghost_color=UCSD_GRAY, ghost_opacity=0.3,
                    ambient=0.38, diffuse=0.82, specular=0.55, lhs_sample_idx=2):
    """
    Generates all requested design point images using off-screen PyVista rendering
    with UCSD Blue for the current configuration and 0.3 opacity UCSD Gray for the
    nominal baseline. All subpanels are cropped to a shared global union bounding
    box so that all subpanel frames are 100% identical in size and alignment.
    """
    os.makedirs(cache_dir, exist_ok=True)

    panel_configs = [
        {
            "id": "panel_a_baseline",
            "title": "(a) Baseline Configuration",
            "desc": "Nominal reference design ($S=209.7\\,\\mathrm{m}^2$, $\\mathrm{AR}=12.16$, $l_{\\mathrm{tail}}=17.18\\,\\mathrm{m}$)",
            "params": dict(tail_ext=0.0, inner_span_stretch=0.0, outer_span_stretch=0.0, rotor_radius_stretch=0.0),
            "show_ghost": False,
        },
        {
            "id": "panel_b_tail_ext",
            "title": "(b) Extended Tail Moment Arm",
            "desc": "$+40\\%$ tail moment arm ($l_{\\mathrm{tail}}=25.37\\,\\mathrm{m}$), translating tail cone & empennage",
            "params": dict(tail_ext=7.25, inner_span_stretch=0.0, outer_span_stretch=0.0, rotor_radius_stretch=0.0),
            "show_ghost": True,
        },
        {
            "id": "panel_c_high_ar",
            "title": "(c) Increased Aspect Ratio",
            "desc": "High aspect ratio ($\\mathrm{AR}=20.0$, $+28\\%$ span stretch at constant area)",
            "params": dict(tail_ext=0.0, inner_span_stretch=7.13, outer_span_stretch=7.13, rotor_radius_stretch=0.0),
            "show_ghost": True,
        },
        {
            "id": "panel_d_low_ar_no_ineq",
            "title": "(d) Low AR (Unconstrained)",
            "desc": "$\\mathrm{AR}=6.0$: Wings compress without constraints; rotors collide with fuselage & overlap",
            "params": dict(tail_ext=0.0, inner_span_stretch=-7.51, outer_span_stretch=-7.51, rotor_radius_stretch=0.0),
            "show_ghost": True,
        },
        {
            "id": "panel_e_low_ar_with_ineq",
            "title": "(e) Low AR (Constrained)",
            "desc": "$\\mathrm{AR}=6.0$: Clearance constraints active; inner wing preserves cabin spacing and rotor radii adapt ($-12\\%$)",
            "params": dict(tail_ext=0.0, inner_span_stretch=-2.5, outer_span_stretch=-14.53, rotor_radius_stretch=-0.12),
            "show_ghost": True,
        },
        {
            "id": "panel_f_lhs_sample",
            "title": "(f) Latin Hypercube Sample",
            "desc": "Representative inequality-constrained LHS benchmark point ($\\mathrm{AR}=16.09$, $l_{\\mathrm{tail}}=16.18\\,\\mathrm{m}$)",
            "params": None,
            "use_lhs_ineq_pkl": True,
            "lhs_sample_idx": lhs_sample_idx,
            "show_ghost": True,
        },
    ]

    if grid_mode == "2x2":
        selected_configs = [panel_configs[0], panel_configs[1], panel_configs[3], panel_configs[4]]
    elif grid_mode == "3x3":
        # Additional 3 diverse LHS / parametric designs
        panel_configs.extend([
            {
                "id": "panel_g_lhs_2",
                "title": "(g) LHS Sample 2 (Compact Cabin)",
                "desc": "Shorter cabin ($-25\\%$) with increased rotor radii ($+10\\%$) and extended moment arm",
                "params": dict(tail_ext=3.2, cabin_stretch=-1.8, cabin_width=0.4, inner_span_stretch=0.5, outer_span_stretch=1.0, rotor_radius_stretch=0.10),
                "show_ghost": True,
            },
            {
                "id": "panel_h_lhs_3",
                "title": "(h) LHS Sample 3 (Wide Cabin)",
                "desc": "Expanded cabin width ($+20\\%$) with reduced rotor radii and swept span adjustment",
                "params": dict(tail_ext=1.0, cabin_stretch=1.0, cabin_width=0.8, inner_span_stretch=-1.0, outer_span_stretch=2.0, rotor_radius_stretch=-0.12),
                "show_ghost": True,
            },
            {
                "id": "panel_i_lhs_4",
                "title": "(i) LHS Sample 4 (High Span & Tail)",
                "desc": "Maximum allowable span stretch coupled with enlarged tail surface and moment arm",
                "params": dict(tail_ext=4.5, cabin_stretch=1.5, inner_span_stretch=2.0, outer_span_stretch=3.5, rotor_radius_stretch=-0.05),
                "show_ghost": True,
            },
        ])
        selected_configs = panel_configs
    else:  # 2x3 default
        selected_configs = panel_configs[:6]

    # Check if all images already exist in cache
    all_cached = all(os.path.exists(os.path.join(cache_dir, f"{cfg['id']}.png")) for cfg in selected_configs)
    if all_cached and not force_rerun:
        print(f"All {len(selected_configs)} panel images found in cache '{cache_dir}'. Skipping 3D re-render.")
        return selected_configs

    # Otherwise, load model and render
    env = load_lpc_model()
    components, orig_coeffs = build_component_lists(env)

    # Preload inequality-constrained LHS dataset if needed
    lhs_pkl_path = os.path.join(LPC_DIR, "lift_plus_cruise_lhs_timing_results_with_sweep_with_inequalities.pkl")
    lhs_data = None
    if any(cfg.get("use_lhs_ineq_pkl", False) for cfg in selected_configs):
        if os.path.exists(lhs_pkl_path):
            import pickle
            print(f"Loading converged inequality-constrained LHS dataset from '{lhs_pkl_path}'...")
            with open(lhs_pkl_path, "rb") as f:
                lhs_data = pickle.load(f)
            print(f"Loaded {lhs_data['n_samples']} inequality-constrained benchmark samples.")

    # Capture initial baseline meshes per component
    reset_to_baseline(components, orig_coeffs)
    comp_ghost_meshes = {}
    for comp, _, _ in components:
        meshes = []
        for elem in comp.plot(show=False):
            m = elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem
            if m is not None:
                meshes.append(m.copy())
        comp_ghost_meshes[comp.name] = meshes

    pv.OFF_SCREEN = True
    # Elevated 3/4 perspective matching the LPC benchmark paper figure (Fig 3c)
    camera_pos = [(-50.0, -50.0, 45.0), (15.0, 0.0, 5.0), (0, 0, 1)]

    raw_paths = []
    for idx, cfg in enumerate(selected_configs):
        raw_path = os.path.join(cache_dir, f"{cfg['id']}_raw.png")
        raw_paths.append(raw_path)

        if os.path.exists(raw_path) and not force_rerun:
            print(f"[{idx+1}/{len(selected_configs)}] Using cached raw: {cfg['title']}")
            continue

        print(f"[{idx+1}/{len(selected_configs)}] Rendering: {cfg['title']}...")
        reset_to_baseline(components, orig_coeffs)

        if cfg.get("use_lhs_ineq_pkl", False) and lhs_data is not None:
            s_idx = cfg.get("lhs_sample_idx", 1)
            sample_res = lhs_data["sample_results"][s_idx]
            for fn_idx, fn in enumerate(env["geometry"].functions.values()):
                fn.coefficients = sample_res[fn_idx].copy()
        else:
            apply_deformation(env, orig_coeffs, **cfg["params"])

        # Capture deformed meshes per component
        comp_deformed_meshes = {}
        for comp, col, name in components:
            meshes = []
            for elem in comp.plot(show=False):
                m = elem["mesh"] if isinstance(elem, dict) and "mesh" in elem else elem
                if m is not None:
                    meshes.append(m)
            comp_deformed_meshes[comp.name] = meshes

        render_configuration(
            camera_pos=camera_pos,
            comp_ghost_meshes=comp_ghost_meshes,
            comp_deformed_meshes=comp_deformed_meshes,
            components=components,
            output_image_path=raw_path,
            show_ghost=cfg["show_ghost"],
            deformed_color=deformed_color,
            ghost_color=ghost_color,
            ghost_opacity=ghost_opacity,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
        )

    # Compute global union bounding box across all rendered raw images
    boxes = []
    for raw_p in raw_paths:
        im = Image.open(raw_p)
        arr = np.array(im)[:, :, :3]
        mask = np.any(arr < 250, axis=-1)
        r = np.where(mask.any(axis=1))[0]
        c = np.where(mask.any(axis=0))[0]
        boxes.append((r[0], r[-1], c[0], c[-1]))

    pad = 12
    global_r_min = max(0, min(b[0] for b in boxes) - pad)
    global_r_max = min(1600, max(b[1] for b in boxes) + 1 + pad)
    global_c_min = max(0, min(b[2] for b in boxes) - pad)
    global_c_max = min(2560, max(b[3] for b in boxes) + 1 + pad)
    print(f"Global union crop box: [{global_c_min}:{global_c_max}, {global_r_min}:{global_r_max}]")
    print(f"Cropped image size: {global_c_max - global_c_min} x {global_r_max - global_r_min} px (IDENTICAL FOR ALL PANELS!)")

    for cfg, raw_p in zip(selected_configs, raw_paths):
        crop_path = os.path.join(cache_dir, f"{cfg['id']}.png")
        im = Image.open(raw_p)
        im_c = im.crop((global_c_min, global_r_min, global_c_max, global_r_max))
        im_c.save(crop_path)

    return selected_configs


def assemble_grid_figure(configs, grid_mode="2x3",
                         cache_dir="figures/lpc_panels",
                         output_pdf="figures/lift_plus_cruise_design_points.pdf",
                         output_png="figures/lift_plus_cruise_design_points.png",
                         deformed_color=UCSD_BLUE,
                         ghost_color=UCSD_GRAY,
                         ghost_opacity=0.3,
                         dpi=300):
    """
    Composites the cropped subpanel renders into a space-efficient ASME JMD publication
    figure with consistent typography, sharp framing, zero redundant legend, and tightly
    packed subpanels.
    """
    print(f"Compositing {len(configs)} panels into a {grid_mode} grid figure...")

    # Set publication typography matching asmejour
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.titlesize": 10,
        "axes.titlesize": 9,
    })

    if grid_mode == "2x2":
        nrows, ncols = 2, 2
        figsize = (7.0, 3.8)
    elif grid_mode == "3x3":
        nrows, ncols = 3, 3
        figsize = (7.0, 4.8)
    else:  # 2x3 default
        nrows, ncols = 2, 3
        figsize = (7.0, 3.1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (ax, cfg) in enumerate(zip(axes_flat, configs)):
        img_path = os.path.join(cache_dir, f"{cfg['id']}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing rendered panel at {img_path}")

        im = Image.open(img_path)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])

        # Subfigure title with minimal padding
        ax.set_title(cfg["title"], fontsize=8.8, fontweight="bold", pad=3, color="#111111")
        # Thin border around each subpanel for clean framing
        for spine in ax.spines.values():
            spine.set_color("#d0d0d0")
            spine.set_linewidth(0.6)

    # Hide unused subplots if any
    for ax in axes_flat[len(configs):]:
        ax.axis("off")

    # Ultra-tight layout: minimal spacing between plots and zero bottom margin for legend
    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.93, hspace=0.10, wspace=0.03)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    print(f"Saving publication PDF: {output_pdf}")
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=dpi)

    print(f"Saving high-res PNG: {output_png}")
    plt.savefig(output_png, format="png", bbox_inches="tight", pad_inches=0.01, dpi=dpi)
    plt.close()
    print("Figure generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication figure for LPC benchmark.")
    parser.add_argument("--grid", choices=["2x2", "2x3", "3x3"], default="2x3",
                        help="Grid layout (default: 2x3 for 6 panels)")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    parser.add_argument("--cache-dir", default="figures/lpc_panels", help="Cache directory for panel PNGs")
    parser.add_argument("--output-pdf", default="figures/lift_plus_cruise_design_points.pdf", help="Output PDF path")
    parser.add_argument("--output-png", default="figures/lift_plus_cruise_design_points.png", help="Output PNG path")
    parser.add_argument("--color", choices=["coastal", "core"], default="coastal",
                        help="Deformed aircraft color: 'coastal' (#0076B6, lighter) or 'core' (#00629B, standard UCSD blue)")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-rendering of panel PNGs")
    parser.add_argument("--lhs-sample", type=int, default=2, help="LHS sample index for panel f (default: 2)")

    args = parser.parse_args()

    active_color = UCSD_COASTAL_BLUE if args.color == "coastal" else UCSD_BLUE

    configs = generate_panels(
        grid_mode=args.grid,
        cache_dir=args.cache_dir,
        force_rerun=args.force_rerun,
        deformed_color=active_color,
        lhs_sample_idx=args.lhs_sample,
    )

    assemble_grid_figure(
        configs=configs,
        grid_mode=args.grid,
        cache_dir=args.cache_dir,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        deformed_color=active_color,
        dpi=args.dpi,
    )

