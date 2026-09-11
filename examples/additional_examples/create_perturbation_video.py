"""
Geometric Design Variable Smooth Perturbation Video Generator
Uses PyVista with high-quality smooth shading, lighting, and HUD overlays
to visualize dynamic perturbations of each geometric design variable:
1. Elevator Deflection Angle
2. Sectional Sweep Distribution
3. Chord Taper Distribution
4. Wingspan Extension
5. Pitch Angle (Angle of Attack)
"""

import os
import sys
import numpy as np
import pyvista as pv
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo as lg
from lsdo_geo import (
    Geometry,
    construct_ffd_block_around_entities,
    SectionalParameterization,
    SectionalParameters,
    import_geometry,
)

def create_perturbation_video():
    print("=== Setting up Geometry and Parameterization Model ===")

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geometry_directory = "examples/example_geometries/"
    file_name = "rectangular_wing_naca0012_10ar"
    stp_path = geometry_directory + file_name + ".stp"

    geometry = import_geometry(file_name=stp_path, parallelize=False)

    # Linearly interpolate spanwise control points to 15
    num_spanwise_cp_target = 15
    for idx in list(geometry.functions.keys()):
        function = geometry.functions[idx]
        coeffs = function.coefficients.value if hasattr(function.coefficients, 'value') else function.coefficients
        axis0_y_range = np.ptp([coeffs[i, :, 1].mean() for i in range(coeffs.shape[0])])
        axis1_y_range = np.ptp([coeffs[:, j, 1].mean() for j in range(coeffs.shape[1])])

        if max(axis0_y_range, axis1_y_range) < 1.0:
            continue

        spanwise_axis = 0 if axis0_y_range >= axis1_y_range else 1
        orig_degree = function.space.degree

        if spanwise_axis == 0:
            n_chord = coeffs.shape[1]
            new_coeffs = np.zeros((num_spanwise_cp_target, n_chord, 3))
            for j in range(n_chord):
                for k in range(3):
                    new_coeffs[:, j, k] = np.linspace(coeffs[0, j, k], coeffs[-1, j, k], num_spanwise_cp_target)
            new_shape = (num_spanwise_cp_target, n_chord)
            new_degree = (min(2, num_spanwise_cp_target - 1), orig_degree[1])
        else:
            n_chord = coeffs.shape[0]
            new_coeffs = np.zeros((n_chord, num_spanwise_cp_target, 3))
            for j in range(n_chord):
                for k in range(3):
                    new_coeffs[j, :, k] = np.linspace(coeffs[j, 0, k], coeffs[j, -1, k], num_spanwise_cp_target)
            new_shape = (n_chord, num_spanwise_cp_target)
            new_degree = (orig_degree[0], min(2, num_spanwise_cp_target - 1))

        new_space = lfs.BSplineSpace(
            num_parametric_dimensions=2,
            degree=new_degree,
            coefficients_shape=new_shape,
        )
        geometry.functions[idx] = lfs.Function(space=new_space, coefficients=new_coeffs)

    geometry = lsdo_geo.Geometry(
        functions=geometry.functions,
        function_names=geometry.function_names,
        name=geometry.name,
        space=geometry.space
    )

    # Key projections
    quarter_chord_center = geometry.project(np.array([0.25, 0.0, 0.0]))
    elevator_hinge = geometry.project(np.array([0.80, 0.0, 0.0]))

    # FFD Block & Sectional Parameterization
    num_ffd_sections = 15
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry, 
        num_coefficients=(2, num_ffd_sections, 2),
        degree=(1, 3, 1)
    )

    ffd_sectional_parameterization = SectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )

    # Design Variables
    num_chord_stations = 8
    num_span_coeffs = 2 * num_chord_stations - 1

    chord_stretch_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations), name='chord_stretch_dvs')
    sweep_dvs = csdl.Variable(shape=(num_chord_stations,), value=np.zeros(num_chord_stations), name='sweep_dvs')
    span_stretch_dv = csdl.Variable(shape=(1,), value=np.array([0.0]), name='span_stretch_dv')
    elevator_angle = csdl.Variable(value=0.0, name='elevator_angle')
    pitch = csdl.Variable(value=0.0, name='pitch')

    space_15_dof = lfs.BSplineSpace(num_parametric_dimensions=1, degree=2, coefficients_shape=(num_span_coeffs,))
    space_2_dof = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

    chord_coeffs = csdl.concatenate(
        [chord_stretch_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [chord_stretch_dvs[i] for i in range(num_chord_stations)]
    )
    sweep_coeffs = csdl.concatenate(
        [sweep_dvs[i] for i in range(num_chord_stations - 1, 0, -1)] +
        [sweep_dvs[i] for i in range(num_chord_stations)]
    )
    wingspan_coeffs = csdl.concatenate([-span_stretch_dv, span_stretch_dv])

    chord_b_spline = lfs.Function(space=space_15_dof, coefficients=chord_coeffs, name='chord_b_spline')
    sweep_b_spline = lfs.Function(space=space_15_dof, coefficients=sweep_coeffs, name='sweep_b_spline')
    wingspan_b_spline = lfs.Function(space=space_2_dof, coefficients=wingspan_coeffs, name='wingspan_b_spline')

    parametric_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
    chord_stretch_params = chord_b_spline.evaluate(parametric_inputs)
    sweep_params = sweep_b_spline.evaluate(parametric_inputs)
    wingspan_params = wingspan_b_spline.evaluate(parametric_inputs)

    sectional_params = SectionalParameters()
    sectional_params.add_stretch(axis=0, stretch=chord_stretch_params)
    sectional_params.add_translation(axis=0, translation=sweep_params)
    sectional_params.add_translation(axis=1, translation=wingspan_params)

    ffd_coeffs = ffd_sectional_parameterization.evaluate(sectional_params, plot=False)
    geom_coeffs = ffd_block.evaluate_ffd(coefficients=ffd_coeffs, plot=False)
    geometry.set_coefficients(geom_coeffs)

    # Elevator rotation
    hinge_origin = geometry.evaluate(elevator_hinge)
    for f_idx, row_slc in [(0, slice(0, 6)), (5, slice(0, 6)), (1, slice(97, None)), (4, slice(97, None))]:
        func = geometry.functions[f_idx]
        sub_pts = func.coefficients[:4, row_slc, :]
        rot_sub = lsdo_geo.rotate(
            points=sub_pts,
            rotation_origin=hinge_origin,
            axis_vector=np.array([0., 1., 0.]),
            angles=elevator_angle,
            units='radians'
        )
        func.coefficients = func.coefficients.set(csdl.slice[:4, row_slc, :], rot_sub)

    # Pitch rotation
    geometry.rotate(
        rotation_origin=geometry.evaluate(quarter_chord_center),
        axis_vector=np.array([0., 1., 0.]),
        angles=pitch,
        units='radians'
    )

    recorder.inline = False
    geom_outputs = [func.coefficients for func in geometry.functions.values()]

    print("Compiling JaxSimulator for fast geometry evaluation...")
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[chord_stretch_dvs, sweep_dvs, span_stretch_dv, elevator_angle, pitch],
        additional_outputs=geom_outputs,
        gpu=False
    )
    jax_sim.run()
    print("Geometry simulator compiled successfully!")

    # Set up PyVista offscreen renderer
    pv.OFF_SCREEN = True
    output_dir = "examples/additional_examples"
    video_path = os.path.join(output_dir, "geometric_variables_perturbation.mp4")
    
    # 1920x1080 Full HD
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    fps = 30
    plotter.open_movie(video_path, framerate=fps)

    # Camera setup: isometric high-angle 3/4 view
    camera_pos = (-4.0, -11.0, 7.5)
    focal_pt = (0.7, 0.0, 0.0)
    view_up = (0.0, 0.0, 1.0)

    # Helper function to reset all inputs
    def reset_inputs():
        jax_sim[chord_stretch_dvs] = np.zeros(num_chord_stations)
        jax_sim[sweep_dvs] = np.zeros(num_chord_stations)
        jax_sim[span_stretch_dv] = np.array([0.0])
        jax_sim[elevator_angle] = 0.0
        jax_sim[pitch] = 0.0

    # Helper function to render a frame
    def render_frame(title_str, var_name_str, val_str, detail_str="", active_station_y=None, station_color='#00e5ff'):
        jax_sim.run()
        plotter.clear()

        # Get surface elements from geometry
        plotting_elements = geometry.plot(show=False)
        for element in plotting_elements:
            mesh = None
            if isinstance(element, dict) and 'mesh' in element:
                mesh = element['mesh']
            elif isinstance(element, tuple) and len(element) == 2:
                mesh = element[0]
            elif isinstance(element, pv.DataSet) or isinstance(element, pv.PolyData):
                mesh = element

            if mesh is not None:
                # Ensure smooth point normals for realistic lighting and specular shading
                try:
                    smooth_mesh = mesh.compute_normals(
                        cell_normals=False, 
                        point_normals=True, 
                        auto_orient_normals=True
                    )
                except Exception:
                    smooth_mesh = mesh

                plotter.add_mesh(
                    smooth_mesh,
                    color='#3870a4',
                    smooth_shading=True,
                    specular=0.7,
                    specular_power=25,
                    ambient=0.22,
                    diffuse=0.78,
                    show_edges=False,
                )

        # Plot elevator hinge line for clear visual reference
        hinge_pts = np.array([
            [0.80, -1.07, 0.0],
            [0.80, 1.07, 0.0]
        ])
        hinge_line = pv.lines_from_points(hinge_pts)
        plotter.add_mesh(hinge_line, color='#ffaa00', line_width=4, label='Elevator Hinge')

        # If an individual station is being perturbed, add active station marker lines
        if active_station_y is not None:
            for sgn in ([1.0] if active_station_y == 0.0 else [1.0, -1.0]):
                y_pos = sgn * active_station_y
                st_line = pv.Line((-0.3, y_pos, 0.015), (1.8, y_pos, 0.015))
                plotter.add_mesh(st_line, color=station_color, line_width=5)

        # Multi-directional lighting kit for depth & shading
        plotter.enable_lightkit()
        plotter.set_background('#12151c', top='#1e2330')  # Sleek dark studio gradient

        # HUD Text Card
        hud_text = (
            f"LSDO_GEO: Parameterization & Aerodynamic Geometry\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Category: {title_str}\n"
            f"Variable: {var_name_str}\n"
            f"Value:    {val_str}\n"
            f"{detail_str}"
        )
        plotter.add_text(
            hud_text,
            position='upper_left',
            font_size=12,
            color='white',
            font='courier',
            shadow=True
        )

        plotter.camera.position = camera_pos
        plotter.camera.focal_point = focal_pt
        plotter.camera.up = view_up
        plotter.write_frame()

    print("Generating video frames...")

    station_y = np.linspace(0.0, 5.0, num_chord_stations)
    station_names = [
        "Center Root (Station 0)",
        "Root Inboard (Station 1)",
        "Inboard (Station 2)",
        "Mid-Inboard (Station 3)",
        "Mid-Outboard (Station 4)",
        "Outboard (Station 5)",
        "Sub-Tip (Station 6)",
        "Wingtip (Station 7)"
    ]

    # --- 0. Title Card / Baseline (30 frames = 1.0 sec) ---
    print("  [0/5] Baseline Geometry...")
    reset_inputs()
    for _ in range(30):
        render_frame(
            "Baseline Geometry",
            "Baseline Rectangular Wing (NACA 0012, AR=10, Span=10m)",
            "Nominal Unperturbed",
            "Initial baseline geometry before parametric perturbation"
        )

    # --- 1. Elevator Deflection Angle (60 frames = 2.0 sec) ---
    print("  [1/5] Perturbing Elevator Deflection Angle (δe)...")
    num_frames_elev = 60
    for i in range(num_frames_elev):
        t = i / num_frames_elev
        elev_deg = 20.0 * np.sin(2.0 * np.pi * t)
        jax_sim[elevator_angle] = np.radians(elev_deg)
        render_frame(
            "Elevator Deflection",
            "Elevator Deflection Angle (δe)",
            f"{elev_deg:+6.1f}°",
            "Trailing 20% chord across middle quarter-span (|y| <= 1.07m)"
        )
    reset_inputs()

    # --- 2. Individual Chord Variables (8 stations, 40 frames each = 320 frames = 10.7 sec) ---
    print("  [2/5] Perturbing 8 Individual Chord Variables (c0 to c7)...")
    num_frames_per_station = 40
    chord_sample_saved = False
    for k in range(num_chord_stations):
        y_k = station_y[k]
        pct = (k / 7.0) * 100.0
        st_desc = station_names[k]
        print(f"    - Chord Variable c{k}: y = ±{y_k:.2f}m ({pct:.1f}% span, {st_desc})")

        for i in range(num_frames_per_station):
            t = i / num_frames_per_station
            scale = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
            delta_c = scale * 0.80  # stretch chord by up to +0.80m
            current_chord = np.zeros(num_chord_stations)
            current_chord[k] = delta_c
            jax_sim[chord_stretch_dvs] = current_chord

            render_frame(
                f"Chord Variable {k}/7",
                f"Chord Stretch c{k} ({st_desc})",
                f"Δc = +{delta_c:.2f}m (local chord: {1.0 + delta_c:.2f}m)",
                f"Location: y = ±{y_k:.2f}m ({pct:.1f}% semi-span)",
                active_station_y=y_k,
                station_color='#00e5ff'
            )

            # Save sample screenshot at peak perturbation of station 3
            if k == 3 and i == num_frames_per_station // 2 and not chord_sample_saved:
                plotter.screenshot(os.path.join(output_dir, "chord_perturbation_sample.png"))
                chord_sample_saved = True

        reset_inputs()

    # --- 3. Individual Sweep Variables (8 stations, 40 frames each = 320 frames = 10.7 sec) ---
    print("  [3/5] Perturbing 8 Individual Sweep Variables (sw0 to sw7)...")
    sweep_sample_saved = False
    for k in range(num_chord_stations):
        y_k = station_y[k]
        pct = (k / 7.0) * 100.0
        st_desc = station_names[k]
        print(f"    - Sweep Variable sw{k}: y = ±{y_k:.2f}m ({pct:.1f}% span, {st_desc})")

        for i in range(num_frames_per_station):
            t = i / num_frames_per_station
            scale = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
            delta_sw = scale * 0.80  # translate aft by up to +0.80m
            current_sweep = np.zeros(num_chord_stations)
            current_sweep[k] = delta_sw
            jax_sim[sweep_dvs] = current_sweep

            render_frame(
                f"Sweep Variable {k}/7",
                f"Sectional Sweep sw{k} ({st_desc})",
                f"Δx_sweep = +{delta_sw:.2f}m (aft translation)",
                f"Location: y = ±{y_k:.2f}m ({pct:.1f}% semi-span)",
                active_station_y=y_k,
                station_color='#ff007f'
            )

            # Save sample screenshot at peak perturbation of station 5
            if k == 5 and i == num_frames_per_station // 2 and not sweep_sample_saved:
                plotter.screenshot(os.path.join(output_dir, "sweep_perturbation_sample.png"))
                sweep_sample_saved = True

        reset_inputs()

    # --- 4. Wingspan Extension (60 frames = 2.0 sec) ---
    print("  [4/5] Perturbing Wingspan Extension...")
    num_frames_span = 60
    for i in range(num_frames_span):
        t = i / num_frames_span
        scale = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
        span_offset = scale * 2.0  # half-span expands by up to 2m (full span: 10m -> 14m)
        jax_sim[span_stretch_dv] = np.array([span_offset])
        total_b = 10.0 + 2.0 * span_offset
        render_frame(
            "Wingspan Stretch",
            "Wingspan Extension (b)",
            f"Total Wingspan = {total_b:.2f}m (b/2 = {total_b/2:.2f}m)",
            "Linear spanwise FFD volume translation across full wingspan"
        )
    reset_inputs()

    # --- 5. Pitch Angle (Angle of Attack) (60 frames = 2.0 sec) ---
    print("  [5/5] Perturbing Pitch Angle (α)...")
    num_frames_pitch = 60
    for i in range(num_frames_pitch):
        t = i / num_frames_pitch
        pitch_deg = 8.0 * np.sin(2.0 * np.pi * t)
        jax_sim[pitch] = np.radians(pitch_deg)
        render_frame(
            "Pitch Angle",
            "Wing Pitch Angle / Angle of Attack (α)",
            f"{pitch_deg:+6.1f}°",
            "Rigid-body rotation about center quarter-chord reference axis"
        )
    reset_inputs()

    # --- Closing Hold (30 frames = 1.0 sec) ---
    for _ in range(30):
        render_frame(
            "Nominal Return",
            "Return to Baseline Rectangular Wing",
            "Complete",
            "All 19 geometric design variable perturbations demonstrated individually"
        )

    plotter.close()
    print(f"Successfully generated video: {video_path}")

    # Copy video and sample screenshots to artifact directory
    artifact_dir = "/home/andrew/.gemini/antigravity/brain/6430b318-3da2-48b6-9610-a3a9a37d089f"
    import shutil
    if os.path.exists(artifact_dir):
        shutil.copy2(video_path, os.path.join(artifact_dir, "geometric_variables_perturbation.mp4"))
        chord_sample_path = os.path.join(output_dir, "chord_perturbation_sample.png")
        sweep_sample_path = os.path.join(output_dir, "sweep_perturbation_sample.png")
        if os.path.exists(chord_sample_path):
            shutil.copy2(chord_sample_path, os.path.join(artifact_dir, "chord_perturbation_sample.png"))
        if os.path.exists(sweep_sample_path):
            shutil.copy2(sweep_sample_path, os.path.join(artifact_dir, "sweep_perturbation_sample.png"))
        print(f"Copied video and sample screenshots to artifact directory: {artifact_dir}")

if __name__ == '__main__':
    create_perturbation_video()
