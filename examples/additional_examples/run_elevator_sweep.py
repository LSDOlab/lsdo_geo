"""
Elevator Angle Sweep Simulation and Aerodynamic Response Analysis
Runs VortexAD panel method over elevator deflection angles [-20 deg, +20 deg]
and plots Pitching Moment (My, Cm), Lift (L, CL), and Induced Drag (Di, CDi).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo
import VortexAD
import meshio
from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.sectional_parameterization import (
    SectionalParameterization,
    SectionalParameters
)

def run_elevator_sweep():
    print("=== Starting Elevator Angle Sweep Simulation ===")
    
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geometry_directory = "examples/example_geometries/"
    file_name = "rectangular_wing_naca0012_10ar"
    stp_path = geometry_directory + file_name + ".stp"
    msh_path = geometry_directory + file_name + ".msh"

    print(f"Loading geometry from {stp_path}...")
    imported_function_set = lfs.import_file_patched(file_name=stp_path, parallelize=False)
    geometry = lsdo_geo.Geometry(
        functions=imported_function_set.functions,
        function_names=imported_function_set.function_names,
        name='imported_geometry',
        space=imported_function_set.space
    )

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

        new_space = lfs.BSplineSpaceNew(
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

    # Mesh & VortexAD setup
    mesh = meshio.read(msh_path)
    points_orig = mesh.points
    cells_dict = mesh.cells_dict
    cell_adjacency_data = VortexAD.find_cell_adjacency(points=points_orig, cells=cells_dict)

    points_orig = cell_adjacency_data[0] 
    cells_dict = cell_adjacency_data[1] 
    cell_adjacency = cell_adjacency_data[2] 
    edges2cells = cell_adjacency_data[3]
    points2cells = cell_adjacency_data[4]

    TE_properties = VortexAD.TE_detection(
        points=points_orig,
        cells=cells_dict,
        edges2cells=edges2cells,
        points2cells=points2cells,
        threshold_theta=125.
    )

    projected_panel_mesh = geometry.project(
        points_orig, 
        grid_search_density_parameter=1, 
        newton_tolerance=1.e-10, 
        grid_search_density_cutoff=30,
        projection_tolerance=1.e-3,
        force_reprojection=False, 
        plot=False
    )

    # FFD Block
    ffd_block = construct_ffd_block_around_entities(
        entities=geometry, 
        num_coefficients=(2, 15, 2),
        degree=(1, 3, 1)
    )

    ffd_sectional_parameterization = SectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )

    # Variables
    elevator_angle = csdl.Variable(value=0.0, name='elevator_angle')
    pitch = csdl.Variable(value=5.0 * np.pi / 180.0, name='pitch')  # nominal 5 deg angle of attack

    # Sectional parameters (nominal identity deformation for FFD)
    space_of_linear_15_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=2, coefficients_shape=(15,))
    chord_stretching_b_spline = lfs.Function(
        space=space_of_linear_15_dof_b_splines,
        coefficients=csdl.Variable(shape=(15,), value=np.zeros(15)),
        name='chord_stretching_b_spline_coefficients'
    )
    parametric_b_spline_inputs = np.linspace(0.0, 1.0, 15).reshape((-1, 1))
    chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)

    sectional_parameters = SectionalParameters()
    sectional_parameters.add_stretch(axis=0, stretch=chord_stretch_sectional_parameters)

    ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients)

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

    # Pitch rotation about quarter chord center
    geometry.rotate(
        rotation_origin=geometry.evaluate(quarter_chord_center),
        axis_vector=np.array([0., 1., 0.]),
        angles=pitch,
        units='radians'
    )

    # Aerodynamic solver inputs
    cruise_speed = csdl.Variable(value=20.0)
    velocity = csdl.concatenate([cruise_speed, csdl.Variable(value=0.0), csdl.Variable(value=0.0)])
    num_nodes = 1

    panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
    panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')

    point_velocities = csdl.expand(velocity, (num_nodes,) + panel_mesh.shape[1:], 'j->iaj')
    rho_array = csdl.Variable(shape=(num_nodes,), value=np.array([1.225]))
    sos_array = csdl.Variable(shape=(num_nodes,), value=np.array([343.0]))

    pm_solver_inputs = {
        'V_inf': -point_velocities,
        'rho': rho_array,
        'sos': sos_array,
        'compressibility': True,
        'Cp cutoff': -5.,
        'partition_size': 1,
        'reuse_AIC': True,
        'ref_area': 10.0,
        'moment_reference': np.array([0.25, 0.0, 0.0]),
    }

    panel_method = VortexAD.PanelMethod(
        solver_input_dict=pm_solver_inputs,
        skip_geometry=True
    )
    panel_method.insert_grid_data(
        mesh=panel_mesh[0, :],
        cell_adjacency_data=cell_adjacency_data,
        TE_properties=TE_properties
    )
    panel_method.declare_outputs(['L', 'Di', 'M', 'CL', 'CDi', 'CM'])

    recorder.inline = False
    outputs = panel_method.evaluate()

    L = outputs['L']
    Di = outputs['Di']
    M = outputs['M']
    CL = outputs['CL']
    CDi = outputs['CDi']
    CM = outputs['CM']

    print("Compiling JaxSimulator...")
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[elevator_angle, pitch],
        additional_outputs=[L, Di, M, CL, CDi, CM],
        gpu=False
    )

    # Sweep elevator angles from -20 deg to +20 deg
    elevator_angles_deg = np.linspace(-20.0, 20.0, 21)
    results = {
        'delta_e_deg': elevator_angles_deg,
        'L': [],
        'Di': [],
        'My': [],
        'CL': [],
        'CDi': [],
        'Cm': []
    }

    print(f"Sweeping elevator over {len(elevator_angles_deg)} angles from -20° to +20°...")
    for deg in elevator_angles_deg:
        rad = np.radians(deg)
        jax_sim[elevator_angle] = rad
        jax_sim[pitch] = np.radians(5.0)  # nominal 5 deg alpha
        jax_sim.run()

        l_val = float(np.asarray(jax_sim[L]).flatten()[0])
        di_val = float(np.asarray(jax_sim[Di]).flatten()[0])
        my_val = float(np.asarray(jax_sim[M])[0, 1])  # Pitching moment My about (0.25, 0, 0)
        cl_val = float(np.asarray(jax_sim[CL]).flatten()[0])
        cdi_val = float(np.asarray(jax_sim[CDi]).flatten()[0])
        cm_val = float(np.asarray(jax_sim[CM])[0, 1])

        results['L'].append(l_val)
        results['Di'].append(di_val)
        results['My'].append(my_val)
        results['CL'].append(cl_val)
        results['CDi'].append(cdi_val)
        results['Cm'].append(cm_val)

        print(f"  δe = {deg:+5.1f}° | L = {l_val:7.1f} N (CL = {cl_val:5.3f}) | My = {my_val:7.2f} N*m (Cm = {cm_val:6.4f}) | Di = {di_val:6.2f} N")

    for k in ['L', 'Di', 'My', 'CL', 'CDi', 'Cm']:
        results[k] = np.array(results[k])

    # Plotting
    print("Generating response plots...")
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    # 1. Pitching Moment My & Cm
    ax1 = axes[0]
    color_m = '#d95f02'
    ax1.plot(results['delta_e_deg'], results['My'], 'o-', color=color_m, linewidth=2.2, markersize=5, label='My [N·m]')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax1.set_ylabel('Pitching Moment $M_y$ (N·m)', color=color_m, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_m)
    ax1.set_title('Aerodynamic Response over Elevator Deflection Angle Sweep (V = 20 m/s, α = 5°)', fontsize=14, fontweight='bold', pad=12)

    ax1_twin = ax1.twinx()
    color_cm = '#7570b3'
    ax1_twin.plot(results['delta_e_deg'], results['Cm'], 's--', color=color_cm, linewidth=1.5, markersize=4, label='$C_m$')
    ax1_twin.set_ylabel('Pitching Moment Coeff $C_m$', color=color_cm, fontsize=12, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_cm)
    ax1_twin.grid(False)

    # Find zero moment crossing (trim angle)
    if np.any(results['My'][:-1] * results['My'][1:] <= 0):
        trim_idx = np.where(results['My'][:-1] * results['My'][1:] <= 0)[0][0]
        y0, y1 = results['My'][trim_idx], results['My'][trim_idx+1]
        x0, x1 = results['delta_e_deg'][trim_idx], results['delta_e_deg'][trim_idx+1]
        trim_angle = x0 - y0 * (x1 - x0) / (y1 - y0)
        ax1.plot(trim_angle, 0.0, 'k*', markersize=12, label=f'Moment Trim $\\delta_e$ = {trim_angle:+.2f}°')
        ax1.annotate(f'Moment Trim: $\\delta_e = {trim_angle:+.2f}^\\circ$',
                     xy=(trim_angle, 0.0), xytext=(trim_angle + 3, results['My'].max()*0.2),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                     fontweight='bold', fontsize=10)

    ax1.legend(loc='upper left', frameon=True)

    # 2. Lift L & CL
    ax2 = axes[1]
    color_l = '#1b9e77'
    ax2.plot(results['delta_e_deg'], results['L'], 'o-', color=color_l, linewidth=2.2, markersize=5, label='Lift L [N]')
    ax2.set_ylabel('Lift Force $L$ (N)', color=color_l, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_l)

    ax2_twin = ax2.twinx()
    color_cl = '#386cb0'
    ax2_twin.plot(results['delta_e_deg'], results['CL'], 's--', color=color_cl, linewidth=1.5, markersize=4, label='$C_L$')
    ax2_twin.set_ylabel('Lift Coefficient $C_L$', color=color_cl, fontsize=12, fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=color_cl)
    ax2_twin.grid(False)
    ax2.legend(loc='upper left', frameon=True)

    # 3. Induced Drag Di & CDi
    ax3 = axes[2]
    color_d = '#e7298a'
    ax3.plot(results['delta_e_deg'], results['Di'], 'o-', color=color_d, linewidth=2.2, markersize=5, label='Induced Drag $D_i$ [N]')
    ax3.set_xlabel('Elevator Deflection Angle $\\delta_e$ (degrees)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Induced Drag $D_i$ (N)', color=color_d, fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=color_d)

    ax3_twin = ax3.twinx()
    color_cd = '#66a61e'
    ax3_twin.plot(results['delta_e_deg'], results['CDi'], 's--', color=color_cd, linewidth=1.5, markersize=4, label='$C_{Di}$')
    ax3_twin.set_ylabel('Induced Drag Coeff $C_{Di}$', color=color_cd, fontsize=12, fontweight='bold')
    ax3_twin.tick_params(axis='y', labelcolor=color_cd)
    ax3_twin.grid(False)
    ax3.legend(loc='upper center', frameon=True)

    plt.tight_layout()
    plot_path = "examples/additional_examples/elevator_sweep_results.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

    # Copy to artifact directory
    artifact_dir = "/home/andrew/.gemini/antigravity/brain/6430b318-3da2-48b6-9610-a3a9a37d089f"
    os.system(f"cp {plot_path} {artifact_dir}/elevator_sweep_results.png")
    print(f"Copied plot to artifact directory: {artifact_dir}/elevator_sweep_results.png")

if __name__ == '__main__':
    run_elevator_sweep()
