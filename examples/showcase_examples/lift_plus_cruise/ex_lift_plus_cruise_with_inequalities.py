import time
import lsdo_geo
import lsdo_function_spaces as lfs
# t01 = time.time()
import csdl_alpha as csdl
import numpy as np
# from python_csdl_backend import Simulator
import lsdo_geo as lg


# t02 = time.time()
# print(t02-t01)
recorder = csdl.Recorder(inline=True)
recorder.start()

import_file_path = 'examples/example_geometries/'
import_file = 'lift_plus_cruise_final.stp'
geometry = lg.import_geometry(import_file_path + import_file, parallelize=False)


camera = {
    'position': [-50.0, -50.0, 76.0],
    'focal_point': [15.0, 0.0, 5.0],
    'viewup': (0, 0, 1),
}
initial_geometry_plot = geometry.plot(camera=camera, opacity=0.3, color='#FFCD00', show=False)

# region Declaring all components
# Wing, tails, fuselage
wing = geometry.declare_component(function_search_names=['Wing'], name='wing')
# wing.plot()
h_tail = geometry.declare_component(function_search_names=['Tail_1'], name='h_tail')
# h_tail.plot()
v_tail = geometry.declare_component(function_search_names=['Tail_2'], name='v_tail')
# v_tail.plot()
fuselage = geometry.declare_component(function_search_names=['Fuselage_***.main'], name='fuselage')
# fuselage.plot()


# Nose hub
nose_hub = geometry.declare_component(name='weird_nose_hub', function_search_names=['EngineGroup_10'])
# nose_hub.plot()


# Pusher prop
pp_disk = geometry.declare_component(name='pp_disk', function_search_names=['Rotor-9-disk'])
# pp_disk.plot()
pp_blade_1 = geometry.declare_component(name='pp_blade_1', function_search_names=['Rotor_9_blades, 0'])
# pp_blade_1.plot()
pp_blade_2 = geometry.declare_component(name='pp_blade_2', function_search_names=['Rotor_9_blades, 1'])
# pp_blade_2.plot()
pp_blade_3 = geometry.declare_component(name='pp_blade_3', function_search_names=['Rotor_9_blades, 2'])
# pp_blade_3.plot()
pp_blade_4 = geometry.declare_component(name='pp_blade_4', function_search_names=['Rotor_9_blades, 3'])
# pp_blade_4.plot()
pp_hub = geometry.declare_component(name='pp_hub', function_search_names=['Rotor_9_Hub'])
# pp_hub.plot()
pp_components = [pp_disk, pp_blade_1, pp_blade_2, pp_blade_3, pp_blade_4, pp_hub]

# Rotor: rear left outer
rlo_disk = geometry.declare_component(name='rlo_disk', function_search_names=['Rotor_2_disk'])
# rlo_disk.plot()
rlo_blade_1 = geometry.declare_component(name='rlo_blade_1', function_search_names=['Rotor_2_blades, 0'])
# rlo_blade_1.plot()
rlo_blade_2 = geometry.declare_component(name='rlo_blade_2', function_search_names=['Rotor_2_blades, 1'])
# rlo_blade_2.plot()
rlo_hub = geometry.declare_component(name='rlo_hub', function_search_names=['Rotor_2_Hub'])
# rlo_hub.plot()
rlo_boom = geometry.declare_component(name='rlo_boom', function_search_names=['Rotor_2_Support'])
# rlo_boom.plot()
rlo_components = [rlo_disk, rlo_blade_1, rlo_blade_2, rlo_hub]

# Rotor: rear left inner
rli_disk = geometry.declare_component(name='rli_disk', function_search_names=['Rotor_4_disk'])
# rli_disk.plot()
rli_blade_1 = geometry.declare_component(name='rli_blade_1', function_search_names=['Rotor_4_blades, 0'])
# rli_blade_1.plot()
rli_blade_2 = geometry.declare_component(name='rli_blade_2', function_search_names=['Rotor_4_blades, 1'])
# rli_blade_2.plot()
rli_hub = geometry.declare_component(name='rli_hub', function_search_names=['Rotor_4_Hub'])
# rli_hub.plot()
rli_boom = geometry.declare_component(name='rli_boom', function_search_names=['Rotor_4_Support'])
# rli_boom.plot()
rli_components = [rli_disk, rli_blade_1, rli_blade_2, rli_hub]

# Rotor: rear right inner
rri_disk = geometry.declare_component(name='rri_disk', function_search_names=['Rotor_6_disk'])
# rri_disk.plot()
rri_blade_1 = geometry.declare_component(name='rri_blade_1', function_search_names=['Rotor_6_blades, 0'])
# rri_blade_1.plot()
rri_blade_2 = geometry.declare_component(name='rri_blade_2', function_search_names=['Rotor_6_blades, 1'])
# rri_blade_2.plot()
rri_hub = geometry.declare_component(name='rri_hub', function_search_names=['Rotor_6_Hub'])
# rri_hub.plot()
rri_boom = geometry.declare_component(name='rri_boom', function_search_names=['Rotor_6_Support'])
# rri_boom.plot()
rri_components = [rri_disk, rri_blade_1, rri_blade_2, rri_hub]

# Rotor: rear right outer
rro_disk = geometry.declare_component(name='rro_disk', function_search_names=['Rotor_8_disk'])
# rro_disk.plot()
rro_blade_1 = geometry.declare_component(name='rro_blade_1', function_search_names=['Rotor_8_blades, 0'])
# rro_blade_1.plot()
rro_blade_2 = geometry.declare_component(name='rro_blade_2', function_search_names=['Rotor_8_blades, 1'])
# rro_blade_2.plot()
rro_hub = geometry.declare_component(name='rro_hub', function_search_names=['Rotor_8_Hub'])
# rro_hub.plot()
rro_boom = geometry.declare_component(name='rro_boom', function_search_names=['Rotor_8_Support'])
# rro_boom.plot()
rro_components = [rro_disk, rro_blade_1, rro_blade_2, rro_hub]

# Rotor: front left outer
flo_disk = geometry.declare_component(name='flo_disk', function_search_names=['Rotor_1_disk'])
# flo_disk.plot()
flo_blade_1 = geometry.declare_component(name='flo_blade_1', function_search_names=['Rotor_1_blades, 0'])
# flo_blade_1.plot()
flo_blade_2 = geometry.declare_component(name='flo_blade_2', function_search_names=['Rotor_1_blades, 1'])
# flo_blade_2.plot()
flo_hub = geometry.declare_component(name='flo_hub', function_search_names=['Rotor_1_Hub'])
# flo_hub.plot()
flo_boom = geometry.declare_component(name='flo_boom', function_search_names=['Rotor_1_Support'])
# flo_boom.plot()
flo_components = [flo_disk, flo_blade_1, flo_blade_2, flo_hub]

# Rotor: front left inner
fli_disk = geometry.declare_component(name='fli_disk', function_search_names=['Rotor_3_disk'])
# fli_disk.plot()
fli_blade_1 = geometry.declare_component(name='fli_blade_1', function_search_names=['Rotor_3_blades, 0'])
# fli_blade_1.plot()
fli_blade_2 = geometry.declare_component(name='fli_blade_2', function_search_names=['Rotor_3_blades, 1'])
# fli_blade_2.plot()
fli_hub = geometry.declare_component(name='fli_hub', function_search_names=['Rotor_3_Hub'])
# fli_hub.plot()
fli_boom = geometry.declare_component(name='fli_boom', function_search_names=['Rotor_3_Support'])
# fli_boom.plot()
fli_components = [fli_disk, fli_blade_1, fli_blade_2, fli_hub]

# Rotor: front right inner
fri_disk = geometry.declare_component(name='fri_disk', function_search_names=['Rotor_5_disk'])
# fri_disk.plot()
fri_blade_1 = geometry.declare_component(name='fri_blade_1', function_search_names=['Rotor_5_blades, 0'])
# fri_blade_1.plot()
fri_blade_2 = geometry.declare_component(name='fri_blade_2', function_search_names=['Rotor_5_blades, 1'])
# fri_blade_2.plot()
fri_hub = geometry.declare_component(name='fri_hub', function_search_names=['Rotor_5_Hub'])
# fri_hub.plot()
fri_boom = geometry.declare_component(name='fri_boom', function_search_names=['Rotor_5_Support'])
# fri_boom.plot()
fri_components = [fri_disk, fri_blade_1, fri_blade_2, fri_hub]

# Rotor: front right outer
fro_disk = geometry.declare_component(name='fro_disk', function_search_names=['Rotor_7_disk'])
# fro_disk.plot()
fro_blade_1 = geometry.declare_component(name='fro_blade_1', function_search_names=['Rotor_7_blades, 0'])
# fro_blade_1.plot()
fro_blade_2 = geometry.declare_component(name='fro_blade_2', function_search_names=['Rotor_7_blades, 1'])
# fro_blade_2.plot()
fro_hub = geometry.declare_component(name='fro_hub', function_search_names=['Rotor_7_Hub'])
# fro_hub.plot()
fro_boom = geometry.declare_component(name='fro_boom', function_search_names=['Rotor_7_Support'])
# fro_boom.plot()
fro_components = [fro_disk, fro_blade_1, fro_blade_2, fro_hub]
lift_rotor_related_components = [rlo_components, rli_components, rri_components, rro_components, 
                                 flo_components, fli_components, fri_components, fro_components]

boom_components = [rlo_boom, rli_boom, rri_boom, rro_boom, flo_boom, fli_boom, fri_boom, fro_boom]

# endregion

# region Defining key points
wing_te_right = wing.project(np.array([13.4, 25.250, 7.5]), plot=False)
wing_te_left = wing.project(np.array([13.4, -25.250, 7.5]), plot=False)
wing_te_center = wing.project(np.array([14.332, 0., 8.439]), plot=False)
wing_le_right = wing.project(np.array([12.356, 25.25, 7.618]), plot=False)
wing_le_left = wing.project(np.array([12.356, -25.25, 7.618]), plot=False)
wing_le_center = wing.project(np.array([8.892, 0., 8.633]), plot=False)
wing_qc_right = wing.project(np.array([0.25*13.4 + 0.75*12.356, 25.250, 8.5]), plot=False)
wing_qc_left = wing.project(np.array([0.25*13.4 + 0.75*12.356, -25.250, 8.5]), plot=False)
wing_qc_center = wing.project(np.array([10.25, 0., 8.5]), plot=False)

tail_te_right = h_tail.project(np.array([31.5, 6.75, 6.]), plot=False)
tail_te_left = h_tail.project(np.array([31.5, -6.75, 6.]), plot=False)
tail_le_right = h_tail.project(np.array([26.5, 6.75, 6.]), plot=False)
tail_le_left = h_tail.project(np.array([26.5, -6.75, 6.]), plot=False)
tail_te_center = h_tail.project(np.array([31.187, 0., 8.009]), plot=False)
tail_le_center = h_tail.project(np.array([27.428, 0., 8.009]), plot=False)
tail_qc_center = h_tail.project(np.array([0.25*31.187 + 0.75*27.428, 0., 8.]), plot=False)
tail_qc_right = h_tail.project(0.25*h_tail.evaluate(tail_te_right) + 0.75*h_tail.evaluate(tail_le_right), plot=False)
tail_qc_left = h_tail.project(0.25*h_tail.evaluate(tail_te_left) + 0.75*h_tail.evaluate(tail_le_left), plot=False)

fuselage_wing_qc = fuselage.project(np.array([10.25, 0., 8.5]), plot=False)
fuselage_wing_te_center = fuselage.project(np.array([14.332, 0., 8.439]), plot=False)
fuselage_tail_qc = fuselage.project(np.array([24.15, 0., 8.]), plot=False)
fuselage_tail_te_center = fuselage.project(np.array([31.187, 0., 8.009]), plot=False)

rlo_disk_pt = np.array([19.200, -18.750, 9.635])
rro_disk_pt = np.array([19.200, 18.750, 9.635])
rlo_boom_pt = np.array([12.000, -18.750, 7.613])
rro_boom_pt = np.array([12.000, 18.750, 7.613])

flo_disk_pt = np.array([5.070, -18.750, 7.355])
fro_disk_pt = np.array([5.070, 18.750, 7.355])
flo_boom_pt = np.array([12.200, -18.750, 7.615])
fro_boom_pt = np.array([12.200, 18.750, 7.615])

rli_disk_pt = np.array([18.760, -8.537, 9.919])
rri_disk_pt = np.array([18.760, 8.537, 9.919])
rli_boom_pt = np.array([11.500, -8.250, 7.898])
rri_boom_pt = np.array([11.500, 8.250, 7.898])

fli_disk_pt = np.array([4.630, -8.217, 7.659])
fri_disk_pt = np.array([4.630, 8.217, 7.659])
fli_boom_pt = np.array([11.741, -8.250, 7.900])
fri_boom_pt = np.array([11.741, 8.250, 7.900])

rlo_disk_center = rlo_disk.project(rlo_disk_pt)
rli_disk_center = rli_disk.project(rli_disk_pt)
rri_disk_center = rri_disk.project(rri_disk_pt)
rro_disk_center = rro_disk.project(rro_disk_pt)
flo_disk_center = flo_disk.project(flo_disk_pt)
fli_disk_center = fli_disk.project(fli_disk_pt)
fri_disk_center = fri_disk.project(fri_disk_pt)
fro_disk_center = fro_disk.project(fro_disk_pt)

rlo_disk_center_on_wing = wing.project(rlo_disk_pt)
rli_disk_center_on_wing = wing.project(rli_disk_pt)
rri_disk_center_on_wing = wing.project(rri_disk_pt)
rro_disk_center_on_wing = wing.project(rro_disk_pt)
flo_disk_center_on_wing = wing.project(flo_disk_pt)
fli_disk_center_on_wing = wing.project(fli_disk_pt)
fri_disk_center_on_wing = wing.project(fri_disk_pt)
fro_disk_center_on_wing = wing.project(fro_disk_pt)

boom_fro = fro_boom.project(fro_boom_pt)
boom_fri = fri_boom.project(fri_boom_pt)
boom_flo = flo_boom.project(flo_boom_pt)
boom_fli = fli_boom.project(fli_boom_pt)
boom_rro = rro_boom.project(rro_boom_pt)
boom_rri = rri_boom.project(rri_boom_pt)
boom_rli = rli_boom.project(rli_boom_pt)
boom_rlo = rlo_boom.project(rlo_boom_pt)

wing_boom_fro = wing.project(fro_boom_pt)
wing_boom_fri = wing.project(fri_boom_pt)
wing_boom_flo = wing.project(flo_boom_pt)
wing_boom_fli = wing.project(fli_boom_pt)
wing_boom_rro = wing.project(rro_boom_pt)
wing_boom_rri = wing.project(rri_boom_pt)
wing_boom_rli = wing.project(rli_boom_pt)
wing_boom_rlo = wing.project(rlo_boom_pt)

fuselage_nose = np.array([2.464, 0., 5.113])
fuselage_rear = np.array([31.889, 0., 7.798])
fuselage_nose_points_parametric = fuselage.project(fuselage_nose, grid_search_density_parameter=20)
fuselage_rear_points_parametric = fuselage.project(fuselage_rear)
fuselage_rear_point_on_pusher_disk_parametric = pp_disk.project(fuselage_rear)

# endregion

# region rotor meshes
# num_radial = 30
# num_spanwise_vlm_rotor = 8
# num_chord_vlm_rotor = 2

# # Pusher prop
# pp_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=pp_disk,
#     origin=np.array([32.625, 0., 7.79]),
#     y1=np.array([31.94, 0.00, 3.29]),
#     y2=np.array([31.94, 0.00, 12.29]),
#     z1=np.array([31.94, -4.50, 7.78]),
#     z2=np.array([31.94, 4.45, 7.77]),
#     create_disk_mesh=False,
#     plot=False,
# )


# # Rear left outer
# rlo_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=rlo_disk,
#     origin=np.array([19.2, -18.75, 9.01]),
#     y1=np.array([19.2, -13.75, 9.01]),
#     y2=np.array([19.2, -23.75, 9.01]),
#     z1=np.array([14.2, -18.75, 9.01]),
#     z2=np.array([24.2, -18.75, 9.01]),
#     create_disk_mesh=False,
#     plot=False,
# )

# rlo_disk_origin_para = rlo_disk.project(np.array([19.2, -18.75, 9.01]))
rlo_disk_y1_para = rlo_disk.project(np.array([19.2, -13.75, 9.01]))
rlo_disk_y2_para = rlo_disk.project(np.array([19.2, -23.75, 9.01]))
# rlo_disk_z1_para = rlo_disk.project(np.array([14.2, -18.75, 9.01]))
# rlo_disk_z2_para = rlo_disk.project(np.array([24.2, -18.75, 9.01]))

# rlo_r1 = csdl.norm((geometry.evaluate(rlo_disk_y1_para) - geometry.evaluate(rlo_disk_y2_para))) 
# rlo_r2 = csdl.norm((geometry.evaluate(rlo_disk_z1_para) - geometry.evaluate(rlo_disk_z2_para)))

# # Rear right outer 
# rro_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=rro_disk,
#     origin=np.array([19.2, 18.75, 9.01]),
#     y1=np.array([19.2, 23.75, 9.01]),
#     y2=np.array([19.2, 13.75, 9.01]),
#     z1=np.array([14.2, 18.75, 9.01]),
#     z2=np.array([24.2, 18.75, 9.01]),
#     create_disk_mesh=False,
#     plot=False,
# )

# rro_disk_origin_para = rro_disk.project(np.array([19.2, 18.75, 9.01]))
rro_disk_y1_para = rro_disk.project(np.array([19.2, 23.75, 9.01]))
rro_disk_y2_para = rro_disk.project(np.array([19.2, 13.75, 9.01]))
# rro_disk_z1_para = rro_disk.project(np.array([14.2, 18.75, 9.01]))
# rro_disk_z2_para = rro_disk.project(np.array([24.2, 18.75, 9.01]))

# rro_r1 = csdl.norm((geometry.evaluate(rro_disk_y1_para) - geometry.evaluate(rro_disk_y2_para))) 
# rro_r2 = csdl.norm((geometry.evaluate(rro_disk_z1_para) - geometry.evaluate(rro_disk_z2_para)))

# # Front left outer 
# flo_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=flo_disk,
#     origin=np.array([5.07, -18.75, 6.73]),
#     y1=np.array([5.070, -13.750, 6.730]),
#     y2=np.array([5.070, -23.750, 6.730]),
#     z1=np.array([0.070, -18.750, 6.730]),
#     z2=np.array([10.070, -18.750, 6.730]),
#     create_disk_mesh=False,
#     plot=False,
# )

# flo_disk_origin_para = flo_disk.project(np.array([5.07, -18.75, 6.73]))
flo_disk_y1_para = flo_disk.project(np.array([5.070, -13.750, 6.730]))
flo_disk_y2_para = flo_disk.project(np.array([5.070, -23.750, 6.730]))
# flo_disk_z1_para = flo_disk.project(np.array([0.070, -18.750, 6.730]))
# flo_disk_z2_para = flo_disk.project(np.array([10.070, -18.750, 6.730]))

# flo_r1 = csdl.norm((geometry.evaluate(flo_disk_y1_para) - geometry.evaluate(flo_disk_y2_para))) 
# flo_r2 = csdl.norm((geometry.evaluate(flo_disk_z1_para) - geometry.evaluate(flo_disk_z2_para)))

# # Front right outer 
# fro_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=fro_disk,
#     origin=np.array([5.07, 18.75, 6.73]),
#     y1=np.array([5.070, 23.750, 6.730]),
#     y2=np.array([5.070, 13.750, 6.730]),
#     z1=np.array([0.070, 18.750, 6.730]),
#     z2=np.array([10.070, 18.750, 6.730]),
#     create_disk_mesh=False,
#     plot=False,
# )

# fro_disk_origin_para = fro_disk.project(np.array([5.07, 18.75, 6.73]))
fro_disk_y1_para = fro_disk.project(np.array([5.070, 23.750, 6.730]))
fro_disk_y2_para = fro_disk.project(np.array([5.070, 13.750, 6.730]))
# fro_disk_z1_para = fro_disk.project(np.array([0.070, 18.750, 6.730]))
# fro_disk_z2_para = fro_disk.project(np.array([10.070, 18.750, 6.730]))

# fro_r1 = csdl.norm((geometry.evaluate(fro_disk_y1_para) - geometry.evaluate(fro_disk_y2_para))) 
# fro_r2 = csdl.norm((geometry.evaluate(fro_disk_z1_para) - geometry.evaluate(fro_disk_z2_para)))

# # Rear left inner
# rli_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=rli_disk,
#     origin=np.array([18.760, -8.537, 9.919]),
#     y1=np.array([18.760, -3.499, 9.996]),
#     y2=np.array([18.760, -13.401, 8.604]),
#     z1=np.array([13.760, -8.450, 9.300]),
#     z2=np.array([23.760, -8.450, 9.300]),
#     create_disk_mesh=False,
#     plot=False,
# )

# rli_disk_origin_para = rli_disk.project(np.array([18.760, -8.537, 9.919]))
rli_disk_y1_para = rli_disk.project(np.array([18.760, -3.499, 9.996]))
rli_disk_y2_para = rli_disk.project(np.array([18.760, -13.401, 8.604]))
# rli_disk_z1_para = rli_disk.project(np.array([13.760, -8.450, 9.30]))
# rli_disk_z2_para = rli_disk.project(np.array([23.760, -8.450, 9.300]))

# rli_r1 = csdl.norm((geometry.evaluate(rli_disk_y1_para) - geometry.evaluate(rli_disk_y2_para))) 
# rli_r2 = csdl.norm((geometry.evaluate(rli_disk_z1_para) - geometry.evaluate(rli_disk_z2_para)))

# # Rear right inner
# rri_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=rri_disk,
#     origin=np.array([18.760, 8.537, 9.919]),
#     y1=np.array([18.760, 13.401, 8.604]),
#     y2=np.array([18.760, 3.499, 9.996]),
#     z1=np.array([13.760, 8.450, 9.300]),
#     z2=np.array([23.760, 8.450, 9.300]),
#     create_disk_mesh=False,
#     plot=False,
# )

# rri_disk_origin_para = rri_disk.project(np.array([18.760, 8.537, 9.919]))
rri_disk_y1_para = rri_disk.project(np.array([18.760, 13.401, 8.60]))
rri_disk_y2_para = rri_disk.project(np.array([18.760, 3.499, 9.996]))
# rri_disk_z1_para = rri_disk.project(np.array([13.760, 8.450, 9.300]))
# rri_disk_z2_para = rri_disk.project(np.array([23.760, 8.450, 9.300]))

# rri_r1 = csdl.norm((geometry.evaluate(rri_disk_y1_para) - geometry.evaluate(rri_disk_y2_para))) 
# rri_r2 = csdl.norm((geometry.evaluate(rri_disk_z1_para) - geometry.evaluate(rri_disk_z2_para)))

# # Front left inner
# fli_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=fli_disk,
#     origin=np.array([4.630, -8.217, 7.659]),
#     y1=np.array([4.630, -3.179, 7.736]),
#     y2=np.array([4.630, -13.081, 6.344]),
#     z1=np.array([-0.370, -8.130, 7.040]),
#     z2=np.array([9.630, -8.130, 7.040]),
#     create_disk_mesh=False,
#     plot=False,
# )

# fli_disk_origin_para = fli_disk.project(np.array([4.630, -8.217, 7.659]))
fli_disk_y1_para = fli_disk.project(np.array([4.630, -3.179, 7.736]))
fli_disk_y2_para = fli_disk.project(np.array([4.630, -13.081, 6.344]))
# fli_disk_z1_para = fli_disk.project(np.array([-0.370, -8.130, 7.040]))
# fli_disk_z2_para = fli_disk.project(np.array([9.630, -8.130, 7.040]))

# fli_r1 = csdl.norm((geometry.evaluate(fli_disk_y1_para) - geometry.evaluate(fli_disk_y2_para))/2) 
# fli_r2 = csdl.norm((geometry.evaluate(fli_disk_z1_para) - geometry.evaluate(fli_disk_z2_para))/2)

# # Front right inner
# fri_mesh = make_rotor_mesh(
#     geometry=geometry,
#     num_radial=num_radial,
#     disk_component=fri_disk,
#     origin=np.array([4.630, 8.217, 7.659]), 
#     y1=np.array([4.630, 13.081, 6.344]),
#     y2=np.array([4.630, 3.179, 7.736]),
#     z1=np.array([-0.370, 8.130, 7.040]),
#     z2=np.array([9.630, 8.130, 7.040]),
#     create_disk_mesh=False,
#     plot=False,
# )

# fri_disk_origin_para = fri_disk.project(np.array([4.630, 8.217, 7.659]))
fri_disk_y1_para = fri_disk.project(np.array([4.630, 13.081, 6.344]))
fri_disk_y2_para = fri_disk.project(np.array([4.630, 3.179, 7.736]))
# fri_disk_z1_para = fri_disk.project(np.array([-0.370, 8.130, 7.040]))
# fri_disk_z2_para = fri_disk.project(np.array([9.630, 8.130, 7.04]))

# fri_r1 = csdl.norm((geometry.evaluate(fri_disk_y1_para) - geometry.evaluate(fri_disk_y2_para))/2) 
# fri_r2 = csdl.norm((geometry.evaluate(fri_disk_z1_para) - geometry.evaluate(fri_disk_z2_para))/2)

# radius_1_list = [rlo_r1, rli_r1, rri_r1, rro_r1,
#                  flo_r1, fli_r1, fri_r1, fro_r1]

# radius_2_list = [rlo_r2, rli_r2, rri_r2, rro_r2,
#                  flo_r2, fli_r2, fri_r2, fro_r2]
rotor_edges = [(rlo_disk_y1_para, rlo_disk_y2_para), (rli_disk_y1_para, rli_disk_y2_para),
                (rri_disk_y1_para, rri_disk_y2_para), (rro_disk_y1_para, rro_disk_y2_para),
                (flo_disk_y1_para, flo_disk_y2_para), (fli_disk_y1_para, fli_disk_y2_para),
                (fri_disk_y1_para, fri_disk_y2_para), (fro_disk_y1_para, fro_disk_y2_para)]

fuselage_fri_collision_point_parametric = fuselage.project(geometry.evaluate(fri_disk_y2_para))
fuselage_fli_collision_point_parametric = fuselage.project(geometry.evaluate(fli_disk_y1_para))
# endregion

# region Projection for meshes
# region Wing camber mesh
wing_num_spanwise_vlm = 23
wing_num_chordwise_vlm = 5
wing_leading_edge_line_parametric = wing.project(np.linspace(np.array([8.356, -26., 7.618]), np.array([8.356, 26., 7.618]), wing_num_spanwise_vlm), 
                                 direction=np.array([0., 0., -1.]), grid_search_density_parameter=10.)
wing_trailing_edge_line_parametric = wing.project(np.linspace(np.array([15.4, -25.250, 7.5]), np.array([15.4, 25.250, 7.5]), wing_num_spanwise_vlm), 
                                  direction=np.array([0., 0., -1.]), grid_search_density_parameter=10.)
wing_leading_edge_line = geometry.evaluate(wing_leading_edge_line_parametric)
wing_trailing_edge_line = geometry.evaluate(wing_trailing_edge_line_parametric)
wing_chord_surface = csdl.linear_combination(wing_leading_edge_line, wing_trailing_edge_line, wing_num_chordwise_vlm)
wing_upper_surface_wireframe_parametric = wing.project(wing_chord_surface.value.reshape((wing_num_chordwise_vlm,wing_num_spanwise_vlm,3))+np.array([0., 0., 1.]), 
                                       direction=np.array([0., 0., -1.]), plot=False, grid_search_density_parameter=10.)
wing_lower_surface_wireframe_parametric = wing.project(wing_chord_surface.value.reshape((wing_num_chordwise_vlm,wing_num_spanwise_vlm,3))+np.array([0., 0., -1.]), 
                                       direction=np.array([0., 0., 1.]), plot=False, grid_search_density_parameter=10.)
wing_upper_surface_wireframe = geometry.evaluate(wing_upper_surface_wireframe_parametric)
wing_lower_surface_wireframe = geometry.evaluate(wing_lower_surface_wireframe_parametric)
wing_camber_surface = csdl.linear_combination(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1).reshape((wing_num_chordwise_vlm, wing_num_spanwise_vlm, 3))
# geometry.plot_meshes([wing_camber_surface], function_opacity=0.5, mesh_color='#FFCD00', show=True)
# endregion Wing camber mesh

# region Htail camber mesh
h_tail_num_spanwise_vlm = 11
h_tail_num_chordwise_vlm = 4
h_tail_leading_edge_line_parametric = h_tail.project(np.linspace(np.array([26.5, -6.75, 6.]), np.array([26.5, 6.75, 6.]), h_tail_num_spanwise_vlm), 
                                 direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
h_tail_trailing_edge_line_parametric = h_tail.project(np.linspace(np.array([31.5, -6.75, 6.]), np.array([31.5, 6.75, 6.]), h_tail_num_spanwise_vlm), 
                                  direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
h_tail_leading_edge_line = geometry.evaluate(h_tail_leading_edge_line_parametric)
h_tail_trailing_edge_line = geometry.evaluate(h_tail_trailing_edge_line_parametric)
h_tail_chord_surface = csdl.linear_combination(h_tail_leading_edge_line, h_tail_trailing_edge_line, h_tail_num_chordwise_vlm)
h_tail_upper_surface_wireframe_parametric = h_tail.project(h_tail_chord_surface.value.reshape((h_tail_num_chordwise_vlm,h_tail_num_spanwise_vlm,3))+np.array([0., 0., 1.]), 
                                       direction=np.array([0., 0., -1.]), plot=False, grid_search_density_parameter=20.)
h_tail_lower_surface_wireframe_parametric = h_tail.project(h_tail_chord_surface.value.reshape((h_tail_num_chordwise_vlm,h_tail_num_spanwise_vlm,3))+np.array([0., 0., -1.]), 
                                       direction=np.array([0., 0., 1.]), plot=False, grid_search_density_parameter=20.)
h_tail_upper_surface_wireframe = geometry.evaluate(h_tail_upper_surface_wireframe_parametric)
h_tail_lower_surface_wireframe = geometry.evaluate(h_tail_lower_surface_wireframe_parametric)
h_tail_camber_surface = csdl.linear_combination(h_tail_upper_surface_wireframe, h_tail_lower_surface_wireframe, 1).reshape((h_tail_num_chordwise_vlm, h_tail_num_spanwise_vlm, 3))
# endregion Htail camber mesh

# region Wing beam mesh
num_beam_nodes = 13
# wing_qc_right_physical = np.array([12.617, 25.250, 7.5])
# wing_qc_left_physical = np.array([12.617, -25.250, 7.5])
# wing_qc_center_physical = np.array([10.25, 0., 8.5])
wing_qc_right_physical = np.array([12.517, 25.250, 7.5])
wing_qc_left_physical = np.array([12.517, -25.250, 7.5])
wing_qc_center_physical = np.array([10.5, 0., 8.5])

left_physical = np.linspace(wing_qc_left_physical, wing_qc_center_physical, num_beam_nodes//2, endpoint=False)
right_physical = np.linspace(wing_qc_center_physical, wing_qc_right_physical, num_beam_nodes//2+1, endpoint=True)
beam_mesh_physical = np.concatenate((left_physical, right_physical), axis=0)
beam_top_parametric = wing.project(beam_mesh_physical+np.array([0., 0., 1.]), plot=False)
beam_bottom_parametric = wing.project(beam_mesh_physical+np.array([0., 0., -1.]), plot=False)
beam_tops = geometry.evaluate(beam_top_parametric)
beam_bottoms = geometry.evaluate(beam_bottom_parametric)
wing_beam_mesh = csdl.linear_combination(beam_tops, beam_bottoms, 1).reshape((num_beam_nodes, 3))
beam_heights = csdl.norm(beam_tops - beam_bottoms, axes=(1,))
# endregion Wing beam mesh

# # Figure plotting the meshes
# plotting_elements = geometry.plot_meshes([wing_camber_surface, h_tail_camber_surface], function_opacity=0.5, mesh_color='#FFCD00', show=False)
# plotting_elements = geometry.plot_meshes([wing_beam_mesh], mesh_line_width=10, function_opacity=0., additional_plotting_elements=plotting_elements, show=False)
# lfs.show_plot(plotting_elements, 'Meshes', axes=False, view_up='z')
# endregion
# plotter.show(plotting_elements, axes=0, viewup='z'

# region Parameterization
wing_span_computed = geometry.evaluate(wing_le_right)[1] - geometry.evaluate(wing_le_left)[1]
wing_root_chord_computed = geometry.evaluate(wing_te_center)[0] - geometry.evaluate(wing_le_center)[0]
wing_tip_chord_left_computed = geometry.evaluate(wing_te_left)[0] - geometry.evaluate(wing_le_left)[0]
wing_tip_chord_right_computed = geometry.evaluate(wing_te_right)[0] - geometry.evaluate(wing_le_right)[0]
wing_leading_edge_line = geometry.evaluate(wing_leading_edge_line_parametric)
wing_trailing_edge_line = geometry.evaluate(wing_trailing_edge_line_parametric)
wing_chord_surface = csdl.linear_combination(wing_leading_edge_line, wing_trailing_edge_line, wing_num_chordwise_vlm)
u_vectors = wing_chord_surface[1:,:] - wing_chord_surface[:-1,:]
v_vectors = wing_chord_surface[:,1:] - wing_chord_surface[:,:-1]
wing_panel_areas = 1/2*csdl.cross(u_vectors[:,:-1], v_vectors[:-1,:], axis=2) \
                 + 1/2*csdl.cross(u_vectors[:,1:], v_vectors[1:,:], axis=2)  # (num_chordwise_vlm-1, num_spanwise_vlm-1, 3)
wing_panel_areas = csdl.norm(wing_panel_areas, axes=(2,))  # (num_chordwise_vlm-1, num_spanwise_vlm-1)
wing_area_computed = csdl.sum(wing_panel_areas)
wing_aspect_ratio_computed = wing_span_computed**2/wing_area_computed
wing_taper_ratio_computed = (wing_tip_chord_left_computed + wing_tip_chord_right_computed)/2/wing_root_chord_computed
wing_qc_line_left = geometry.evaluate(wing_qc_left) - geometry.evaluate(wing_qc_center)
wing_sweep_left = csdl.arctan(wing_qc_line_left[0]/(-wing_qc_line_left[1]))
wing_qc_line_right = geometry.evaluate(wing_qc_right) - geometry.evaluate(wing_qc_center)
wing_sweep_right = csdl.arctan(wing_qc_line_right[0]/wing_qc_line_right[1])
wing_sweep_computed = 1/2*(wing_sweep_left + wing_sweep_right)
wing_area = csdl.Variable(name='wing_area', value=wing_area_computed.value)
wing_aspect_ratio = csdl.Variable(name='aspect_ratio', value=wing_aspect_ratio_computed.value)
wing_taper_ratio = csdl.Variable(name='taper_ratio', value=wing_taper_ratio_computed.value)
wing_sweep = csdl.Variable(name='wing_sweep', value=wing_sweep_computed.value)

# wing_span = csdl.Variable(name='wing_span', value=np.array([100.]))
# wing_span = csdl.Variable(name='wing_span', value=wing_span_computed.value)
# wing_span = csdl.Variable(name='wing_span', value=37.)
# wing_span = csdl.Variable(name='wing_span', value=np.array([45.]))
# wing_span = csdl.Variable(name='wing_span', value=np.array([48.]))
# wing_root_chord = csdl.Variable(name='wing_root_chord', value=wing_root_chord_computed.value)
# wing_tip_chord = csdl.Variable(name='wing_tip_chord_left', value=wing_tip_chord_left_computed.value)

h_tail_span_computed = csdl.norm(geometry.evaluate(tail_le_right) - geometry.evaluate(tail_le_left))
h_tail_root_chord_computed = csdl.norm(geometry.evaluate(tail_te_center) - geometry.evaluate(tail_le_center))
h_tail_tip_chord_left_computed = csdl.norm(geometry.evaluate(tail_te_left) - geometry.evaluate(tail_le_left))
h_tail_tip_chord_right_computed = csdl.norm(geometry.evaluate(tail_te_right) - geometry.evaluate(tail_le_right))
h_tail_leading_edge_line = geometry.evaluate(h_tail_leading_edge_line_parametric)
h_tail_trailing_edge_line = geometry.evaluate(h_tail_trailing_edge_line_parametric)
h_tail_chord_surface = csdl.linear_combination(h_tail_leading_edge_line, h_tail_trailing_edge_line, h_tail_num_chordwise_vlm)
u_vectors = h_tail_chord_surface[1:,:] - h_tail_chord_surface[:-1,:]
v_vectors = h_tail_chord_surface[:,1:] - h_tail_chord_surface[:,:-1]
h_tail_panel_areas = 1/2*csdl.cross(u_vectors[:,:-1], v_vectors[:-1,:], axis=2) \
                 + 1/2*csdl.cross(u_vectors[:,1:], v_vectors[1:,:], axis=2)  # (num_chordwise_vlm-1, num_spanwise_vlm-1, 3)
h_tail_panel_areas = csdl.norm(h_tail_panel_areas, axes=(2,))  # (num_chordwise_vlm-1, num_spanwise_vlm-1)
h_tail_area_computed = csdl.sum(h_tail_panel_areas)
h_tail_aspect_ratio_computed = h_tail_span_computed**2/h_tail_area_computed
h_tail_taper_ratio_computed = (h_tail_tip_chord_left_computed + h_tail_tip_chord_right_computed)/2/h_tail_root_chord_computed
h_tail_qc_line_left = geometry.evaluate(tail_qc_left) - geometry.evaluate(tail_qc_center)
h_tail_sweep_left = csdl.arctan(h_tail_qc_line_left[0]/(-h_tail_qc_line_left[1]))
h_tail_qc_line_right = geometry.evaluate(tail_qc_right) - geometry.evaluate(tail_qc_center)
h_tail_sweep_right = csdl.arctan(h_tail_qc_line_right[0]/h_tail_qc_line_right[1])
h_tail_sweep_computed = 1/2*(h_tail_sweep_left + h_tail_sweep_right)
h_tail_area = csdl.Variable(name='h_tail_area', value=h_tail_area_computed.value)
h_tail_aspect_ratio = csdl.Variable(name='h_tail_aspect_ratio', value=h_tail_aspect_ratio_computed.value)
h_tail_taper_ratio = csdl.Variable(name='h_tail_taper_ratio', value=h_tail_taper_ratio_computed.value)
h_tail_sweep = csdl.Variable(name='h_tail_sweep', value=h_tail_sweep_computed.value)
# h_tail_span = csdl.Variable(name='h_tail_span', value=h_tail_span_computed.value)
# h_tail_root_chord = csdl.Variable(name='h_tail_root_chord', value=h_tail_root_chord_computed.value)
# h_tail_tip_chord = csdl.Variable(name='h_tail_tip_chord_left', value=h_tail_tip_chord_left_computed.value)

tail_moment_arm_computed = csdl.norm(geometry.evaluate(tail_qc_center) - geometry.evaluate(wing_qc_center))
tail_moment_arm = csdl.Variable(name='tail_moment_arm', value=tail_moment_arm_computed.value)


outer_rotors_radius_computed = csdl.norm(geometry.evaluate(rotor_edges[0][1]) - geometry.evaluate(rotor_edges[0][0]))/2
inner_rotors_radius_computed = csdl.norm(geometry.evaluate(rotor_edges[1][1]) - geometry.evaluate(rotor_edges[1][0]))/2
flo_radius = fro_radius = front_outer_radius = csdl.Variable(name='front_outer_radius', value=outer_rotors_radius_computed.value)
fli_radius = fri_radius = front_inner_radius = csdl.Variable(name='front_inner_radius', value=inner_rotors_radius_computed.value)
rlo_radius = rro_radius = rear_outer_radius = csdl.Variable(name='rear_outer_radius', value=outer_rotors_radius_computed.value)
rli_radius = rri_radius = rear_inner_radius = csdl.Variable(name='rear_inner_radius', value=inner_rotors_radius_computed.value)

# wing_aspect_ratio.set_value(wing_aspect_ratio.value*0.75)
# h_tail_area.set_value(h_tail_area.value*1.25)
# tail_moment_arm.set_value(tail_moment_arm.value*1.25)

# wing_aspect_ratio.set_value(wing_aspect_ratio.value*1.5)
# h_tail_area.set_value(h_tail_area.value*1.25)
# tail_moment_arm.set_value(tail_moment_arm.value*1.25)

# wing_aspect_ratio.set_value(wing_aspect_ratio.value*0.9)
# wing_aspect_ratio.set_value(wing_aspect_ratio.value*3.)


constant_b_spline_curve_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
linear_b_spline_curve_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
linear_b_spline_curve_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
cubic_b_spline_curve_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))

# region Parameterization Setup
initial_wing_chord_stretch_coefficients = csdl.Variable(value=np.zeros((3,)), name='initial_guess_wing_chord_stretch_coefficients')
initial_wing_linear_stretch = csdl.Variable(value=0., name='initial_wing_linear_stretch')
initial_wing_midspan_stretch = csdl.Variable(value=0., name='initial_wing_midspan_stretch')
initial_wing_rigid_body_translation = csdl.Variable(value=np.zeros((3,)), name='initial_wing_rigid_body_translation')

initial_h_tail_chord_stretch_coefficients = csdl.Variable(value=np.zeros((3,)), name='initial_h_tail_chord_stretch_coefficients')
initial_h_tail_span_stretch_coefficients = csdl.Variable(value=np.zeros((2,)), name='initial_h_tail_span_stretch_coefficients')
initial_h_tail_translation_x_coefficients = csdl.Variable(value=np.zeros((1,)), name='initial_h_tail_translation_x_coefficients')
initial_h_tail_translation_z_coefficients = csdl.Variable(value=np.zeros((1,)), name='initial_h_tail_translation_z_coefficients')

initial_fuselage_stretch_coefficients = csdl.Variable(value=np.zeros((2,)), name='initial_fuselage_stretch_coefficients')

initial_lift_rotor_stretch_coefficients : list[csdl.Variable] = []
initial_lift_rotor_rigid_body_translations : list[csdl.Variable] = []
for i, component_set in enumerate(lift_rotor_related_components):
    initial_stretch = csdl.Variable(value=0., name=f'initial_{component_set[0].name[:3]}_rotor_stretch_coefficient')
    initial_lift_rotor_stretch_coefficients.append(initial_stretch)
    initial_translation = csdl.Variable(value=np.zeros((3,)), name=f'initial_{component_set[0].name[:3]}_rotor_rigid_body_translation')
    initial_lift_rotor_rigid_body_translations.append(initial_translation)

initial_pusher_prop_rigid_body_translation = csdl.Variable(value=np.zeros((3,)), name='initial_pusher_prop_rigid_body_translation')

initial_v_tail_rigid_body_translation = csdl.Variable(value=np.zeros((3,)), name='initial_v_tail_rigid_body_translation')

initial_linear_penalty_factor_inner_outer = csdl.Variable(value=0., name='initial_linear_penalty_factor_inner_outer')
initial_linear_penalty_factor_inner_inner = csdl.Variable(value=0., name='initial_linear_penalty_factor_inner_inner')
# initial_quadratic_penalty_factor_inner_outer = csdl.Variable(value=2.e2, name='initial_quadratic_penalty_factor_inner_outer')
# initial_quadratic_penalty_factor_inner_outer = csdl.Variable(value=0., name='initial_quadratic_penalty_factor_inner_outer')
initial_quadratic_penalty_factor_inner_outer = None
# initial_quadratic_penalty_factor_inner_outer = csdl.Variable(value=2000., name='initial_quadratic_penalty_factor_inner_outer')
# initial_quadratic_penalty_factor_inner_inner = csdl.Variable(value=2.e3, name='initial_quadratic_penalty_factor_inner_inner')
# initial_quadratic_penalty_factor_inner_inner = csdl.Variable(value=0., name='initial_quadratic_penalty_factor_inner_inner')
initial_quadratic_penalty_factor_inner_inner = None
# initial_quadratic_penalty_factor_inner_inner = csdl.Variable(value=3000., name='initial_quadratic_penalty_factor_inner_inner')

# with csdl.experimental.enter_loop(1) as loop_builder:
# i = loop_builder.get_loop_indices()
# initial_guess_wing_chord_stretch_coefficients = loop_builder.initialize_feedback(initial_wing_chord_stretch_coefficients)
# initial_guess_wing_linear_stretch = loop_builder.initialize_feedback(initial_wing_linear_stretch)
# initial_guess_wing_midspan_stretch = loop_builder.initialize_feedback(initial_wing_midspan_stretch)
# initial_guess_wing_rigid_body_translation = loop_builder.initialize_feedback(initial_wing_rigid_body_translation)

# initial_guess_h_tail_chord_stretch_coefficients = loop_builder.initialize_feedback(initial_h_tail_chord_stretch_coefficients)
# initial_guess_h_tail_span_stretch_coefficients = loop_builder.initialize_feedback(initial_h_tail_span_stretch_coefficients)
# initial_guess_h_tail_translation_x_coefficients = loop_builder.initialize_feedback(initial_h_tail_translation_x_coefficients)
# initial_guess_h_tail_translation_z_coefficients = loop_builder.initialize_feedback(initial_h_tail_translation_z_coefficients)

# initial_guess_fuselage_stretch_coefficients = loop_builder.initialize_feedback(initial_fuselage_stretch_coefficients)

# initial_guess_lift_rotor_stretch_coefficients : list[csdl.Variable] = []
# initial_guess_lift_rotor_rigid_body_translations : list[csdl.Variable] = []
# for j, component_set in enumerate(lift_rotor_related_components):
#     initial_guess = loop_builder.initialize_feedback(initial_lift_rotor_stretch_coefficients[j])
#     initial_guess_lift_rotor_stretch_coefficients.append(initial_guess)
#     initial_guess_translation = loop_builder.initialize_feedback(initial_lift_rotor_rigid_body_translations[j])
#     initial_guess_lift_rotor_rigid_body_translations.append(initial_guess_translation)

# initial_guess_pusher_prop_rigid_body_translation = loop_builder.initialize_feedback(initial_pusher_prop_rigid_body_translation)

# initial_guess_v_tail_rigid_body_translation = loop_builder.initialize_feedback(initial_v_tail_rigid_body_translation)

# linear_penalty_factor_inner_outer = loop_builder.initialize_feedback(initial_linear_penalty_factor_inner_outer)
# linear_penalty_factor_inner_inner = loop_builder.initialize_feedback(initial_linear_penalty_factor_inner_inner)
# quadratic_penalty_factor_inner_outer = loop_builder.initialize_feedback(initial_quadratic_penalty_factor_inner_outer)
# quadratic_penalty_factor_inner_inner = loop_builder.initialize_feedback(initial_quadratic_penalty_factor_inner_inner)
linear_penalty_factor_inner_outer = initial_linear_penalty_factor_inner_outer
linear_penalty_factor_inner_inner = initial_linear_penalty_factor_inner_inner
quadratic_penalty_factor_inner_outer = initial_quadratic_penalty_factor_inner_outer
quadratic_penalty_factor_inner_inner = initial_quadratic_penalty_factor_inner_inner

parameterization_solver = lsdo_geo.ParameterizationSolver()
parameterization_design_parameters = lsdo_geo.GeometricVariables()

# region Wing Parameterization setup
corners = np.zeros((2,7,2,3))
corners[0,:,:,0] = 8.892
corners[1,:,:,0] = 14.333
corners[:,0,:,1] = -25.25
corners[:,1,:,1] = -17.
corners[:,2,:,1] = -14.736
corners[:,3,:,1] = 0.
corners[:,4,:,1] = 14.736
corners[:,5,:,1] = 17.
corners[:,6,:,1] = 25.25
corners[:,:,0,2] = 7.5
corners[:,:,1,2] = 9.133
wing_ffd_block = lsdo_geo.construct_ffd_block_from_corners(entities=wing, corners=corners)
# wing_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2,11,2), degree=(1,3,1))
wing_ffd_block_sectional_parameterization = lsdo_geo.SectionalParameterization(name='wing_sectional_parameterization',
                                                                            parameterized_points=wing_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

# plotting_element = wing_ffd_block.plot()
# geometry.plot(additional_plotting_elements=plotting_element, show=True)
# exit()

wing_root_chord_stretch = csdl.Variable(name='wing_root_chord_stretch', value=0.)
wing_tip_chord_stretch = csdl.Variable(name='wing_tip_chord_stretch', value=0.)
wing_chord_stretch_coefficients = csdl.Variable(name='wing_chord_stretch_coefficients', shape=(3,), value=0.)
wing_chord_stretch_coefficients = wing_chord_stretch_coefficients.set(csdl.slice[1], wing_root_chord_stretch)
wing_chord_stretch_coefficients = wing_chord_stretch_coefficients.set(csdl.slice[[0, -1]], wing_tip_chord_stretch)
wing_chord_stretch_b_spline = lfs.Function(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                        coefficients=wing_chord_stretch_coefficients)

# wing_wingspan_stretch_coefficients = csdl.Variable(name='wing_wingspan_stretch_coefficients', value=np.array([-0., 0.]))
# wing_wingspan_stretch_b_spline = lfs.Function(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                           coefficients=wing_wingspan_stretch_coefficients)
wing_linear_stretch = csdl.Variable(name='wing_linear_stretch', value=0.)
wing_midspan_stretch = csdl.Variable(name='wing_midspan_stretch', value=0.)

wing_twist_coefficients = csdl.Variable(name='wing_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
wing_twist_b_spline = lfs.Function(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                        coefficients=wing_twist_coefficients)

wing_tip_shear = csdl.Variable(name='wing_tip_shear', value=0.)
wing_sweep_translation_coefficients = wing_tip_shear*np.array([1., 0., 1.])
wing_sweep_translation_b_spline = lfs.Function(name='wing_sweep_translation_b_spline', space=linear_b_spline_curve_3_dof_space,
                                          coefficients=wing_sweep_translation_coefficients)

wing_rigid_body_translation = csdl.Variable(name='wing_rigid_body_translation', shape=(3,), value=np.array([0., 0., 0.]))

# parameterization_solver.add_state(state=wing_chord_stretch_coefficients)
parameterization_solver.add_state(state=wing_root_chord_stretch)
parameterization_solver.add_state(state=wing_tip_chord_stretch)
# parameterization_solver.add_state(state=wing_wingspan_stretch_coefficients, cost=1.e3)
parameterization_solver.add_state(state=wing_linear_stretch, cost=1.e1)
parameterization_solver.add_state(state=wing_midspan_stretch, cost=1.e1)
parameterization_solver.add_state(state=wing_tip_shear)
parameterization_solver.add_state(state=wing_rigid_body_translation)

# parameterization_solver.add_state(state=wing_chord_stretch_coefficients, initial_value=initial_guess_wing_chord_stretch_coefficients)
# # parameterization_solver.add_state(state=wing_wingspan_stretch_coefficients, cost=1.e3)
# parameterization_solver.add_state(state=wing_linear_stretch, cost=1.e1, initial_value=initial_guess_wing_linear_stretch)
# parameterization_solver.add_state(state=wing_midspan_stretch, cost=1.e1, initial_value=initial_guess_wing_midspan_stretch)
# # parameterization_solver.add_state(state=wing_translation_x_coefficients)
# # parameterization_solver.add_state(state=wing_translation_z_coefficients)
# parameterization_solver.add_state(state=wing_rigid_body_translation, initial_value=initial_guess_wing_rigid_body_translation)
# endregion Wing Parameterization setup

# region Horizontal Stabilizer setup
h_tail_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name='h_tail_ffd_block', entities=h_tail, num_coefficients=(2,3,2), degree=(1,1,1))
h_tail_ffd_block_sectional_parameterization = lsdo_geo.SectionalParameterization(name='h_tail_sectional_parameterization',
                                                                            parameterized_points=h_tail_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

# h_tail_chord_stretch_coefficients = csdl.Variable(name='h_tail_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
# h_tail_chord_stretch_b_spline = lfs.Function(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
#                                         coefficients=h_tail_chord_stretch_coefficients)

# h_tail_span_stretch_coefficients = csdl.Variable(name='h_tail_span_stretch_coefficients', value=np.array([-0., 0.]))
# h_tail_span_stretch_b_spline = lfs.Function(name='h_tail_span_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
#                                         coefficients=h_tail_span_stretch_coefficients)

# h_tail_twist_coefficients = csdl.Variable(name='h_tail_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
# h_tail_twist_b_spline = lfs.Function(name='h_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#                                           coefficients=h_tail_twist_coefficients)

# h_tail_translation_x_coefficients = csdl.Variable(name='h_tail_translation_x_coefficients', value=np.array([0.]))
# h_tail_translation_x_b_spline = lfs.Function(name='h_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=h_tail_translation_x_coefficients)
# h_tail_translation_z_coefficients = csdl.Variable(name='h_tail_translation_z_coefficients', value=np.array([0.]))
# h_tail_translation_z_b_spline = lfs.Function(name='h_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
#                                         coefficients=h_tail_translation_z_coefficients)

h_tail_sectional_chord_stretch_dof = csdl.Variable(name='h_tail_sectional_chord_stretch_dof', shape=(2,), value=np.array([0., 0.]))
h_tail_sectional_span_stretch_dof = csdl.Variable(name='h_tail_sectional_span_stretch_dof', shape=(1,), value=0.)
h_tail_sectional_twist_dof = csdl.Variable(name='h_tail_sectional_twist_dof', shape=(2,), value=np.array([0., 0.]))
h_tail_sectional_sweep_shear_dof = csdl.Variable(name='h_tail_sectional_sweep_shear_dof', shape=(1,), value=0.)

h_tail_sectional_chord_stretch = csdl.Variable(name='h_tail_sectional_chord_stretch', shape=(h_tail_ffd_block_sectional_parameterization.num_sections,), value=np.zeros((h_tail_ffd_block_sectional_parameterization.num_sections,)))
h_tail_sectional_chord_stretch = h_tail_sectional_chord_stretch.set(csdl.slice[[0,-1]], h_tail_sectional_chord_stretch_dof[0])
h_tail_sectional_chord_stretch = h_tail_sectional_chord_stretch.set(csdl.slice[1], h_tail_sectional_chord_stretch_dof[1])
h_tail_sectional_span_stretch = h_tail_sectional_span_stretch_dof*np.array([-1., 0., 1.])
h_tail_sectional_twist = csdl.Variable(name='h_tail_sectional_twist', shape=(h_tail_ffd_block_sectional_parameterization.num_sections,), value=np.zeros((h_tail_ffd_block_sectional_parameterization.num_sections,)))
h_tail_sectional_twist = h_tail_sectional_twist.set(csdl.slice[[0,-1]], h_tail_sectional_twist_dof[0])
h_tail_sectional_twist = h_tail_sectional_twist.set(csdl.slice[1], h_tail_sectional_twist_dof[1])
h_tail_sectional_sweep_shear = h_tail_sectional_sweep_shear_dof*np.array([1., 0., 1.])

h_tail_rigid_body_translation = csdl.Variable(name='h_tail_rigid_body_translation', shape=(3,), value=np.array([0., 0., 0.]))

parameterization_solver.add_state(state=h_tail_sectional_chord_stretch_dof)
parameterization_solver.add_state(state=h_tail_sectional_span_stretch_dof)
parameterization_solver.add_state(state=h_tail_sectional_sweep_shear_dof)
parameterization_solver.add_state(state=h_tail_rigid_body_translation)

# parameterization_solver.add_state(state=h_tail_chord_stretch_coefficients)
# parameterization_solver.add_state(state=h_tail_span_stretch_coefficients)
# parameterization_solver.add_state(state=h_tail_translation_x_coefficients)
# parameterization_solver.add_state(state=h_tail_translation_z_coefficients)

# parameterization_solver.add_state(state=h_tail_chord_stretch_coefficients, initial_value=initial_guess_h_tail_chord_stretch_coefficients)
# parameterization_solver.add_state(state=h_tail_span_stretch_coefficients, initial_value=initial_guess_h_tail_span_stretch_coefficients)
# parameterization_solver.add_state(state=h_tail_translation_x_coefficients, initial_value=initial_guess_h_tail_translation_x_coefficients)
# parameterization_solver.add_state(state=h_tail_translation_z_coefficients, initial_value=initial_guess_h_tail_translation_z_coefficients)
# endregion Horizontal Stabilizer setup

# region Fuselage setup
fuselage_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=[fuselage, nose_hub], num_coefficients=(2,2,2), degree=(1,1,1))
fuselage_ffd_block_sectional_parameterization = lsdo_geo.SectionalParameterization(name='fuselage_sectional_parameterization',
                                                                            parameterized_points=fuselage_ffd_block.coefficients,
                                                                            principal_parametric_dimension=0)
# fuselage_ffd_block_sectional_parameterization.add_translation(name='sectional_fuselage_stretch', axis=0)

fuselage_stretch_coefficients = csdl.Variable(name='fuselage_stretch_coefficients', shape=(2,), value=np.array([0., -0.]))
fuselage_stretch_b_spline = lfs.Function(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                        coefficients=fuselage_stretch_coefficients)

parameterization_solver.add_state(state=fuselage_stretch_coefficients)
# endregion

# region Lift Rotors setup
lift_rotor_ffd_blocks = []
lift_rotor_sectional_parameterizations = []
lift_rotor_parameterization_b_splines = []
lift_rotor_stretch_coefficients = []
for i, component_set in enumerate(lift_rotor_related_components):
    rotor_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name=f'{component_set[0].name[:3]}_rotor_ffd_block', entities=component_set, num_coefficients=(2,2,2), degree=(1,1,1))
    rotor_ffd_block_sectional_parameterization = lsdo_geo.SectionalParameterization(name=f'{component_set[0].name[:3]}_rotor_sectional_parameterization',
                                                                                parameterized_points=rotor_ffd_block.coefficients,
                                                                                principal_parametric_dimension=2)
    
    rotor_stretch_coefficient = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_stretch_coefficient', shape=(1,), value=0.)
    lift_rotor_stretch_coefficients.append(rotor_stretch_coefficient)
    lift_rotor_sectional_stretch_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_sectional_stretch_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                                coefficients=rotor_stretch_coefficient)
    
    lift_rotor_ffd_blocks.append(rotor_ffd_block)
    lift_rotor_sectional_parameterizations.append(rotor_ffd_block_sectional_parameterization)
    lift_rotor_parameterization_b_splines.append(lift_rotor_sectional_stretch_b_spline)                 

    parameterization_solver.add_state(state=rotor_stretch_coefficient, cost=3.e1)
    # parameterization_solver.add_state(state=rotor_stretch_coefficient, cost=3.e1, initial_value=initial_guess_lift_rotor_stretch_coefficients[i])
# endregion Lift Rotors setup

# # region Plot parameterization
# plotting_elements = []
# plotting_elements = geometry.plot(color='#00629B', additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = wing_ffd_block.plot(opacity=0.25, color='#B6B1A9', plot_embedded_points=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = h_tail_ffd_block.plot(opacity=0.25, color='#B6B1A9', plot_embedded_points=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = fuselage_ffd_block.plot(opacity=0.25, color='#B6B1A9', plot_embedded_points=False, additional_plotting_elements=plotting_elements, show=False)
# for rotor_ffd_block in lift_rotor_ffd_blocks:
#     plotting_elements = rotor_ffd_block.plot(opacity=0.25, color='#B6B1A9', plot_embedded_points=False, additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = wing_ffd_block_sectional_parameterization.plot(opacity=0.5, color='#182B49', additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = h_tail_ffd_block_sectional_parameterization.plot(opacity=0.5, color='#182B49', additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = fuselage_ffd_block_sectional_parameterization.plot(opacity=0.5, color='#182B49', additional_plotting_elements=plotting_elements, show=False)
# for rotor_ffd_block_sectional_parameterization in lift_rotor_sectional_parameterizations:
#     plotting_elements = rotor_ffd_block_sectional_parameterization.plot(opacity=0.5, color='#182B49', additional_plotting_elements=plotting_elements, show=False)

# lfs.show_plot(plotting_elements, 'Parameterization', axes=False, view_up='z')
# exit()

# # endregion Plot parameterization

# endregion Parameterization Setup

# region Parameterization Solver Setup Evaluations

# region Wing Parameterization Evaluation for Parameterization Solver
# section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
max_y_coordinate = np.max(wing_ffd_block.coefficients[:, :, :, 1].value)
section_parametric_coordinates = wing_ffd_block.coefficients[0,:,0,1].value/(2*max_y_coordinate) + 0.5
sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_sweep_translation = wing_sweep_translation_b_spline.evaluate(section_parametric_coordinates)

midspan_location = 14.736/25.25
outer_midspan_location = 17./25.25
sectional_wing_linear_stretch = csdl.concatenate((-wing_linear_stretch, -outer_midspan_location*wing_linear_stretch, -midspan_location*wing_linear_stretch, csdl.Variable(value=0.), midspan_location*wing_linear_stretch, outer_midspan_location*wing_linear_stretch, wing_linear_stretch))
sectional_wing_midspan_stretch = csdl.Variable(value=np.zeros((7,)))
sectional_wing_midspan_stretch = sectional_wing_midspan_stretch.set(csdl.slice[2], -wing_midspan_stretch)
sectional_wing_midspan_stretch = sectional_wing_midspan_stretch.set(csdl.slice[4], wing_midspan_stretch)
sectional_wing_wingspan_stretch = sectional_wing_linear_stretch + sectional_wing_midspan_stretch
sectional_wing_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lsdo_geo.SectionalParameters(
    stretches={0: sectional_wing_chord_stretch},
    translations={1: sectional_wing_wingspan_stretch, 0 : sectional_sweep_translation},
    rotations={1: sectional_wing_twist}
)

wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
wing_ffd_block_coefficients += csdl.expand(wing_rigid_body_translation, wing_ffd_block_coefficients.shape, action='l->ijkl')
wing_coefficients = wing_ffd_block.evaluate_ffd(wing_ffd_block_coefficients, plot=False)
wing.set_coefficients(wing_coefficients)

# for function in wing.functions.values():
#     function.coefficients = function.coefficients + csdl.expand(wing_rigid_body_translation, function.coefficients.shape, action='k->ijk')

# plotting_elements = wing_ffd_block.plot(show=False)
# geometry.plot(additional_plotting_elements=plotting_elements, show=True)
# exit()

# endregion Wing Parameterization Evaluation for Parameterization Solver

# region Horizontal Stabilizer Parameterization Evaluation for Parameterization Solver
# section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
# sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
# # sectional_h_tail_twist = h_tail_twist_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = {
#     'sectional_h_tail_chord_stretch':sectional_h_tail_chord_stretch,
#     'sectional_h_tail_span_stretch':sectional_h_tail_span_stretch,
#     # 'sectional_h_tail_twist':sectional_h_tail_twist,
#     'sectional_h_tail_translation_x':sectional_h_tail_translation_x,
#     'sectional_h_tail_translation_z':sectional_h_tail_translation_z
#                         }
sectional_parameters = lsdo_geo.SectionalParameters(
    stretches={0: h_tail_sectional_chord_stretch},
    translations={1: h_tail_sectional_span_stretch, 0: h_tail_sectional_sweep_shear},
    rotations={1: h_tail_sectional_twist}
)

h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
h_tail_ffd_block_coefficients += csdl.expand(h_tail_rigid_body_translation, h_tail_ffd_block_coefficients.shape, action='l->ijkl')
h_tail_coefficients = h_tail_ffd_block.evaluate_ffd(h_tail_ffd_block_coefficients, plot=False)
h_tail.set_coefficients(coefficients=h_tail_coefficients)
# geometry.plot()
# endregion

# region Fuselage Parameterization Evaluation for Parameterization Solver
section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = {'sectional_fuselage_stretch':sectional_fuselage_stretch}
sectional_parameters = lsdo_geo.SectionalParameters(
    translations={0: sectional_fuselage_stretch}
)

fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
fuselage_and_nose_hub_coefficients = fuselage_ffd_block.evaluate_ffd(fuselage_ffd_block_coefficients, plot=False)
fuselage_coefficients = fuselage_and_nose_hub_coefficients[0]
nose_hub_coefficients = fuselage_and_nose_hub_coefficients[1]

fuselage.set_coefficients(coefficients=fuselage_coefficients)
nose_hub.set_coefficients(coefficients=nose_hub_coefficients)
# geometry.plot()

# endregion

# region Lift Rotors Parameterization Evaluation for Parameterization Solver
for i, component_set in enumerate(lift_rotor_related_components):
    rotor_ffd_block = lift_rotor_ffd_blocks[i]
    rotor_ffd_block_sectional_parameterization = lift_rotor_sectional_parameterizations[i]
    rotor_stretch_b_spline = lift_rotor_parameterization_b_splines[i]

    section_parametric_coordinates = np.linspace(0., 1., rotor_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
    sectional_stretch = rotor_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = lsdo_geo.SectionalParameters(
        stretches={0: sectional_stretch, 1:sectional_stretch}
    )
    rotor_ffd_block_coefficients = rotor_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)


    rigid_body_translation = csdl.Variable(shape=(3,), value=0., name=f'{component_set[0].name[:3]}_rotor_rigid_body_translation')
    boom = boom_components[i]
    for function in boom.functions.values():
        function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

    rotor_ffd_block_coefficients += csdl.expand(rigid_body_translation, rotor_ffd_block_coefficients.shape, action='l->ijkl')

    parameterization_solver.add_state(state=rigid_body_translation)

    rotor_coefficients = rotor_ffd_block.evaluate_ffd(rotor_ffd_block_coefficients, plot=False)
    for i, component in enumerate(component_set):
        component.set_coefficients(rotor_coefficients[i])
    # geometry.plot()

# endregion Lift Rotors Parameterization Evaluation for Parameterization Solver

# # region Lift Rotors rigid body translation
# lift_rotor_rigid_body_translations : list[csdl.Variable] = []
# for i, component_set in enumerate(lift_rotor_related_components):
#     # disk = component_set[0]
#     # blade_1 = component_set[1]
#     # blade_2 = component_set[2]
#     # hub = component_set[3]

#     boom = boom_components[i]

#     # Add rigid body translation
#     rigid_body_translation = csdl.Variable(shape=(3,), value=0., name=f'{component_set[0].name[:3]}_rotor_rigid_body_translation')
#     lift_rotor_rigid_body_translations.append(rigid_body_translation)

#     for component in component_set:
#         for function in component.functions.values():
#             function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

#     for function in boom.functions.values():
#         function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

#     parameterization_solver.add_state(state=rigid_body_translation)
#     # parameterization_solver.add_state(state=rigid_body_translation, initial_value=initial_guess_lift_rotor_rigid_body_translations[i])
# # endregion Lift Rotors rigid body translation

# region pusher rigid body translation
pusher_prop_rigid_body_translation = csdl.Variable(shape=(3,), value=0., name='pp_rotor_rigid_body_translation')
for component in pp_components:
    for function in component.functions.values():
        function.coefficients = function.coefficients + csdl.expand(pusher_prop_rigid_body_translation, function.coefficients.shape, action='k->ijk')

parameterization_solver.add_state(state=pusher_prop_rigid_body_translation)
# parameterization_solver.add_state(state=pusher_prop_rigid_body_translation, initial_value=initial_guess_pusher_prop_rigid_body_translation)
# endregion pusher rigid body translation

# region Vertical Stabilizer rigid body translation
v_tail_rigid_body_translation = csdl.Variable(shape=(3,), value=0., name='v_tail_rigid_body_translation')
for function in v_tail.functions.values():
    function.coefficients = function.coefficients + csdl.expand(v_tail_rigid_body_translation, function.coefficients.shape, action='k->ijk')

parameterization_solver.add_state(state=v_tail_rigid_body_translation)
# parameterization_solver.add_state(state=v_tail_rigid_body_translation, initial_value=initial_guess_v_tail_rigid_body_translation)
# endregion Vertical Stabilizer rigid body translation

# endregion Parameterization Solver Setup Evaluations

# region Define Design Parameters

# region wing design parameters
wing_span_computed = geometry.evaluate(wing_le_right)[1] - geometry.evaluate(wing_le_left)[1]
wing_root_chord_computed = geometry.evaluate(wing_te_center)[0] - geometry.evaluate(wing_le_center)[0]
wing_tip_chord_left_computed = geometry.evaluate(wing_te_left)[0] - geometry.evaluate(wing_le_left)[0]
wing_tip_chord_right_computed = geometry.evaluate(wing_te_right)[0] - geometry.evaluate(wing_le_right)[0]
wing_leading_edge_line = geometry.evaluate(wing_leading_edge_line_parametric)
wing_trailing_edge_line = geometry.evaluate(wing_trailing_edge_line_parametric)
wing_chord_surface = csdl.linear_combination(wing_leading_edge_line, wing_trailing_edge_line, wing_num_chordwise_vlm)
u_vectors = wing_chord_surface[1:,:] - wing_chord_surface[:-1,:]
v_vectors = wing_chord_surface[:,1:] - wing_chord_surface[:,:-1]
wing_panel_areas = 1/2*csdl.cross(u_vectors[:,:-1], v_vectors[:-1,:], axis=2) \
                 + 1/2*csdl.cross(u_vectors[:,1:], v_vectors[1:,:], axis=2)  # (num_chordwise_vlm-1, num_spanwise_vlm-1, 3)
wing_panel_areas = csdl.norm(wing_panel_areas, axes=(2,))  # (num_chordwise_vlm-1, num_spanwise_vlm-1)
wing_area_computed = csdl.sum(wing_panel_areas)
wing_aspect_ratio_computed = wing_span_computed**2/wing_area_computed
wing_left_taper_ratio_computed = wing_tip_chord_left_computed/wing_root_chord_computed
wing_right_taper_ratio_computed = wing_tip_chord_right_computed/wing_root_chord_computed
wing_taper_ratio_computed = 1/2*(wing_left_taper_ratio_computed + wing_right_taper_ratio_computed)
wing_qc_line_left = geometry.evaluate(wing_qc_left) - geometry.evaluate(wing_qc_center)
wing_sweep_left = csdl.arctan(wing_qc_line_left[0]/(-wing_qc_line_left[1]))
wing_qc_line_right = geometry.evaluate(wing_qc_right) - geometry.evaluate(wing_qc_center)
wing_sweep_right = csdl.arctan(wing_qc_line_right[0]/wing_qc_line_right[1])
wing_sweep_computed = 1/2*(wing_sweep_left + wing_sweep_right)

parameterization_design_parameters.add_variable(computed_value=wing_area_computed, desired_value=wing_area)
InversionTransform = csdl.transforms.EqualityInversion()
inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=wing_aspect_ratio_computed, rhs=wing_aspect_ratio, debug = False, aux_info=True)
parameterization_design_parameters.add_variable(computed_value=inverted_variable, desired_value=inverted_constant)
# parameterization_design_parameters.add_variable(computed_value=wing_aspect_ratio_computed, desired_value=wing_aspect_ratio)
InversionTransform = csdl.transforms.EqualityInversion()
inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=wing_taper_ratio_computed, rhs=wing_taper_ratio, debug = False, aux_info=True)
parameterization_design_parameters.add_variable(computed_value=inverted_variable, desired_value=inverted_constant)
# parameterization_design_parameters.add_variable(computed_value=wing_taper_ratio_computed, desired_value=wing_taper_ratio)
InversionTransform = csdl.transforms.EqualityInversion()
inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=wing_sweep_computed, rhs=wing_sweep, debug = False, aux_info=True)
parameterization_design_parameters.add_variable(computed_value=inverted_variable, desired_value=inverted_constant)
# parameterization_design_parameters.add_variable(computed_value=wing_sweep_computed, desired_value=wing_sweep)

# InversionTransform = csdl.transforms.EqualityInversion()
# inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=wing_left_taper_ratio_computed, rhs=csdl.arctan(wing_taper_ratio), debug = False, aux_info=True)
# print(f'Inverted {wing_left_taper_ratio_computed.name} = {wing_taper_ratio.name} : {([op.name for op in ops])}')
# parameterization_design_parameters.add_variable(computed_value=inverted_variable, desired_value=inverted_constant)


# parameterization_design_parameters.add_variable(computed_value=wing_span_computed, desired_value=wing_span)
# parameterization_design_parameters.add_variable(computed_value=wing_root_chord_computed, desired_value=wing_root_chord)
# parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_left_computed, desired_value=wing_tip_chord)
# parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_right_computed, desired_value=wing_tip_chord)
# endregion wing design parameters

# region h_tail design parameterization inputs
h_tail_span_computed = csdl.norm(geometry.evaluate(tail_le_right) - geometry.evaluate(tail_le_left))
h_tail_root_chord_computed = csdl.norm(geometry.evaluate(tail_te_center) - geometry.evaluate(tail_le_center))
h_tail_tip_chord_left_computed = csdl.norm(geometry.evaluate(tail_te_left) - geometry.evaluate(tail_le_left))
h_tail_tip_chord_right_computed = csdl.norm(geometry.evaluate(tail_te_right) - geometry.evaluate(tail_le_right))
h_tail_leading_edge_line = geometry.evaluate(h_tail_leading_edge_line_parametric)
h_tail_trailing_edge_line = geometry.evaluate(h_tail_trailing_edge_line_parametric)
h_tail_chord_surface = csdl.linear_combination(h_tail_leading_edge_line, h_tail_trailing_edge_line, h_tail_num_chordwise_vlm)
u_vectors = h_tail_chord_surface[1:,:] - h_tail_chord_surface[:-1,:]
v_vectors = h_tail_chord_surface[:,1:] - h_tail_chord_surface[:,:-1]
h_tail_panel_areas = 1/2*csdl.cross(u_vectors[:,:-1], v_vectors[:-1,:], axis=2) \
                 + 1/2*csdl.cross(u_vectors[:,1:], v_vectors[1:,:], axis=2)  # (num_chordwise_vlm-1, num_spanwise_vlm-1, 3)
h_tail_panel_areas = csdl.norm(h_tail_panel_areas, axes=(2,))  # (num_chordwise_vlm-1, num_spanwise_vlm-1)
h_tail_area_computed = csdl.sum(h_tail_panel_areas)
h_tail_aspect_ratio_computed = h_tail_span_computed**2/h_tail_area_computed
h_tail_left_taper_ratio_computed = h_tail_tip_chord_left_computed/h_tail_root_chord_computed
h_tail_right_taper_ratio_computed = h_tail_tip_chord_right_computed/h_tail_root_chord_computed
h_tail_taper_ratio_computed = 1/2*(h_tail_left_taper_ratio_computed + h_tail_right_taper_ratio_computed)
h_tail_qc_line_left = geometry.evaluate(tail_qc_left) - geometry.evaluate(tail_qc_center)
h_tail_sweep_left = csdl.arctan(h_tail_qc_line_left[0]/(-h_tail_qc_line_left[1]))
h_tail_qc_line_right = geometry.evaluate(tail_qc_right) - geometry.evaluate(tail_qc_center)
h_tail_sweep_right = csdl.arctan(h_tail_qc_line_right[0]/h_tail_qc_line_right[1])
h_tail_sweep_computed = 1/2*(h_tail_sweep_left + h_tail_sweep_right)

parameterization_design_parameters.add_variable(computed_value=h_tail_area_computed, desired_value=h_tail_area)
InversionTransform = csdl.transforms.EqualityInversion()
inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=h_tail_aspect_ratio_computed, rhs=h_tail_aspect_ratio, debug = False, aux_info=True)
parameterization_design_parameters.add_variable(computed_value=inverted_variable, desired_value=inverted_constant)
# parameterization_design_parameters.add_variable(computed_value=h_tail_aspect_ratio_computed, desired_value=h_tail_aspect_ratio)
InversionTransform = csdl.transforms.EqualityInversion()
inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=h_tail_taper_ratio_computed, rhs=h_tail_taper_ratio, debug = False, aux_info=True)
parameterization_design_parameters.add_variable(computed_value=inverted_variable, desired_value=inverted_constant)
# parameterization_design_parameters.add_variable(computed_value=h_tail_taper_ratio_computed, desired_value=h_tail_taper_ratio)
InversionTransform = csdl.transforms.EqualityInversion()
inverted_variable, inverted_constant, ops = InversionTransform.apply(lhs=h_tail_sweep_computed, rhs=h_tail_sweep, debug = False, aux_info=True)
parameterization_design_parameters.add_variable(computed_value=inverted_variable, desired_value=inverted_constant)
# parameterization_design_parameters.add_variable(computed_value=h_tail_sweep_computed, desired_value=h_tail_sweep)

# parameterization_design_parameters.add_variable(computed_value=h_tail_area_computed, desired_value=h_tail_area)
# parameterization_design_parameters.add_variable(computed_value=h_tail_aspect_ratio_computed, desired_value=h_tail_aspect_ratio)
# parameterization_design_parameters.add_variable(computed_value=h_tail_left_taper_ratio_computed, desired_value=h_tail_taper_ratio)
# parameterization_design_parameters.add_variable(computed_value=h_tail_right_taper_ratio_computed, desired_value=h_tail_taper_ratio)
# parameterization_design_parameters.add_variable(computed_value=h_tail_sweep_computed, desired_value=h_tail_sweep)

# parameterization_design_parameters.add_variable(computed_value=h_tail_span_computed, desired_value=h_tail_span)
# parameterization_design_parameters.add_variable(computed_value=h_tail_root_chord_computed, desired_value=h_tail_root_chord)
# parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_left_computed, desired_value=h_tail_tip_chord)
# parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_right_computed, desired_value=h_tail_tip_chord)
# endregion h_tail design parameterization inputs

# region tail moment arm variables
tail_moment_arm_computed = csdl.norm(geometry.evaluate(tail_qc_center) - geometry.evaluate(wing_qc_center))
tail_moment_arm_computed.add_name('tail_moment_arm_computed')
parameterization_design_parameters.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm)

wing_fuselage_connection = geometry.evaluate(wing_te_center) - geometry.evaluate(fuselage_wing_te_center)
wing_fuselage_connection.add_name('wing_fuselage_connection_computed')
h_tail_fuselage_connection = geometry.evaluate(tail_te_center) - geometry.evaluate(fuselage_tail_te_center)
h_tail_fuselage_connection.add_name('h_tail_fuselage_connection_computed')
parameterization_design_parameters.add_variable(computed_value=wing_fuselage_connection, desired_value=wing_fuselage_connection.value)
parameterization_design_parameters.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)

# endregion tail moment arm variables

# region v-tail connection
vtail_fuselage_connection_point = geometry.evaluate(v_tail.project(np.array([30.543, 0., 8.231])))
vtail_fuselage_connection = geometry.evaluate(fuselage_rear_points_parametric) - vtail_fuselage_connection_point
vtail_fuselage_connection.add_name('vtail_fuselage_connection_computed')
parameterization_design_parameters.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)

# endregion v-tail connection

# region lift + pusher rotor parameterization inputs
pusher_fuselage_connection = geometry.evaluate(fuselage_rear_points_parametric) - geometry.evaluate(fuselage_rear_point_on_pusher_disk_parametric)
pusher_fuselage_connection.add_name('pusher_fuselage_connection_computed')
parameterization_design_parameters.add_variable(computed_value=pusher_fuselage_connection, desired_value=pusher_fuselage_connection.value)

dv_radius_list = [rlo_radius, rli_radius, rri_radius, rro_radius, flo_radius, fli_radius, fri_radius, fro_radius]

boom_points = [boom_rlo, boom_rli, boom_rri, boom_rro, boom_flo, boom_fli, boom_fri, boom_fro]
boom_points_on_wing = [wing_boom_rlo, wing_boom_rli, wing_boom_rri, wing_boom_rro, wing_boom_flo, wing_boom_fli, wing_boom_fri, wing_boom_fro]
rotor_prefixes = ['rlo', 'rli', 'rri', 'rro', 'flo', 'fli', 'fri', 'fro']

for i in range(len(boom_points)):
    boom_connection = geometry.evaluate(boom_points[i]) - geometry.evaluate(boom_points_on_wing[i])
    boom_connection.add_name(f'boom_connection_{rotor_prefixes[i]}_computed')

    parameterization_design_parameters.add_variable(computed_value=boom_connection, desired_value=boom_connection.value)
    # parameterization_design_parameters.add_variable(computed_value=boom_connection, desired_value=np.zeros((3,)))
    
    component_rotor_edges = rotor_edges[i]
    radius_computed = csdl.norm(geometry.evaluate(component_rotor_edges[0]) - geometry.evaluate(component_rotor_edges[1]))/2
    radius_computed.add_name(f'rotor_radius_{rotor_prefixes[i]}_computed')
    if i == 1 or i == 2 or i == 5 or i == 6:    # For inner rotors, make radius variable weak to make sure rotors don't collide
        parameterization_design_parameters.add_variable(computed_value=radius_computed, desired_value=dv_radius_list[i], penalty_value=1.e2)
        # parameterization_design_parameters.add_variable(computed_value=radius_computed, desired_value=dv_radius_list[i], penalty_value=None)
    else:
        parameterization_design_parameters.add_variable(computed_value=radius_computed, desired_value=dv_radius_list[i], penalty_value=1.e2)
        # parameterization_design_parameters.add_variable(computed_value=radius_computed, desired_value=dv_radius_list[i], penalty_value=None)

# endregion lift + pusher rotor parameterization inputs

# region rotor overlap inequality constraints
def sigmoid_activation(x:csdl.Variable, rho:float=50) -> csdl.Variable:
    return 1/(1 + csdl.exp(-rho*x))
    # return 1/(1 + csdl.exp(x))
    # return 1/(1 + csdl.exp(-csdl.absolute(x)))
    return csdl.exp(rho*x)/(1 + csdl.exp(rho*x))
    # return csdl.sin(rho*x)    # This is wrong but I'm debugging

def mellowmax_with_zero(x:csdl.Variable, rho:float=50) -> csdl.Variable:
    return (1/rho)*csdl.log((1 + csdl.exp(rho*x))/2)

def softplus(x:csdl.Variable, rho:float=50) -> csdl.Variable:
    return (1/rho)*csdl.log(1 + csdl.exp(rho*x))


# Inner-outer rotor collision constraints
right_front_inner_disk_outer_point = geometry.evaluate(fri_disk_y1_para)
left_front_inner_disk_outer_point = geometry.evaluate(fli_disk_y2_para)
right_front_outer_disk_inner_point = geometry.evaluate(fro_disk_y2_para)
left_front_outer_disk_inner_point = geometry.evaluate(flo_disk_y1_para)
right_rear_inner_disk_outer_point = geometry.evaluate(rri_disk_y1_para)
left_rear_inner_disk_outer_point = geometry.evaluate(rli_disk_y2_para)
right_rear_outer_disk_inner_point = geometry.evaluate(rro_disk_y2_para)
left_rear_outer_disk_inner_point = geometry.evaluate(rlo_disk_y1_para)

front_left_disk_distance = left_front_inner_disk_outer_point[1] - left_front_outer_disk_inner_point[1]
front_right_disk_distance = right_front_outer_disk_inner_point[1] - right_front_inner_disk_outer_point[1]
rear_left_disk_distance = left_rear_inner_disk_outer_point[1] - left_rear_outer_disk_inner_point[1]
rear_right_disk_distance = right_rear_outer_disk_inner_point[1] - right_rear_inner_disk_outer_point[1]

# quadratic_penalty_factor_inner_outer = 1.e2
# quadratic_penalty_factor = 0.
# quadratic_activation_factor = 1.e1
# quadratic_activation_factor = 20.
quadratic_activation_factor = 1.
# linear_penalty_factor = 1.e2
linear_penalty_factor = linear_penalty_factor_inner_outer
linear_activation_factor = 1.
# linear_activation_factor = 10.
# linear_activation_factor = 1.
inner_outer_min_distance = 0.4


constraint = -(front_left_disk_distance - inner_outer_min_distance)
constraint.add_name('front_left_disk_distance_constraint')
# parameterization_solver.add_inequality_constraint(constraint=constraint, linear_penalty_factor=None, linear_activation_factor=linear_activation_factor, 
#                                                     quadratic_penalty_factor=quadratic_penalty_factor_inner_outer, quadratic_activation_factor=quadratic_activation_factor)
constraint = -(front_right_disk_distance - inner_outer_min_distance)
constraint.add_name('front_right_disk_distance_constraint')
# parameterization_solver.add_inequality_constraint(constraint=constraint, linear_penalty_factor=None, linear_activation_factor=linear_activation_factor, 
#                                                     quadratic_penalty_factor=quadratic_penalty_factor_inner_outer, quadratic_activation_factor=quadratic_activation_factor)
constraint = -(rear_left_disk_distance - inner_outer_min_distance)
constraint.add_name('rear_left_disk_distance_constraint')
# parameterization_solver.add_inequality_constraint(constraint=constraint, linear_penalty_factor=None, linear_activation_factor=linear_activation_factor, 
#                                                     quadratic_penalty_factor=quadratic_penalty_factor_inner_outer, quadratic_activation_factor=quadratic_activation_factor)
constraint = -(rear_right_disk_distance - inner_outer_min_distance)
constraint.add_name('rear_right_disk_distance_constraint')
# parameterization_solver.add_inequality_constraint(constraint=constraint, linear_penalty_factor=None, linear_activation_factor=linear_activation_factor,
#                                                     quadratic_penalty_factor=quadratic_penalty_factor_inner_outer, quadratic_activation_factor=quadratic_activation_factor)

# Inner-inner rotor collision constraint on rear and inner-fuselage collision constraints on front
# quadratic_penalty_factor_inner_inner = 1.e3
# quadratic_penalty_factor = 0.
# quadratic_activation_factor = 10.
# quadratic_activation_factor = 20.
quadratic_activation_factor = 1.
# linear_penalty_factor = 1.e3
linear_penalty_factor = linear_penalty_factor_inner_inner
# linear_activation_factor = 1.
# linear_activation_factor = 10.
linear_activation_factor = 1.e2
inner_inner_min_distance = 1.
right_front_inner_disk_inner_point = geometry.evaluate(fri_disk_y2_para)
left_front_inner_disk_inner_point = geometry.evaluate(fli_disk_y1_para)
right_rear_inner_disk_inner_point = geometry.evaluate(rri_disk_y2_para)
left_rear_inner_disk_inner_point = geometry.evaluate(rli_disk_y1_para)
# front_inner_disk_distance = right_front_inner_disk_inner_point[1] - left_front_inner_disk_inner_point[1]
rear_inner_disk_distance = right_rear_inner_disk_inner_point[1] - left_rear_inner_disk_inner_point[1]
# constraint_penalty = compute_constraint_penalty(rear_inner_disk_distance, min_distance, quadratic_penalty_factor, quadratic_activation_factor, linear_penalty_factor, linear_activation_factor)
# parameterization_solver.add_constraint(constraint=rear_inner_disk_distance, desired_value=min_distance, penalty=constraint_penalty)

front_right_disk_fuselage_distance = right_front_inner_disk_inner_point[1] - geometry.evaluate(fuselage_fri_collision_point_parametric)[1]
front_left_disk_fuselage_distance = geometry.evaluate(fuselage_fli_collision_point_parametric)[1] - left_front_inner_disk_inner_point[1]

constraint = -(front_right_disk_fuselage_distance - inner_inner_min_distance)
constraint.add_name('front_right_disk_fuselage_distance_constraint')
# parameterization_solver.add_inequality_constraint(constraint=constraint,
#                                                     linear_penalty_factor=None, linear_activation_factor=linear_activation_factor,
#                                                     quadratic_penalty_factor=quadratic_penalty_factor_inner_inner, quadratic_activation_factor=quadratic_activation_factor)
constraint = -(front_left_disk_fuselage_distance - inner_inner_min_distance)
constraint.add_name('front_left_disk_fuselage_distance_constraint')
# parameterization_solver.add_inequality_constraint(constraint=constraint,
#                                                     linear_penalty_factor=None, linear_activation_factor=linear_activation_factor,
#                                                     quadratic_penalty_factor=quadratic_penalty_factor_inner_inner, quadratic_activation_factor=quadratic_activation_factor)

# endregion rotor overlap inequality constraints

# endregion Define Design Parameters

# geometry.plot()
# print('============================')
# list_of_constraint_arrays = parameterization_design_parameters.computed_values
# num_constraints = np.sum([list_of_constraint_arrays[i].shape[0] for i in range(len(list_of_constraint_arrays))])
# print('Number of constraints: ', num_constraints)

# list_of_states_arrays = parameterization_solver.parameters
# num_states = np.sum([list_of_states_arrays[i].shape[0] for i in range(len(list_of_states_arrays))])
# print('Number of states: ', num_states)

parameterization_solver.evaluate(parameterization_design_parameters)

inner_outer_front_left_rotor_constraint_value = -(front_left_disk_distance - inner_outer_min_distance)
inner_outer_front_right_rotor_constraint_value = -(front_right_disk_distance - inner_outer_min_distance)
inner_outer_rear_left_rotor_constraint_value = -(rear_left_disk_distance - inner_outer_min_distance)
inner_outer_rear_right_rotor_constraint_value = -(rear_right_disk_distance - inner_outer_min_distance)
# softplus_constraint_inner_outer_front = softplus(inner_outer_front_rotor_constraint_value, quadratic_activation_factor)
# softplus_constraint_inner_outer_rear = softplus(inner_outer_rear_rotor_constraint_value, quadratic_activation_factor)
# new_linear_penalty_factor_inner_outer = linear_penalty_factor_inner_outer + 2*quadratic_penalty_factor_inner_outer*softplus_constraint_inner_outer
# loop_builder.finalize_feedback(linear_penalty_factor_inner_outer, new_linear_penalty_factor_inner_outer)

inner_inner_left_rotor_constraint_value = -(front_left_disk_fuselage_distance - inner_inner_min_distance)
inner_inner_right_rotor_constraint_value = -(front_right_disk_fuselage_distance - inner_inner_min_distance)
# softplus_constraint_inner_inner = softplus(inner_inner_rotor_constraint_value, quadratic_activation_factor)
# new_linear_penalty_factor_inner_inner = linear_penalty_factor_inner_inner + 2*quadratic_penalty_factor_inner_inner*softplus_constraint_inner_inner
# loop_builder.finalize_feedback(linear_penalty_factor_inner_inner, new_linear_penalty_factor_inner_inner)

# new_quadratic_penalty_factor_inner_outer = quadratic_penalty_factor_inner_outer * 1.2
# loop_builder.finalize_feedback(quadratic_penalty_factor_inner_outer, new_quadratic_penalty_factor_inner_outer)
# new_quadratic_penalty_factor_inner_inner = quadratic_penalty_factor_inner_inner * 1.2
# loop_builder.finalize_feedback(quadratic_penalty_factor_inner_inner, new_quadratic_penalty_factor_inner_inner)

# loop_builder.finalize_feedback(initial_guess_wing_chord_stretch_coefficients, wing_chord_stretch_coefficients)
# loop_builder.finalize_feedback(initial_guess_wing_linear_stretch, wing_linear_stretch)
# loop_builder.finalize_feedback(initial_guess_wing_midspan_stretch, wing_midspan_stretch)
# loop_builder.finalize_feedback(initial_guess_wing_rigid_body_translation, wing_rigid_body_translation)

# loop_builder.finalize_feedback(initial_guess_h_tail_chord_stretch_coefficients, h_tail_chord_stretch_coefficients)
# loop_builder.finalize_feedback(initial_guess_h_tail_span_stretch_coefficients, h_tail_span_stretch_coefficients)
# loop_builder.finalize_feedback(initial_guess_h_tail_translation_x_coefficients, h_tail_translation_x_coefficients)
# loop_builder.finalize_feedback(initial_guess_h_tail_translation_z_coefficients, h_tail_translation_z_coefficients)

# loop_builder.finalize_feedback(initial_guess_fuselage_stretch_coefficients, fuselage_stretch_coefficients)

# for i in range(len(lift_rotor_related_components)):
#     loop_builder.finalize_feedback(initial_guess_lift_rotor_stretch_coefficients[i], lift_rotor_stretch_coefficients[i])
#     loop_builder.finalize_feedback(initial_guess_lift_rotor_rigid_body_translations[i], lift_rotor_rigid_body_translations[i])

# loop_builder.finalize_feedback(initial_guess_pusher_prop_rigid_body_translation, pusher_prop_rigid_body_translation)

# loop_builder.finalize_feedback(initial_guess_v_tail_rigid_body_translation, v_tail_rigid_body_translation)

# geometry.plot()

# stacked_linear_penalty_factor_inner_outer = loop_builder.add_stack(linear_penalty_factor_inner_outer)
# stacked_linear_penalty_factor_inner_inner = loop_builder.add_stack(linear_penalty_factor_inner_inner)
# stacked_quadratic_penalty_factor_inner_outer = loop_builder.add_stack(quadratic_penalty_factor_inner_outer)
# stacked_quadratic_penalty_factor_inner_inner = loop_builder.add_stack(quadratic_penalty_factor_inner_inner)
# stacked_constraint_values_inner_outer_front = loop_builder.add_stack(inner_outer_front_rotor_constraint_value)
# stacked_constraint_values_inner_outer_rear = loop_builder.add_stack(inner_outer_rear_rotor_constraint_value)
# stacked_constraint_values_inner_inner = loop_builder.add_stack(inner_inner_rotor_constraint_value)

# recorder.inline = False
# loop_builder.finalize(add_all_outputs=True)
# geometry.plot()

mellowmax_constraint_inner_outer_front_left = mellowmax_with_zero(inner_outer_front_left_rotor_constraint_value, rho=10.)
mellowmax_constraint_inner_outer_front_right = mellowmax_with_zero(inner_outer_front_right_rotor_constraint_value, rho=10.)
mellowmax_constraint_inner_outer_rear_left = mellowmax_with_zero(inner_outer_rear_left_rotor_constraint_value, rho=10.)
mellowmax_constraint_inner_outer_rear_right = mellowmax_with_zero(inner_outer_rear_right_rotor_constraint_value, rho=10.)
mellowmax_constraint_inner_inner_left = mellowmax_with_zero(inner_inner_left_rotor_constraint_value, rho=10.)
mellowmax_constraint_inner_inner_right = mellowmax_with_zero(inner_inner_right_rotor_constraint_value, rho=10.)

# print('linear_penalty_factor_inner_outer: ', linear_penalty_factor_inner_outer.value)
# print('new_linear_penalty_factor_inner_outer: ', new_linear_penalty_factor_inner_outer.value)
print('constraint value inner_outer front left: ', inner_outer_front_left_rotor_constraint_value.value)
print('constraint value inner_outer front right: ', inner_outer_front_right_rotor_constraint_value.value)
print('constraint value inner_outer rear left: ', inner_outer_rear_left_rotor_constraint_value.value)
print('constraint value inner_outer rear right: ', inner_outer_rear_right_rotor_constraint_value.value)
# print('mellowmax value inner_outer: ', mellowmax_constraint_inner_outer_front.value)
print('============================')
# print('linear_penalty_factor_inner_inner: ', linear_penalty_factor_inner_inner.value)
# print('new_linear_penalty_factor_inner_inner: ', new_linear_penalty_factor_inner_inner.value)
print('constraint value inner_inner left: ', inner_inner_left_rotor_constraint_value.value)
print('constraint value inner_inner right: ', inner_inner_right_rotor_constraint_value.value)
# print('mellowmax value inner_inner: ', mellowmax_constraint_inner_inner.value)
# plotting_element = wing_ffd_block.plot(show=False)
# geometry.plot(additional_plotting_elements=plotting_element, show=True)
# python_sim = csdl.experimental.PySimulator(
#     recorder=recorder
# )
# python_sim.run()
# geometry.plot()
# exit()
# endregion

# # region Visualize Geometry
# plotting_elements = wing_ffd_block.plot(plot_embedded_points=False, show=False, additional_plotting_elements=initial_geometry_plot)
# plotting_elements = wing_ffd_block_sectional_parameterization.plot(opacity=0.3, color='#6E963B', additional_plotting_elements=plotting_elements, show=False)
# plotting_elements = h_tail_ffd_block.plot(plot_embedded_points=False, show=False, additional_plotting_elements=plotting_elements)
# plotting_elements = fuselage_ffd_block.plot(plot_embedded_points=False, show=False, additional_plotting_elements=plotting_elements)
# for rotor_ffd_block in lift_rotor_ffd_blocks:
#     plotting_elements = rotor_ffd_block.plot(plot_embedded_points=False, show=False, additional_plotting_elements=plotting_elements)
# geometry.plot(opacity=0.9, additional_plotting_elements=plotting_elements, show=True, camera=camera)
# # exit()
# # endregion Visualize Geometry

# # region Mesh Evaluation
# upper_surface_wireframe = geometry.evaluate(wing_upper_surface_wireframe_parametric)
# lower_surface_wireframe = geometry.evaluate(wing_lower_surface_wireframe_parametric)
# wing_vlm_mesh = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((wing_num_chordwise_vlm, wing_num_spanwise_vlm, 3))

# upper_surface_wireframe = geometry.evaluate(h_tail_upper_surface_wireframe_parametric)
# lower_surface_wireframe = geometry.evaluate(h_tail_lower_surface_wireframe_parametric)
# h_tail_vlm_mesh = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((h_tail_num_chordwise_vlm, h_tail_num_spanwise_vlm, 3))

# beam_tops = wing.evaluate(beam_top_parametric)
# beam_bottoms = wing.evaluate(beam_bottom_parametric)
# wing_beam_mesh = csdl.linear_combination(beam_tops, beam_bottoms, 1).reshape((num_beam_nodes, 3))
# beam_heights = csdl.norm(beam_tops - beam_bottoms, axes=(1,))
# # endregion Mesh Evaluation


# jax_inputs = [wing_span, wing_root_chord, wing_tip_chord, h_tail_span, h_tail_root_chord, h_tail_tip_chord,
#               tail_moment_arm, front_outer_radius, front_inner_radius, rear_inner_radius, rear_outer_radius]
jax_inputs = [wing_area, wing_aspect_ratio, wing_taper_ratio, wing_sweep, h_tail_area, h_tail_aspect_ratio, h_tail_taper_ratio,
              h_tail_sweep, tail_moment_arm, front_outer_radius, front_inner_radius, rear_inner_radius, rear_outer_radius]

# jax outputs is a list containing all the geometry coefficients (geometry.functions[:].coefficients)
# jax_outputs = [geometry_function.coefficients for geometry_function in geometry.functions.values()] + \
#                [linear_penalty_factor_inner_outer, linear_penalty_factor_inner_inner,
#                ks_max_constraint_inner_outer, ks_max_constraint_inner_inner,
#                inner_outer_rotor_constraint_value, inner_inner_rotor_constraint_value,
#                stacked_linear_penalty_factor_inner_outer, stacked_linear_penalty_factor_inner_inner,
#                stacked_quadratic_penalty_factor_inner_outer, stacked_quadratic_penalty_factor_inner_inner,
#                stacked_constraint_values_inner_outer, stacked_constraint_values_inner_inner]
# jax_outputs = [geometry_function.coefficients for geometry_function in geometry.functions.values()] + \
#                [stacked_linear_penalty_factor_inner_outer, stacked_linear_penalty_factor_inner_inner,
#                stacked_quadratic_penalty_factor_inner_outer, stacked_quadratic_penalty_factor_inner_inner,
#                stacked_constraint_values_inner_outer_front, stacked_constraint_values_inner_outer_rear,
#                 stacked_constraint_values_inner_inner]
jax_outputs = [geometry_function.coefficients for geometry_function in geometry.functions.values()] + \
               [inner_outer_front_left_rotor_constraint_value, inner_outer_rear_left_rotor_constraint_value,
                inner_inner_left_rotor_constraint_value, inner_outer_front_right_rotor_constraint_value,
                inner_outer_rear_right_rotor_constraint_value, inner_inner_right_rotor_constraint_value,
                mellowmax_constraint_inner_outer_front_left, mellowmax_constraint_inner_outer_rear_left,
                mellowmax_constraint_inner_inner_left, mellowmax_constraint_inner_outer_front_right,
                mellowmax_constraint_inner_outer_rear_right, mellowmax_constraint_inner_inner_right]


jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=jax_inputs,
    additional_outputs=jax_outputs,
    gpu=False
)

# jax_sim.run()

# region Create Video Sweeping Wing Span
import pyvista as pv
camera_pos = [(-50, -50, 40), (15, 0, 5), (0, 0, 1)]

file_path = 'examples/showcase_examples/lift_plus_cruise/videos/'
file_name = 'wingspan_sweep.mp4'
plotter = pv.Plotter(off_screen=True, window_size=[1920, 1200])
plotter.open_movie(file_path + file_name, framerate=11)
wing_span_values_1 = np.linspace(wing_span_computed.value, wing_span_computed.value*2/3, 11)
wing_span_values_2 = np.linspace(wing_span_computed.value*2/3, wing_span_computed.value*4/3, 21)[1:]
wing_span_values_3 = np.linspace(wing_span_computed.value*4/3, wing_span_computed.value, 11)[1:]
wing_span_values = np.concatenate((wing_span_values_1, wing_span_values_2, wing_span_values_3))
initial_wing_span = wing_span_computed.value
initial_wing_area = wing_area.value
for wing_span_value in wing_span_values:
    wing_area_value = initial_wing_area * (wing_span_value / initial_wing_span)
    wing_aspect_ratio_value = wing_span_value**2/wing_area_value
    # jax_sim[wing_span] = wing_span_value
    jax_sim[wing_area] = wing_area_value
    jax_sim[wing_aspect_ratio] = wing_aspect_ratio_value
    jax_sim.run()
    plotter.clear()
    frame = geometry.plot(show=False)
    for element in frame:
        if isinstance(element, dict) and 'mesh' in element:
            plotter.add_mesh(element['mesh'], **element.get('kwargs', {}))
        elif isinstance(element, pv.DataSet):
            plotter.add_mesh(element)
    plotter.camera_position = camera_pos
    plotter.write_frame()

    print('wing span: ', wing_span_value)
    print('front inner outer constraint values: ', inner_outer_front_left_rotor_constraint_value.value)
    print('rear inner outer constraint values: ', inner_outer_rear_left_rotor_constraint_value.value)
    print('inner inner constraint values: ', inner_inner_left_rotor_constraint_value.value)
    # print('inner outer linear penalty factors: ', stacked_linear_penalty_factor_inner_outer.value)
    # print('inner inner linear penalty factors: ', stacked_linear_penalty_factor_inner_inner.value)
    # print('inner outer quadratic penalty factors: ', stacked_quadratic_penalty_factor_inner_outer.value)
    # print('inner inner quadratic penalty factors: ', stacked_quadratic_penalty_factor_inner_inner.value)
    print('============================')

plotter.close()
# exit()
# endregion Create Video Sweeping Wing Span

# region Create Data For Timing Study
# Latin Hypercube Sampling for Design Variables
import numpy as np
from scipy.stats import qmc
import pickle
import matplotlib.pyplot as plt

# Define design variable bounds (lower and upper bounds for each variable)
design_variable_bounds = {
    'wing_area': [wing_area.value.item()*0.5, wing_area.value.item()*2],              # Wing area bounds
    'wing_aspect_ratio': [wing_aspect_ratio.value.item()*0.5, wing_aspect_ratio.value.item()*2],         # Wing aspect ratio bounds
    'wing_taper_ratio': [wing_taper_ratio.value.item()*0.5, wing_taper_ratio.value.item()*2],          # Wing taper ratio bounds
    'wing_sweep': [wing_sweep.value.item()*0.5, wing_sweep.value.item()*2],                  # Wing sweep bounds
    'h_tail_area': [h_tail_area.value.item()*0.5, h_tail_area.value.item()*2],             # Horizontal tail area bounds
    'h_tail_aspect_ratio': [h_tail_aspect_ratio.value.item()*0.5, h_tail_aspect_ratio.value.item()*2],       # H-tail aspect ratio bounds
    'h_tail_taper_ratio': [h_tail_taper_ratio.value.item()*0.5, h_tail_taper_ratio.value.item()*2],        # H-tail taper ratio bounds
    'h_tail_sweep': [h_tail_sweep.value.item()*0.5, h_tail_sweep.value.item()*2],               # H-tail sweep bounds
    'tail_moment_arm': [tail_moment_arm.value.item()*0.5, tail_moment_arm.value.item()*2],         # Tail moment arm bounds
    'front_outer_radius': [front_outer_radius.value.item()*0.5, front_outer_radius.value.item()*1.25],              # rotor radii bounds
    'front_inner_radius': [front_inner_radius.value.item()*0.5, front_inner_radius.value.item()*1.25],
    'rear_inner_radius': [rear_inner_radius.value.item()*0.5, rear_inner_radius.value.item()*1.25],
    'rear_outer_radius': [rear_outer_radius.value.item()*0.5, rear_outer_radius.value.item()*1.25],
}

# Number of samples to generate
# n_samples = 300
n_samples = 100

# Extract bounds in the same order as jax_inputs
variable_names = ['wing_area', 'wing_aspect_ratio', 'wing_taper_ratio', 'wing_sweep', 'h_tail_area', 
                  'h_tail_aspect_ratio', 'h_tail_taper_ratio', 'h_tail_sweep', 'tail_moment_arm',
                  'front_outer_radius', 'front_inner_radius', 'rear_inner_radius', 'rear_outer_radius']

lower_bounds = np.array([design_variable_bounds[name][0] for name in variable_names])
upper_bounds = np.array([design_variable_bounds[name][1] for name in variable_names])

# Generate Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=len(variable_names), seed=42)
unit_samples = sampler.random(n=n_samples)

# Scale samples to actual bounds
lhs_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

print(f"Generated {n_samples} Latin Hypercube samples")
print(f"Sample shape: {lhs_samples.shape}")
print(f"Variable order: {variable_names}")

# Compute baseline values (current values of jax_inputs)
baseline_values = np.array([jax_input.value.item() for jax_input in jax_inputs])

# Compute input norms for each sample
sample_input_difference_norms = []
for sample in lhs_samples:
    input_difference_norm = np.linalg.norm(sample - baseline_values)
    sample_input_difference_norms.append(input_difference_norm)


sample_results = []
sample_timings = []
total_start_time = time.time()

for i, sample in enumerate(lhs_samples):
    print(f"Running sample {i+1}/{n_samples}")
    
    # Set design variable values
    for j, var_value in enumerate(sample):
        jax_inputs[j].value = np.array([var_value])
        jax_sim[jax_inputs[j]] = var_value
    
    # Run simulation
    sample_start_time = time.time()
    jax_sim.run()
    sample_end_time = time.time()
    sample_duration = sample_end_time - sample_start_time
    
    # Store results (geometry coefficients)
    # sample_result = [output.value.copy() for output in jax_outputs]
    sample_result = [inner_outer_front_left_rotor_constraint_value.value.copy(), inner_outer_rear_left_rotor_constraint_value.value.copy(),
                inner_inner_left_rotor_constraint_value.value.copy(), inner_outer_front_right_rotor_constraint_value.value.copy(),
                inner_outer_rear_right_rotor_constraint_value.value.copy(), inner_inner_right_rotor_constraint_value.value.copy(),
                mellowmax_constraint_inner_outer_front_left.value.copy(), mellowmax_constraint_inner_outer_rear_left.value.copy(),
                mellowmax_constraint_inner_inner_left.value.copy(), mellowmax_constraint_inner_outer_front_right.value.copy(),
                mellowmax_constraint_inner_outer_rear_right.value.copy(), mellowmax_constraint_inner_inner_right.value.copy()]
    sample_results.append(sample_result)

    # Store timing information
    sample_timings.append(sample_duration)

total_end_time = time.time()
total_duration = total_end_time - total_start_time

print("Latin Hypercube Sampling completed!")
print(f"Generated {len(sample_results)} simulation results")
print(f"Total time: {total_duration:.4f} seconds")
print(f"Average time per sample: {np.mean(sample_timings):.4f} seconds")
print(f"Min time: {np.min(sample_timings):.4f} seconds")
print(f"Max time: {np.max(sample_timings):.4f} seconds")
print(f"Std deviation: {np.std(sample_timings):.4f} seconds")
print(f"Input norm range: {np.min(sample_input_difference_norms):.4f} to {np.max(sample_input_difference_norms):.4f}")


# Save timing data to file
timing_data = {
    'sample_inputs': lhs_samples,
    'sample_results': sample_results,
    'sample_timings': sample_timings,
    'total_time': total_duration,
    'average_time': np.mean(sample_timings),
    'min_time': np.min(sample_timings),
    'max_time': np.max(sample_timings),
    'std_time': np.std(sample_timings),
    'n_samples': n_samples,
    'sample_input_norms': sample_input_difference_norms,
    'baseline_values': baseline_values,
}

# Save to file (optional)
save_file_path = 'examples/showcase_examples/lift_plus_cruise/'
with open(save_file_path + 'lift_plus_cruise_lhs_timing_results.pkl', 'wb') as f:
    pickle.dump(timing_data, f)

print(f"Timing data saved to '{save_file_path}lift_plus_cruise_lhs_timing_results.pkl'")

plt.figure(figsize=(10, 6))
plt.scatter(sample_input_difference_norms, sample_timings, alpha=0.6, s=20)
plt.xlabel('Norm of Input Difference from Baseline')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Input Perturbation Magnitude')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(sample_input_difference_norms, sample_timings, 1)
p = np.poly1d(z)
plt.semilogy(sample_input_difference_norms, p(sample_input_difference_norms), "r--", alpha=0.8, 
         label=f'Trend line (slope: {z[0]:.2e})')
plt.legend()

plt.tight_layout()
plt.savefig('examples/showcase_examples/lift_plus_cruise/lift_plus_cruise_time_vs_input_norm.png', dpi=300, bbox_inches='tight')
plt.show()


exit()
# endregion Create Data For Timing Study


