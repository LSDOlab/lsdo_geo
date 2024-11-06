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
wing_le_left = wing.project(np.array([12.356, -25.25, 7.618]), plot=False)
wing_le_right = wing.project(np.array([12.356, 25.25, 7.618]), plot=False)
wing_le_center = wing.project(np.array([8.892, 0., 8.633]), plot=False)
wing_qc = wing.project(np.array([10.25, 0., 8.5]), plot=False)

tail_te_right = h_tail.project(np.array([31.5, 6.75, 6.]), plot=False)
tail_te_left = h_tail.project(np.array([31.5, -6.75, 6.]), plot=False)
tail_le_right = h_tail.project(np.array([26.5, 6.75, 6.]), plot=False)
tail_le_left = h_tail.project(np.array([26.5, -6.75, 6.]), plot=False)
tail_te_center = h_tail.project(np.array([31.187, 0., 8.009]), plot=False)
tail_le_center = h_tail.project(np.array([27.428, 0., 8.009]), plot=False)
tail_qc = h_tail.project(np.array([24.15, 0., 8.]), plot=False)

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
fueslage_rear_points_parametric = fuselage.project(fuselage_rear)
fuselage_rear_point_on_pusher_disk_parametric = pp_disk.project(fuselage_rear)

# endregion

# # region rotor meshes
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
# # endregion

# region Projection for meshes
# region Wing camber mesh
wing_num_spanwise_vlm = 23
wing_num_chordwise_vlm = 5
leading_edge_line_parametric = wing.project(np.linspace(np.array([8.356, -26., 7.618]), np.array([8.356, 26., 7.618]), wing_num_spanwise_vlm), 
                                 direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
trailing_edge_line_parametric = wing.project(np.linspace(np.array([15.4, -25.250, 7.5]), np.array([15.4, 25.250, 7.5]), wing_num_spanwise_vlm), 
                                  direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
leading_edge_line = geometry.evaluate(leading_edge_line_parametric)
trailing_edge_line = geometry.evaluate(trailing_edge_line_parametric)
chord_surface = csdl.linear_combination(leading_edge_line, trailing_edge_line, wing_num_chordwise_vlm)
wing_upper_surface_wireframe_parametric = wing.project(chord_surface.value.reshape((wing_num_chordwise_vlm,wing_num_spanwise_vlm,3))+np.array([0., 0., 1.]), 
                                       direction=np.array([0., 0., -1.]), plot=False, grid_search_density_parameter=10.)
wing_lower_surface_wireframe_parametric = wing.project(chord_surface.value.reshape((wing_num_chordwise_vlm,wing_num_spanwise_vlm,3))+np.array([0., 0., -1.]), 
                                       direction=np.array([0., 0., 1.]), plot=False, grid_search_density_parameter=10.)
upper_surface_wireframe = geometry.evaluate(wing_upper_surface_wireframe_parametric)
lower_surface_wireframe = geometry.evaluate(wing_lower_surface_wireframe_parametric)
wing_camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((wing_num_chordwise_vlm, wing_num_spanwise_vlm, 3))
# endregion Wing camber mesh

# region Htail camber mesh
h_tail_num_spanwise_vlm = 11
h_tail_num_chordwise_vlm = 4
leading_edge_line_parametric = h_tail.project(np.linspace(np.array([26.5, -6.75, 6.]), np.array([26.5, 6.75, 6.]), h_tail_num_spanwise_vlm), 
                                 direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
trailing_edge_line_parametric = h_tail.project(np.linspace(np.array([31.5, -6.75, 6.]), np.array([31.5, 6.75, 6.]), h_tail_num_spanwise_vlm), 
                                  direction=np.array([0., 0., -1.]), grid_search_density_parameter=20.)
leading_edge_line = geometry.evaluate(leading_edge_line_parametric)
trailing_edge_line = geometry.evaluate(trailing_edge_line_parametric)
chord_surface = csdl.linear_combination(leading_edge_line, trailing_edge_line, h_tail_num_chordwise_vlm)
h_tail_upper_surface_wireframe_parametric = h_tail.project(chord_surface.value.reshape((h_tail_num_chordwise_vlm,h_tail_num_spanwise_vlm,3))+np.array([0., 0., 1.]), 
                                       direction=np.array([0., 0., -1.]), plot=False, grid_search_density_parameter=20.)
h_tail_lower_surface_wireframe_parametric = h_tail.project(chord_surface.value.reshape((h_tail_num_chordwise_vlm,h_tail_num_spanwise_vlm,3))+np.array([0., 0., -1.]), 
                                       direction=np.array([0., 0., 1.]), plot=False, grid_search_density_parameter=20.)
upper_surface_wireframe = geometry.evaluate(h_tail_upper_surface_wireframe_parametric)
lower_surface_wireframe = geometry.evaluate(h_tail_lower_surface_wireframe_parametric)
h_tail_camber_surface = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((h_tail_num_chordwise_vlm, h_tail_num_spanwise_vlm, 3))
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
# import vedo
# plotter = vedo.Plotter()
# plotter.show(plotting_elements, axes=0, viewup='z')
# endregion

# region Parameterization

constant_b_spline_curve_1_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=0, coefficients_shape=(1,))
linear_b_spline_curve_2_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))
linear_b_spline_curve_3_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(3,))
cubic_b_spline_curve_5_dof_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(5,))

# region Parameterization Setup
parameterization_solver = lsdo_geo.ParameterizationSolver()
parameterization_design_parameters = lsdo_geo.GeometricVariables()

# region Wing Parameterization setup
wing_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2,11,2), degree=(1,3,1))
wing_ffd_block_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(name='wing_sectional_parameterization',
                                                                            parameterized_points=wing_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

wing_chord_stretch_coefficients = csdl.Variable(name='wing_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
wing_chord_stretch_b_spline = lfs.Function(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                          coefficients=wing_chord_stretch_coefficients)

wing_wingspan_stretch_coefficients = csdl.Variable(name='wing_wingspan_stretch_coefficients', value=np.array([-0., 0.]))
wing_wingspan_stretch_b_spline = lfs.Function(name='wing_wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=wing_wingspan_stretch_coefficients)

wing_twist_coefficients = csdl.Variable(name='wing_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
wing_twist_b_spline = lfs.Function(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                          coefficients=wing_twist_coefficients)

wing_translation_x_coefficients = csdl.Variable(name='wing_translation_x_coefficients', value=np.array([0.]))
wing_translation_x_b_spline = lfs.Function(name='wing_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=wing_translation_x_coefficients)

wing_translation_z_coefficients = csdl.Variable(name='wing_translation_z_coefficients', value=np.array([0.]))
wing_translation_z_b_spline = lfs.Function(name='wing_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=wing_translation_z_coefficients)

parameterization_solver.add_parameter(parameter=wing_chord_stretch_coefficients)
parameterization_solver.add_parameter(parameter=wing_wingspan_stretch_coefficients, cost=1.e3)
parameterization_solver.add_parameter(parameter=wing_translation_x_coefficients)
parameterization_solver.add_parameter(parameter=wing_translation_z_coefficients)
# endregion Wing Parameterization setup

# region Horizontal Stabilizer setup
h_tail_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name='h_tail_ffd_block', entities=h_tail, num_coefficients=(2,11,2), degree=(1,3,1))
h_tail_ffd_block_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(name='h_tail_sectional_parameterization',
                                                                            parameterized_points=h_tail_ffd_block.coefficients,
                                                                            principal_parametric_dimension=1)

h_tail_chord_stretch_coefficients = csdl.Variable(name='h_tail_chord_stretch_coefficients', value=np.array([0., 0., 0.]))
h_tail_chord_stretch_b_spline = lfs.Function(name='h_tail_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space, 
                                          coefficients=h_tail_chord_stretch_coefficients)

h_tail_span_stretch_coefficients = csdl.Variable(name='h_tail_span_stretch_coefficients', value=np.array([-0., 0.]))
h_tail_span_stretch_b_spline = lfs.Function(name='h_tail_span_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=h_tail_span_stretch_coefficients)

# h_tail_twist_coefficients = csdl.Variable(name='h_tail_twist_coefficients', value=np.array([0., 0., 0., 0., 0.]))
# h_tail_twist_b_spline = lfs.Function(name='h_tail_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
#                                           coefficients=h_tail_twist_coefficients)

h_tail_translation_x_coefficients = csdl.Variable(name='h_tail_translation_x_coefficients', value=np.array([0.]))
h_tail_translation_x_b_spline = lfs.Function(name='h_tail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=h_tail_translation_x_coefficients)
h_tail_translation_z_coefficients = csdl.Variable(name='h_tail_translation_z_coefficients', value=np.array([0.]))
h_tail_translation_z_b_spline = lfs.Function(name='h_tail_translation_z_b_spline', space=constant_b_spline_curve_1_dof_space,
                                          coefficients=h_tail_translation_z_coefficients)

parameterization_solver.add_parameter(parameter=h_tail_chord_stretch_coefficients)
parameterization_solver.add_parameter(parameter=h_tail_span_stretch_coefficients)
parameterization_solver.add_parameter(parameter=h_tail_translation_x_coefficients)
parameterization_solver.add_parameter(parameter=h_tail_translation_z_coefficients)
# endregion Horizontal Stabilizer setup

# region Fuselage setup
fuselage_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=[fuselage, nose_hub], num_coefficients=(2,2,2), degree=(1,1,1))
fuselage_ffd_block_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(name='fuselage_sectional_parameterization',
                                                                            parameterized_points=fuselage_ffd_block.coefficients,
                                                                            principal_parametric_dimension=0)
# fuselage_ffd_block_sectional_parameterization.add_sectional_translation(name='sectional_fuselage_stretch', axis=0)

fuselage_stretch_coefficients = csdl.Variable(name='fuselage_stretch_coefficients', shape=(2,), value=np.array([0., -0.]))
fuselage_stretch_b_spline = lfs.Function(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                          coefficients=fuselage_stretch_coefficients)

parameterization_solver.add_parameter(parameter=fuselage_stretch_coefficients)
# endregion

# region Lift Rotors setup
lift_rotor_ffd_blocks = []
lift_rotor_sectional_parameterizations = []
lift_rotor_parameterization_b_splines = []
for i, component_set in enumerate(lift_rotor_related_components):
    rotor_ffd_block = lsdo_geo.construct_ffd_block_around_entities(name=f'{component_set[0].name[:3]}_rotor_ffd_block', entities=component_set, num_coefficients=(2,2,2), degree=(1,1,1))
    rotor_ffd_block_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(name=f'{component_set[0].name[:3]}_rotor_sectional_parameterization',
                                                                                parameterized_points=rotor_ffd_block.coefficients,
                                                                                principal_parametric_dimension=2)
    
    rotor_stretch_coefficient = csdl.Variable(name=f'{component_set[0].name[:3]}_rotor_stretch_coefficient', shape=(1,), value=0.)
    lift_rotor_sectional_stretch_b_spline = lfs.Function(name=f'{component_set[0].name[:3]}_rotor_sectional_stretch_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                                coefficients=rotor_stretch_coefficient)
    
    lift_rotor_ffd_blocks.append(rotor_ffd_block)
    lift_rotor_sectional_parameterizations.append(rotor_ffd_block_sectional_parameterization)
    lift_rotor_parameterization_b_splines.append(lift_rotor_sectional_stretch_b_spline)                 

    parameterization_solver.add_parameter(parameter=rotor_stretch_coefficient)
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

# import vedo
# plotter = vedo.Plotter()
# plotter.show(plotting_elements, axes=0, viewup='z')
# exit()

# # endregion Plot parameterization

# endregion Parameterization Setup

# region Parameterization Solver Setup Evaluations

# region Wing Parameterization Evaluation for Parameterization Solver
section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_wingspan_stretch = wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_wing_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_translation_x = wing_translation_x_b_spline.evaluate(section_parametric_coordinates)
sectional_wing_translation_z = wing_translation_z_b_spline.evaluate(section_parametric_coordinates)

sectional_parameters = lsdo_geo.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_wing_chord_stretch},
    translations={1: sectional_wing_wingspan_stretch, 0: sectional_wing_translation_x, 2: sectional_wing_translation_z}
)

wing_ffd_block_coefficients = wing_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
wing.set_coefficients(wing_coefficients)
# geometry.plot()

# endregion Wing Parameterization Evaluation for Parameterization Solver

# region Horizontal Stabilizer Parameterization Evaluation for Parameterization Solver
section_parametric_coordinates = np.linspace(0., 1., h_tail_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_h_tail_chord_stretch = h_tail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_span_stretch = h_tail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
# sectional_h_tail_twist = h_tail_twist_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_translation_x = h_tail_translation_x_b_spline.evaluate(section_parametric_coordinates)
sectional_h_tail_translation_z = h_tail_translation_z_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = {
#     'sectional_h_tail_chord_stretch':sectional_h_tail_chord_stretch,
#     'sectional_h_tail_span_stretch':sectional_h_tail_span_stretch,
#     # 'sectional_h_tail_twist':sectional_h_tail_twist,
#     'sectional_h_tail_translation_x':sectional_h_tail_translation_x,
#     'sectional_h_tail_translation_z':sectional_h_tail_translation_z
#                         }
sectional_parameters = lsdo_geo.VolumeSectionalParameterizationInputs(
    stretches={0: sectional_h_tail_chord_stretch},
    translations={1: sectional_h_tail_span_stretch, 0: sectional_h_tail_translation_x, 2: sectional_h_tail_translation_z}
)

h_tail_ffd_block_coefficients = h_tail_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
h_tail_coefficients = h_tail_ffd_block.evaluate(h_tail_ffd_block_coefficients, plot=False)
h_tail.set_coefficients(coefficients=h_tail_coefficients)
# geometry.plot()
# endregion

# region Fuselage Parameterization Evaluation for Parameterization Solver
section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sectional_parameterization.num_sections).reshape((-1,1))
sectional_fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

# sectional_parameters = {'sectional_fuselage_stretch':sectional_fuselage_stretch}
sectional_parameters = lsdo_geo.VolumeSectionalParameterizationInputs(
    translations={0: sectional_fuselage_stretch}
)

fuselage_ffd_block_coefficients = fuselage_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
fuselage_and_nose_hub_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
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

    sectional_parameters = lsdo_geo.VolumeSectionalParameterizationInputs(
        stretches={0: sectional_stretch, 1:sectional_stretch}
    )

    rotor_ffd_block_coefficients = rotor_ffd_block_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    rotor_coefficients = rotor_ffd_block.evaluate(rotor_ffd_block_coefficients, plot=False)
    for i, component in enumerate(component_set):
        component.set_coefficients(rotor_coefficients[i])
    # geometry.plot()

# endregion Lift Rotors Parameterization Evaluation for Parameterization Solver

# region Lift Rotors rigid body translation
for i, component_set in enumerate(lift_rotor_related_components):
    # disk = component_set[0]
    # blade_1 = component_set[1]
    # blade_2 = component_set[2]
    # hub = component_set[3]

    boom = boom_components[i]

    # Add rigid body translation
    rigid_body_translation = csdl.Variable(shape=(3,), value=0., name=f'{component_set[0].name[:3]}_rotor_rigid_body_translation')

    for component in component_set:
        for function in component.functions.values():
            function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

    for function in boom.functions.values():
        function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

    parameterization_solver.add_parameter(parameter=rigid_body_translation)
# endregion Lift Rotors rigid body translation

# region pusher rigid body translation
rigid_body_translation = csdl.Variable(shape=(3,), value=0., name='pp_rotor_rigid_body_translation')
for component in pp_components:
    for function in component.functions.values():
        function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

parameterization_solver.add_parameter(parameter=rigid_body_translation)
# endregion pusher rigid body translation

# region Vertical Stabilizer rigid body translation
rigid_body_translation = csdl.Variable(shape=(3,), value=0., name='pp_rotor_rigid_body_translation')
for function in v_tail.functions.values():
    function.coefficients = function.coefficients + csdl.expand(rigid_body_translation, function.coefficients.shape, action='k->ijk')

parameterization_solver.add_parameter(parameter=rigid_body_translation)
# endregion Vertical Stabilizer rigid body translation

# endregion Parameterization Solver Setup Evaluations

# region Define Design Parameters

# region wing design parameters
wing_span_computed = csdl.norm(geometry.evaluate(wing_le_right) - geometry.evaluate(wing_le_left))
wing_root_chord_computed = csdl.norm(geometry.evaluate(wing_te_center) - geometry.evaluate(wing_le_center))
wing_tip_chord_left_computed = csdl.norm(geometry.evaluate(wing_te_left) - geometry.evaluate(wing_le_left))
wing_tip_chord_right_computed = csdl.norm(geometry.evaluate(wing_te_right) - geometry.evaluate(wing_le_right))

wing_span = csdl.Variable(name='wing_span', value=np.array([50.]))
wing_root_chord = csdl.Variable(name='wing_root_chord', value=np.array([5.]))
wing_tip_chord = csdl.Variable(name='wing_tip_chord_left', value=np.array([1.]))

parameterization_design_parameters.add_variable(computed_value=wing_span_computed, desired_value=wing_span)
parameterization_design_parameters.add_variable(computed_value=wing_root_chord_computed, desired_value=wing_root_chord)
parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_left_computed, desired_value=wing_tip_chord)
parameterization_design_parameters.add_variable(computed_value=wing_tip_chord_right_computed, desired_value=wing_tip_chord)
# endregion wing design parameters

# region h_tail design parameterization inputs
h_tail_span_computed = csdl.norm(geometry.evaluate(tail_le_right) - geometry.evaluate(tail_le_left))
h_tail_root_chord_computed = csdl.norm(geometry.evaluate(tail_te_center) - geometry.evaluate(tail_le_center))
h_tail_tip_chord_left_computed = csdl.norm(geometry.evaluate(tail_te_left) - geometry.evaluate(tail_le_left))
h_tail_tip_chord_right_computed = csdl.norm(geometry.evaluate(tail_te_right) - geometry.evaluate(tail_le_right))

h_tail_span = csdl.Variable(name='h_tail_span', value=np.array([12.]))
h_tail_root_chord = csdl.Variable(name='h_tail_root_chord', value=np.array([3.]))
h_tail_tip_chord = csdl.Variable(name='h_tail_tip_chord_left', value=np.array([2.]))

parameterization_design_parameters.add_variable(computed_value=h_tail_span_computed, desired_value=h_tail_span)
parameterization_design_parameters.add_variable(computed_value=h_tail_root_chord_computed, desired_value=h_tail_root_chord)
parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_left_computed, desired_value=h_tail_tip_chord)
parameterization_design_parameters.add_variable(computed_value=h_tail_tip_chord_right_computed, desired_value=h_tail_tip_chord)
# endregion h_tail design parameterization inputs

# region tail moment arm variables
tail_moment_arm_computed = csdl.norm(geometry.evaluate(tail_qc) - geometry.evaluate(wing_qc))
tail_moment_arm = csdl.Variable(name='tail_moment_arm', value=np.array([25.]))
parameterization_design_parameters.add_variable(computed_value=tail_moment_arm_computed, desired_value=tail_moment_arm)

wing_fuselage_connection = geometry.evaluate(wing_te_center) - geometry.evaluate(fuselage_wing_te_center)
h_tail_fuselage_connection = geometry.evaluate(tail_te_center) - geometry.evaluate(fuselage_tail_te_center)
parameterization_design_parameters.add_variable(computed_value=wing_fuselage_connection, desired_value=wing_fuselage_connection.value)
parameterization_design_parameters.add_variable(computed_value=h_tail_fuselage_connection, desired_value=h_tail_fuselage_connection.value)

# endregion tail moment arm variables

# region v-tail connection
vtail_fuselage_connection_point = geometry.evaluate(v_tail.project(np.array([30.543, 0., 8.231])))
vtail_fuselage_connection = geometry.evaluate(fueslage_rear_points_parametric) - vtail_fuselage_connection_point
parameterization_design_parameters.add_variable(computed_value=vtail_fuselage_connection, desired_value=vtail_fuselage_connection.value)

# endregion v-tail connection

# region lift + pusher rotor parameterization inputs
pusher_fuselage_connection = geometry.evaluate(fueslage_rear_points_parametric) - geometry.evaluate(fuselage_rear_point_on_pusher_disk_parametric)
parameterization_design_parameters.add_variable(computed_value=pusher_fuselage_connection, desired_value=pusher_fuselage_connection.value)

flo_radius = fro_radius = front_outer_radius = csdl.Variable(name='front_outer_radius', value=10/2)
fli_radius = fri_radius = front_inner_radius = csdl.Variable(name='front_inner_radius', value=10/2)
rlo_radius = rro_radius = rear_outer_radius = csdl.Variable(name='rear_outer_radius', value=10/2)
rli_radius = rri_radius = rear_inner_radius = csdl.Variable(name='rear_inner_radius', value=10/2)
dv_radius_list = [rlo_radius, rli_radius, rri_radius, rro_radius, flo_radius, fli_radius, fri_radius, fro_radius]

boom_points = [boom_rlo, boom_rli, boom_rri, boom_rro, boom_flo, boom_fli, boom_fri, boom_fro]
boom_points_on_wing = [wing_boom_rlo, wing_boom_rli, wing_boom_rri, wing_boom_rro, wing_boom_flo, wing_boom_fli, wing_boom_fri, wing_boom_fro]
rotor_prefixes = ['rlo', 'rli', 'rri', 'rro', 'flo', 'fli', 'fri', 'fro']

for i in range(len(boom_points)):
    boom_connection = geometry.evaluate(boom_points[i]) - geometry.evaluate(boom_points_on_wing[i])

    parameterization_design_parameters.add_variable(computed_value=boom_connection, desired_value=boom_connection.value)
    
    component_rotor_edges = rotor_edges[i]
    radius_computed = csdl.norm(geometry.evaluate(component_rotor_edges[0]) - geometry.evaluate(component_rotor_edges[1]))/2
    parameterization_design_parameters.add_variable(computed_value=radius_computed, desired_value=dv_radius_list[i])

# endregion lift + pusher rotor parameterization inputs

# endregion Define Design Parameters

# geometry.plot()
parameterization_solver.evaluate(parameterization_design_parameters)
# geometry.plot()

# endregion

# region Mesh Evaluation
upper_surface_wireframe = geometry.evaluate(wing_upper_surface_wireframe_parametric)
lower_surface_wireframe = geometry.evaluate(wing_lower_surface_wireframe_parametric)
wing_vlm_mesh = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((wing_num_chordwise_vlm, wing_num_spanwise_vlm, 3))

upper_surface_wireframe = geometry.evaluate(h_tail_upper_surface_wireframe_parametric)
lower_surface_wireframe = geometry.evaluate(h_tail_lower_surface_wireframe_parametric)
h_tail_vlm_mesh = csdl.linear_combination(upper_surface_wireframe, lower_surface_wireframe, 1).reshape((h_tail_num_chordwise_vlm, h_tail_num_spanwise_vlm, 3))

beam_tops = wing.evaluate(beam_top_parametric)
beam_bottoms = wing.evaluate(beam_bottom_parametric)
wing_beam_mesh = csdl.linear_combination(beam_tops, beam_bottoms, 1).reshape((num_beam_nodes, 3))
beam_heights = csdl.norm(beam_tops - beam_bottoms, axes=(1,))
# endregion Mesh Evaluation
