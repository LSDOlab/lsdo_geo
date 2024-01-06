import lsdo_geo
import time
from lsdo_geo.core.geometry.geometry_functions import import_geometry
import m3l
import numpy as np
from python_csdl_backend import Simulator

geometry = import_geometry('lsdo_geo/splines/b_splines/sample_geometries/rectangular_wing.stp', parallelize=False)
geometry.refit(parallelize=False, fit_resolution=(50,50))

m3l_model = m3l.Model()

# axis_origin = geometry.evaluate(geometry.project(np.array([0.5, -10., 0.5])))
# axis_vector = geometry.evaluate(geometry.project(np.array([0.5, 10., 0.5]))) - axis_origin
axis_origin = np.array([0.5, 0., 0.])
axis_vector = np.array([0., 1., 0.])
angles = 45

leading_edge_parametric_coordinates = [
        ('WingGeom, 0, 3', np.array([1.,  0.])),
        ('WingGeom, 0, 3', np.array([0.777, 0.])),
        ('WingGeom, 0, 3', np.array([0.555, 0.])),
        ('WingGeom, 0, 3', np.array([0.333, 0.])),
        ('WingGeom, 0, 3', np.array([0.111, 0.])),
        ('WingGeom, 1, 8', np.array([0.111 , 0.])),
        ('WingGeom, 1, 8', np.array([0.333, 0.])),
        ('WingGeom, 1, 8', np.array([0.555, 0.])),
        ('WingGeom, 1, 8', np.array([0.777, 0.])),
        ('WingGeom, 1, 8', np.array([1., 0.])),
    ]

geometry2 = geometry.copy()
geometry2.coefficients.name = 'geometry2_coefficients'
geometry3 = geometry.copy()
geometry3.coefficients.name = 'geometry3_coefficients'

geometry2.rotate(axis_origin, axis_vector, angles)
geometry2_csdl_name = geometry2.coefficients.operation.name + '.' + geometry2.coefficients.name

leading_edge2 = geometry2.evaluate(leading_edge_parametric_coordinates)

geometry3.rotate(axis_origin, axis_vector, -angles)
geometry3_csdl_name = geometry3.coefficients.operation.name + '.' + geometry3.coefficients.name

# geometry.plot()

# geometry2.plot()

# geometry3.plot()

# leading_edge1 = geometry.evaluate(leading_edge_parametric_coordinates)
leading_edge3 = geometry3.evaluate(leading_edge_parametric_coordinates)

# m3l_model.register_output(leading_edge1)
m3l_model.register_output(leading_edge2)
m3l_model.register_output(leading_edge3)
# m3l_model.register_output(geometry2.coefficients)
# m3l_model.register_output(geometry3.coefficients)

csdl_model = m3l_model.assemble()
sim = Simulator(csdl_model)
sim.run()

# leading_edge1 = sim[leading_edge1.operation.name + '.' + leading_edge1.name].reshape((-1,3))
leading_edge2 = sim[leading_edge2.operation.name + '.' + leading_edge2.name].reshape((-1,3))
leading_edge3 = sim[leading_edge3.operation.name + '.' + leading_edge3.name].reshape((-1,3))

geometry2_coefficients = sim[geometry2_csdl_name].reshape((-1,3))
import vedo
geometry2_coefficients_points = vedo.Points(geometry2_coefficients, r=10)
plotter = vedo.Plotter()
plotter.show(geometry2_coefficients_points, vedo.Points(leading_edge2, r=12, c='g'), axes=1)
# plotter.show(geometry2_coefficients_points, axes=1)

geometry3_coefficients = sim[geometry3_csdl_name].reshape((-1,3))
import vedo
geometry3_coefficients_points = vedo.Points(geometry3_coefficients, r=10)
plotter = vedo.Plotter()
plotter.show(geometry3_coefficients_points, vedo.Points(leading_edge3, r=12, c='g'), axes=1)
# plotter.show(geometry3_coefficients_points, axes=1)

# print(leading_edge1)
# geometry.plot_meshes([leading_edge1])
print(leading_edge2)
geometry2.plot_meshes([leading_edge2])
print(leading_edge3)
geometry3.plot_meshes([leading_edge3])
print('hi')