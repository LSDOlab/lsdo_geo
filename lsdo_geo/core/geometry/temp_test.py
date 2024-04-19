import numpy as np
import csdl_alpha as csdl
from geometry_functions import rotate

recorder = csdl.Recorder(inline=True)
recorder.start()

points = csdl.Variable(shape=(3, 3), name='points', value=np.zeros((3, 3)))
point_1 = csdl.Variable(shape=(3,), value=np.array([1., 2., 3.]))
point_2 = csdl.Variable(shape=(3,), value=np.array([1., 0., 0.]))
point_3 = csdl.Variable(shape=(3,), value=np.array([0., 1., 0.]))

points = points.set(csdl.slice[0], point_1)
points = points.set(csdl.slice[1], point_2)
points = points.set(csdl.slice[2], point_3)

axis_origin = csdl.Variable(value=np.array([0., 0., 0.]))
axis_vector = csdl.Variable(value=np.array([0., 0., 1.]))

angles = csdl.Variable(value=np.array([45., 90.]))


rotated_points = rotate(points, axis_origin, axis_vector, angles)
print(rotated_points.value)