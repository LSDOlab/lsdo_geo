import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

from basis_matrix_curve_py import get_basis_curve_matrix
from curve_projection_py import compute_curve_projection
from get_open_uniform_py import get_open_uniform


order = 4
num_control_points = 6
num_points = 10

nnz = num_points * order
data = np.zeros(nnz)
row_indices = np.zeros(nnz, np.int32)
col_indices = np.zeros(nnz, np.int32)

u_vec = np.linspace(0., 1., num_points)

knot_vector = np.zeros(num_control_points+order)
get_open_uniform(order, num_control_points, knot_vector)

get_basis_curve_matrix(
    order, num_control_points, 0, u_vec, knot_vector,
    num_points, data, row_indices, col_indices,
)

basis0 = sps.csc_matrix(
    (data, (row_indices, col_indices)), 
    shape=(num_points, num_control_points),
)

cps = np.zeros((num_control_points, 3))
cps[:, 0] = np.linspace(0., 1., num_control_points)
cps[:, 1] = np.linspace(0., 1., num_control_points)
cps = cps.reshape((num_control_points, 3))

pts = basis0.dot(cps)

plt.plot(pts[:, 0], pts[:, 1])
# plt.plot(cps[:, 0], cps[:, 1], 'or')
#plt.show()

# --------------------------------------------------------------------  

num_points = 10
max_iter = 500
points = np.random.rand(num_points, 3)
u_vec = 0.5 * np.ones(num_points)

compute_curve_projection(
    order, num_control_points,
    num_points, max_iter,
    points.reshape(num_points * 3), 
    cps.reshape(num_control_points * 3),
    u_vec, knot_vector, 50
)

nnz = num_points * order
data = np.zeros(nnz)
row_indices = np.zeros(nnz, np.int32)
col_indices = np.zeros(nnz, np.int32)


get_basis_curve_matrix(
    order, num_control_points, 0, u_vec, knot_vector,
    num_points, data, row_indices, col_indices,
)
basis0 = sps.csc_matrix(
    (data, (row_indices, col_indices)), 
    shape=(num_points, num_control_points),
)

cps = cps.reshape((num_control_points, 3))

pts = basis0.dot(cps)


plt.plot(points[:, 0], points[:, 1], 'ok')
plt.plot(pts[:, 0], pts[:, 1], 'or')
plt.legend(["curve","random Points","projected points"])
plt.axis("equal")
plt.show()
print(points - pts)