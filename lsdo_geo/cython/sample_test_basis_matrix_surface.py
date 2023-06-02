import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

from basis_matrix_surface_py import get_basis_surface_matrix
from get_open_uniform_py import get_open_uniform

order_u = 4
order_v = 4
num_control_points_u = 4
num_control_points_v = 4
num_points_u = 5
num_points_v = 6

nnz = num_points_u * num_points_v * order_u * order_v
data = np.zeros(nnz)
row_indices = np.zeros(nnz, np.int32)
col_indices = np.zeros(nnz, np.int32)

u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()
#u_vec = np.linspace(0, 1, 4)
# print('u_vec',u_vec)
# print('v_vec',v_vec)
knot_vector_u = np.zeros(num_control_points_u+order_u)
knot_vector_v = np.zeros(num_control_points_v+order_v)

get_open_uniform(order_u, num_control_points_u, knot_vector_u)
get_open_uniform(order_v, num_control_points_v, knot_vector_v)

get_basis_surface_matrix(
    order_u, num_control_points_u, 0, u_vec, knot_vector_u,
    order_v, num_control_points_v, 0, v_vec, knot_vector_v,
    num_points_u * num_points_v, data, row_indices, col_indices,
)
print(knot_vector_u)
print(knot_vector_v)
basis0 = sps.csc_matrix(
    (data, (row_indices, col_indices)), 
    shape=(num_points_u * num_points_v, num_control_points_u * num_control_points_v),
)
print(np.shape(basis0.toarray()))
#print(basis0.toarray())

cps = np.zeros((num_control_points_u, num_control_points_v, 2))
cps[:, :, 0] = np.einsum('i,j->ij', np.linspace(0., 1., num_control_points_u), np.ones(num_control_points_v))
cps[:, :, 1] = np.einsum('i,j->ij', np.ones(num_control_points_u), np.linspace(0., 1., num_control_points_v))
cps = cps.reshape((num_control_points_u * num_control_points_v, 2))

pts = basis0.dot(cps)

plt.plot(pts[:, 0], pts[:, 1], 'ok')
plt.plot(cps[:, 0], cps[:, 1], 'or', alpha = 0.5)

print(cps)
a = np.matmul(basis0.toarray().T, basis0.toarray())
print(np.linalg.det(a))
#print(pts)
x2 = np.linalg.solve(np.matmul(basis0.toarray().T, basis0.toarray()), np.matmul(basis0.toarray().T, pts))
x1,_,_,_ = np.linalg.lstsq(np.matmul(basis0.toarray().T, basis0.toarray()), np.matmul(basis0.toarray().T, pts), rcond=None)
print(x1)
print(x2)
#plt.show()