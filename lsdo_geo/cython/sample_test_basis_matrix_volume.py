import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
import scipy.sparse as sps

from basis_matrix_volume_py import get_basis_volume_matrix
from get_open_uniform_py import get_open_uniform

order_u = 3
order_v = 4
order_w = 5

num_control_points_u = 10
num_control_points_v = 11
num_control_points_w = 12

''' 
These are the number of evaluation points in each direction.
'''
num_points_u = 5
num_points_v = 6
num_points_w = 7


nnz = num_points_u * num_points_v * num_points_w * order_u * order_v * order_w
data = np.zeros(nnz)
row_indices = np.zeros(nnz, np.int32)
col_indices = np.zeros(nnz, np.int32)

'''
Below we create the points that we want to evaluate the bspline volume at
'''
u_vec = np.einsum('i,j,k->ijk', np.linspace(0., 1., num_points_u), np.ones(num_points_v), np.ones(num_points_w)).flatten()
v_vec = np.einsum('i,j,k->ijk', np.ones(num_points_u), np.linspace(0., 1., num_points_v), np.ones(num_points_w)).flatten()
w_vec = np.einsum('i,j,k->ijk', np.ones(num_points_u), np.ones(num_points_v), np.linspace(0., 1., num_points_w)).flatten()

#u_vec = np.linspace(0, 1, 4)
# print('u_vec',u_vec)
# print('v_vec',v_vec)
# print('w_vec',w_vec)

knot_vector_u = np.zeros(num_control_points_u + order_u)
knot_vector_v = np.zeros(num_control_points_v + order_v)
knot_vector_w = np.zeros(num_control_points_w + order_w)


get_open_uniform(order_u, num_control_points_u, knot_vector_u)
get_open_uniform(order_v, num_control_points_v, knot_vector_v)
get_open_uniform(order_w, num_control_points_w, knot_vector_w)

get_basis_volume_matrix(
    order_u, num_control_points_u, 0, u_vec, knot_vector_u,
    order_v, num_control_points_v, 0, v_vec, knot_vector_v,
    order_w, num_control_points_w, 0, w_vec, knot_vector_w,
    num_points_u * num_points_v * num_points_w, data, row_indices, col_indices,
)
# print(row_indices)
# print(col_indices)

# print(knot_vector_u)
# print(knot_vector_v)
# print(knot_vector_w)

#TODO: Fix the negative column index issue ... ???

basis0 = sps.csc_matrix(
    (data, (row_indices, col_indices)), 
    shape=(num_points_u * num_points_v * num_points_w, num_control_points_u * num_control_points_v * num_control_points_w),
)
# print(np.shape(basis0.toarray()))
#print(basis0.toarray())

cps = np.zeros((num_control_points_u, num_control_points_v, num_control_points_w, 3))
cps[:, :, :, 0] = np.einsum('i,j,k->ijk', np.linspace(0., 1., num_control_points_u), np.ones(num_control_points_v), np.ones(num_control_points_w))
cps[:, :, :, 1] = np.einsum('i,j,k->ijk', np.ones(num_control_points_u), np.linspace(0., 1., num_control_points_v), np.ones(num_control_points_w))
cps[:, :, :, 2] = np.einsum('i,j,k->ijk', np.ones(num_control_points_u), np.ones(num_control_points_v), np.linspace(0., 1., num_control_points_w))

cps = cps.reshape((num_control_points_u * num_control_points_v * num_control_points_w, 3))

# print('cps: ', cps)
# print('\n')

# print(type(basis0))
# dense_basis0 = basis0.todense()
# print('basis0: ', dense_basis0)
# print('\n')


pts = basis0.dot(cps)

# print('pts: ', pts)
# print('\n')

# print('pts[:, 0] : ', pts[:, 0])
# print('\n')

# print('pts[:, 1] : ', pts[:, 1])
# print('\n')

# print('pts[:, 2] : ', pts[:, 2])
# print('\n')

# ax = plt.axes(projection='3d')
# ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], 'gray')

# plt.scatter3d(pts[:, 0], pts[:, 1], pts[:, 2], 'ok')
# plt.plot(cps[:, 0], cps[:, 1], 'or', alpha = 0.5)

# print(cps)
# a = np.matmul(basis0.toarray().T, basis0.toarray())
# print(np.linalg.det(a))
# #print(pts)
# x2 = np.linalg.solve(np.matmul(basis0.toarray().T, basis0.toarray()), np.matmul(basis0.toarray().T, pts))
# x1,_,_,_ = np.linalg.lstsq(np.matmul(basis0.toarray().T, basis0.toarray()), np.matmul(basis0.toarray().T, pts), rcond=None)
# print(x1)
# print(x2)
# plt.show()