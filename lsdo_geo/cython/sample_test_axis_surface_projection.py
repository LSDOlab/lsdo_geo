import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from basis_matrix_surface_py import get_basis_surface_matrix
from surface_projection_py import compute_surface_projection
from get_open_uniform_py import get_open_uniform

order_u = 4
order_v = 4
num_control_points_u = 6
num_control_points_v = 6
num_points_u = 10
num_points_v = 10

nnz = num_points_u * num_points_v * order_u * order_v
data = np.zeros(nnz)
row_indices = np.zeros(nnz, np.int32)
col_indices = np.zeros(nnz, np.int32)

u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()

knot_vector_u = np.zeros(num_control_points_u+order_u)
knot_vector_v = np.zeros(num_control_points_v+order_v)

get_open_uniform(order_u, num_control_points_u, knot_vector_u)
get_open_uniform(order_v, num_control_points_v, knot_vector_v)

get_basis_surface_matrix(
    order_u, num_control_points_u, 0, u_vec, knot_vector_u,
    order_v, num_control_points_v, 0, v_vec, knot_vector_v,
    num_points_u * num_points_v, data, row_indices, col_indices,
)
basis0 = sps.csc_matrix(
    (data, (row_indices, col_indices)), 
    shape=(num_points_u * num_points_v, num_control_points_u * num_control_points_v),
)

cps = np.zeros((num_control_points_u, num_control_points_v, 3))
cps[:, :, 0] = np.einsum('i,j->ij', np.linspace(0., 1., num_control_points_u), np.ones(num_control_points_v))
cps[:, :, 1] = np.einsum('i,j->ij', np.ones(num_control_points_u), np.linspace(0., 1., num_control_points_v))
cps[:, :, 2] = np.einsum('i,j->ij', np.ones(num_control_points_u), -(np.linspace(0., 1., num_control_points_v)))
cps = cps.reshape((num_control_points_u * num_control_points_v, 3))
pts = basis0.dot(cps)

num_points = 2
max_iter = 300
#[0.86443701 0.44101438]
#[0.18034219 0.09606887]
v2 =np.array([-0.3, -0.2, -0.5])#[0.1,-0.2, -0.5] #[-0.3, -0.2, -0.5]
points = np.vstack(([0.8,0.9,0.3],[0.8,0.9,0.3], ))
u_vec = 0.5 * np.ones(num_points)
v_vec = 0.5 * np.ones(num_points)
axis = np.vstack(([0,0,0],v2,))
knot_vector_u = np.zeros(num_control_points_u+order_u)
knot_vector_v = np.zeros(num_control_points_v+order_v)
get_open_uniform(order_u, num_control_points_u, knot_vector_u)
get_open_uniform(order_v, num_control_points_v, knot_vector_v)
compute_surface_projection(
    order_u, num_control_points_u,
    order_v, num_control_points_v,
    num_points, max_iter,
    points.reshape(num_points * 3), 
    cps.reshape(num_control_points_u * num_control_points_v * 3),
    knot_vector_u, knot_vector_v,
    u_vec, v_vec,50,
    axis.reshape(num_points * 3),
)

print(u_vec)
print(v_vec)

nnz = num_points * order_u * order_v
data = np.zeros(nnz)
row_indices = np.zeros(nnz, np.int32)
col_indices = np.zeros(nnz, np.int32)
knot_vector_u = np.zeros(num_control_points_u+order_u)
knot_vector_v = np.zeros(num_control_points_v+order_v)
get_open_uniform(order_u, num_control_points_u, knot_vector_u)
get_open_uniform(order_v, num_control_points_v, knot_vector_v)

get_basis_surface_matrix(
    order_u, num_control_points_u, 0, u_vec, knot_vector_u,
    order_v, num_control_points_v, 0, v_vec, knot_vector_v,
    num_points, data, row_indices, col_indices,
)
basis0 = sps.csc_matrix(
    (data, (row_indices, col_indices)), 
    shape=(num_points, num_control_points_u * num_control_points_v),
)

cps = cps.reshape((num_control_points_u * num_control_points_v, 3))

pts = basis0.dot(cps)

def set_ax1_equal(ax):
    '''Make ax1 of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


fig=plt.figure()
    
ax1 = fig.gca(projection='3d')

ax1.scatter(pts[:,0],pts[:,1],pts[:,2],marker = 'x')
ax1.scatter(points[:,0],points[:,1],points[:,2],marker = '^')
ax1.scatter(cps[:,0],cps[:,1],cps[:,2],marker = 'o')

for i in range(len(pts)):
    ax1.plot([points[i,0],pts[i,0]],[points[i,1],pts[i,1]],[points[i,2],pts[i,2]],linestyle='--')

v3 = points[0,:] + 2*v2 
xv3 =[v3[0], points[0,0]]
yv3 =[v3[1], points[0,1]]
zv3 =[v3[2], points[0,2]]

v1 = pts[1,:] - points[1,:]
n1 = np.linalg.norm(v1)
n2 = np.linalg.norm(v2)
vv3 = [-xv3[1]+xv3[0], -yv3[1]+yv3[0], -zv3[1]+zv3[0]]
n3 = np.linalg.norm(vv3)
print(np.dot(v1/n1,v2/n2))
print(np.dot(v2/n2,vv3/n3))

ax1.plot(xv3, yv3, zv3, c='r', marker = '*')
ax1.set_aspect('auto')
plt.legend(["without direction","with direction","axis",])
set_ax1_equal(ax1)
plt.show()
