import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from basis_matrix_surface_py import get_basis_surface_matrix
from surface_projection_py import compute_surface_projection
from get_open_uniform_py import get_open_uniform

import os
from pathlib import Path

import pyiges
from pyiges import examples, geometry


print('eVTOL-sw-bounded.IGS')
#here = os.path.dirname(Path.abspath(__file__).parent.absolute())
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
file_name = os.path.join(path, 'DesignGeometry/CAD/eVTOL-sw-bounded.IGS')
iges = pyiges.read(file_name)#examples.impeller
bsurf = iges.bspline_surfaces()
print('Number of surfaces: ', len(bsurf))
#ns = 27
temp = 1e16
for ns in range(27,28):##27#len(bsurf)
    print(ns)
    order_u = bsurf[ns]._m1 + 1
    order_v = bsurf[ns]._m2 + 1
    num_control_points_u = bsurf[ns]._k1 + 1 
    num_control_points_v = bsurf[ns]._k2 + 1 

    if num_control_points_u == 2 or num_control_points_v == 2:
        cps1 = bsurf[ns]._cp[0]
        for j in range(len(bsurf[ns]._cp)):
            if j == 0:
                pass
            else:
                if j % 2 == 0:
                    cps1 = np.vstack((cps1,bsurf[ns]._cp[j]))
                elif j == 1:
                    cps2 = bsurf[ns]._cp[1]
                else:
                    cps2 = np.vstack((cps2,bsurf[ns]._cp[j]))
        #print(np.shape(cps1))
        #print(np.shape(cps2))
        #cps1 = cps.reshape((len(cps1), 3))
        #cps2 = cps.reshape((len(cps2), 3))
        num1 = 5
        cps = np.vstack((cps1,cps2))
        #print(np.shape(cps))
    else:
        cps = bsurf[ns]._cp[0]
        for j in range(len(bsurf[ns]._cp)):
            if j == 0:
                pass
            else:
                cps =  np.vstack((cps,bsurf[ns]._cp[j]))
    num1 = 5
    cps = cps.reshape((num_control_points_u * num_control_points_v, 3)) 
    if num_control_points_u == 2 or num_control_points_v == 2:
        if num_control_points_u == 2:
            interpolate_matrix = np.zeros((num1*num_control_points_v, num_control_points_u*num_control_points_v))
            num_control_points = num_control_points_v
            num_control_points_u = num1
            order_u = 4 
        else:
            interpolate_matrix = np.zeros((num1*num_control_points_u, num_control_points_u*num_control_points_v))
            num_control_points = num_control_points_u
            num_control_points_v = num1
            order_v = 4
        j = 0
        for i in range(num1*num_control_points):
            if i%5 == 0 and j<num_control_points:
                interpolate_matrix[i,j] = np.linspace(0,1,num1)[0]
                interpolate_matrix[i,j+num_control_points] = np.linspace(0,1,num1)[-1]
            if i%5 == 1 and j<num_control_points:
                    interpolate_matrix[i,j] = np.linspace(0,1,num1)[1]
                    interpolate_matrix[i,j+num_control_points] = np.linspace(0,1,num1)[-2]
            if i%5 == 2 and j<num_control_points:
                    interpolate_matrix[i,j] = np.linspace(0,1,num1)[2]
                    interpolate_matrix[i,j+num_control_points] = np.linspace(0,1,num1)[-3]                         
            if i%5 == 3 and j<num_control_points:
                    interpolate_matrix[i,j] = np.linspace(0,1,num1)[3]
                    interpolate_matrix[i,j+num_control_points] = np.linspace(0,1,num1)[-4] 
            if i%5 == 4 and j<num_control_points:
                    interpolate_matrix[i,j] = np.linspace(0,1,num1)[4]
                    interpolate_matrix[i,j+num_control_points] = np.linspace(0,1,num1)[-5]
                    j = j+1                             
    else:
        interpolate_matrix = np.identity(num_control_points_u * num_control_points_v)
        cps = cps.reshape((num_control_points_u * num_control_points_v, 3)) 
    #print(interpolate_matrix)
    #print(np.shape(interpolate_matrix))
    #print(np.shape(cps))
    cps = np.matmul(interpolate_matrix,cps)    
    #print(np.shape(cps))
               
    num_points = 1
    max_iter = 500
    point = np.array([54666.65, 14666.65, 53333.35])
    print(np.shape(point))
    points = np.array((point),ndmin=2)
    print(np.shape(points))
    axis = np.array([0., 0., 0.])
    u_vec = np.ones(num_points)
    v_vec = np.ones(num_points)
    compute_surface_projection(
        order_u, num_control_points_u,
        order_v, num_control_points_v,
        num_points, max_iter,
        points.reshape(num_points * 3), 
        cps.reshape((num_control_points_u * num_control_points_v * 3)),
        u_vec, v_vec,100,
        axis.reshape(num_points * 3),
    )
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
    


    pts = basis0.dot(cps)
    dist = np.linalg.norm(pts-points)

    if dist < temp:
        pts_final = pts
        cps_final = cps
        temp = dist
        ns_final = ns
        u_vec_final = u_vec
        v_vec_final = v_vec
        basis0_final = basis0
        interpolate_matrix_final = interpolate_matrix

print('ns_final', ns_final)
pts1 = pts_final
cps1 = cps_final
print(u_vec_final)
print(v_vec_final)
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
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

check = 0
fig=plt.figure()
#print(cps)
#print(np.shape(cps))
#np.savetxt("cps.txt",cps1,fmt='%12.4f') 
#np.savetxt("cps_x.txt",cps[:,0],fmt='%.18f') 
#np.savetxt("cps_y.txt",cps[:,1]) 
#np.savetxt("cps_z.txt",cps[:,2]) 
ax1 = fig.gca(projection='3d')
#print(pts[check,:])
print(pts1)
print(points)
#ax1.scatter(pts[:,0],pts[:,1],pts[:,2],marker = 'x')
#ax1.scatter(points[:,0],points[:,1],points[:,2],marker = '^')
ax1.scatter(pts1[:,0],pts1[:,1],pts1[:,2],marker = 'x',s=50)
ax1.scatter(points[:,0],points[:,1],points[:,2],marker = '^')
ax1.scatter(cps1[:,0],cps1[:,1],cps1[:,2],marker = 'o')
plt.legend(["Projected points","Test points","Control points of b-spline surface 27"])#"answer","random","cp"
plt.show()
x1 = [pts1[check,0], points[check,0]]
y1 = [pts1[check,1], points[check,1]]
z1 = [pts1[check,2], points[check,2]]
ax1.plot(x1, y1, z1, c='r')
x3 = [0, -5000*0.01890829]
y3 = [0, -5000*0.94024978]
z3 = [0, -5000*-0.33996005]
#ax1.plot(x3, y3, z3,c='b')
ax1.set_aspect('auto')
set_axes_equal(ax1)

v1 = points[check,:] - pts1[check,:]
v2 = [0.01890829, 0.94024978, -0.33996005]#check=1 ,nn=60
n1 = np.linalg.norm(v1)
n2 = np.linalg.norm(v2)
#print('normal',v1/n1)
#print('normal',v2/n2)
#print(np.dot(v1/n1,v2/n2))# rounding errors #-0.9916760645313097

fig2=plt.figure()
plt.scatter(pts[:,0],pts[:,1],marker = 'x')
plt.scatter(points[:,0],points[:,1],marker = '^')
plt.scatter(cps[:,0],cps[:,1],marker = 'o')
plt.legend(["Projected points","Random test points","Control points of b-spline surfaces"])#"answer","random","cp"

#54111.11111111,32444.44444444,55555.55555556 27 1
#54333.33333333,29333.33333333,54666.66666667 2
#27#
#points = np.vstack(( [55000.,20000.,52000.],[54333.3,9333.3,54666.7]))
#point1 = np.array([55000., 20000., 52000.])
#point2 = np.array([54333.3,9333.3,54666.7])
#points = np.linspace(point1, point2 ,num = num2)
    
#29#points = np.vstack(( [53222.22222222,44888.88888889,59111.11111111],[53000., 48000. ,60000.]))#[52.,0.03,35.]
#points = np.array(( [53222.22222222,44888.88888889,59111.11111111],),ndmin=2)

#print(cps1)
#print()
#print(cps2)
#np.savetxt("cps_in.txt",cps,fmt='%12.4f') 
#print(np.shape(cps))
#print(num_control_points_u)
#print(num_control_points_v)
#print(len(cps))
#print(len(cps1))
#print(len(cps2))

#print(ns,dist,'order_u:',order_u,'order_v:',order_v)
#open('Surface_method_ns_1.txt', 'a').write(str(ns))
#open('Surface_method_ns_1.txt', 'a').write('\n')
#.write(np.array_str(x))