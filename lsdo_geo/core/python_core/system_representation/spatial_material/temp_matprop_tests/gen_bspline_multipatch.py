import numpy as np
import caddee
import caddee.primitives.bsplines as bs
import caddee.primitives.bsplines.bspline_functions as bsf
import caddee.caddee_core.system_representation.spatial_representation as spatial_representation
import caddee.caddee_core.system_representation.spatial_material.ls_primitive as ls_primitive

#spatial and thickness data
nx, ny = (40, 40)
x1 = np.linspace(-1,1,nx)
x2 = np.linspace(-1,0,nx)
x3 = np.linspace(0,1,nx)
y1 = np.linspace(0,2,ny)
y2 = np.linspace(-2,0,ny)
x1v, y1v = np.meshgrid(x1,y1)
z1v = (-x1v**2+1)/4
t1v = .015+.01*y1v
x2v, y2v = np.meshgrid(x2,y2)
z2v = (-x2v**2+1)/4
t2v = .015+.01*y2v
x3v, y3v = np.meshgrid(x3,y2)
z3v = (-x3v**2+1)/4
t3v = .015+.01*y3v

spatial_data1 = np.dstack((x1v,y1v,z1v))
spatial_bspline1 = bsf.fit_bspline(spatial_data1)
spatial_data2 = np.dstack((x2v,y2v,z2v))
spatial_bspline2 = bsf.fit_bspline(spatial_data2)
spatial_data3 = np.dstack((x3v,y3v,z3v))
spatial_bspline3 = bsf.fit_bspline(spatial_data3)

# ms = spatial_representation.SpatialRepresentation()
# ms.primitives["spatial"] = spatial_bspline
# # ms.plot(plot_type="mesh", show=False)
# ms.plot(show=False)

#spatial_bspline.plot()

nu, nv = (40,40)
u = np.linspace(0,1,nu)
v = np.linspace(0,1,nv)
u_v, v_v = np.meshgrid(u,v)

points1 = np.zeros((nu,nv,3))
points2 = np.zeros((nu,nv,3))
points3 = np.zeros((nu,nv,3))

for i in range(0,nv):
    points1[i,:,:] = spatial_bspline1.evaluate_points(u_v[i,:],v_v[i,:])
    points2[i,:,:] = spatial_bspline2.evaluate_points(u_v[i,:],v_v[i,:])
    points3[i,:,:] = spatial_bspline3.evaluate_points(u_v[i,:],v_v[i,:])

# make level set data
r = 0.5
ls_data1 = np.zeros(z1v.shape)
ls_data2 = np.zeros(z2v.shape)
ls_data3 = np.zeros(z3v.shape)

parametric_coords1 = spatial_bspline1.project(spatial_data1, return_parametric_coordinates=True)
parametric_coords2 = spatial_bspline2.project(spatial_data2, return_parametric_coordinates=True)
parametric_coords3 = spatial_bspline3.project(spatial_data3, return_parametric_coordinates=True)

for i in range(0,nx):
    for j in range(0,ny):
        if x1v[i,j]**2 + y1v[i,j]**2/4 <= r:
            ls_data1[i,j] = 1
        else:
            ls_data1[i,j] = -1
        if x2v[i,j]**2 + y2v[i,j]**2/4 <= r:
            ls_data2[i,j] = 1
        else:
            ls_data2[i,j] = -1
        if x3v[i,j]**2 + y3v[i,j]**2/4 <= r:
            ls_data3[i,j] = 1
        else:
            ls_data3[i,j] = -1


#ls bsplines
ls_data1 = np.expand_dims(ls_data1, axis=2)
ls_data2 = np.expand_dims(ls_data2, axis=2)
ls_data3 = np.expand_dims(ls_data3, axis=2)

n_cp = 15

ls_bspline1 = bsf.fit_bspline(ls_data1)
ls_bspline2 = bsf.fit_bspline(ls_data2)
ls_bspline3 = bsf.fit_bspline(ls_data3)

levelset1 = ls_primitive.LSPrimitive(primitive=ls_bspline1)
levelset2 = ls_primitive.LSPrimitive(primitive=ls_bspline2)
levelset3 = ls_primitive.LSPrimitive(primitive=ls_bspline3)

ls_values1 = np.zeros((nu,nv,1))
ls_values2 = np.zeros((nu,nv,1))
ls_values3 = np.zeros((nu,nv,1))

for i in range(0,nv):
    ls_values1[i,:,:] = levelset1.evaluate_points(u_v[i,:],v_v[i,:])
    ls_values2[i,:,:] = levelset2.evaluate_points(u_v[i,:],v_v[i,:])
    ls_values3[i,:,:] = levelset3.evaluate_points(u_v[i,:],v_v[i,:])

# thickness bsplines
t1v = np.expand_dims(t1v, axis=2)
t2v = np.expand_dims(t2v, axis=2)
t3v = np.expand_dims(t3v, axis=2)

t1_bspline = bsf.fit_bspline(t1v)
t2_bspline = bsf.fit_bspline(t2v)
t3_bspline = bsf.fit_bspline(t3v)

t_values1 = np.zeros((nu,nv,1))
t_values2 = np.zeros((nu,nv,1))
t_values3 = np.zeros((nu,nv,1))

for i in range(0,nv):
    t_values1[i,:,:] = t1_bspline.evaluate_points(u_v[i,:],v_v[i,:])
    t_values2[i,:,:] = t2_bspline.evaluate_points(u_v[i,:],v_v[i,:])
    t_values3[i,:,:] = t3_bspline.evaluate_points(u_v[i,:],v_v[i,:])


max_t = max([np.amax(t_values1), np.amax(t_values2), np.amax(t_values3)])

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # for i in range(0,nu):
# #     ax.scatter(points[i,:,0], points[i,:,1], points[i,:,2])
# ax.scatter(points1[:,:,0], points1[:,:,1], points1[:,:,2], c=ls_values1)
# ax.scatter(points2[:,:,0], points2[:,:,1], points2[:,:,2], c=ls_values2)
# ax.scatter(points3[:,:,0], points3[:,:,1], points3[:,:,2], c=ls_values3)

# ax.axis('equal')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# fourth dimention - colormap
# create colormap according to x-value (can use any 50x50 array)
color_dimension = np.squeeze(ls_values1*t_values1) # change to desired fourth dimension
minn, maxx = 0, max_t
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
m.set_array([])
fcolors = m.to_rgba(color_dimension)
# plot
ax.plot_surface(points1[:,:,0], points1[:,:,1], points1[:,:,2], rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)


# fourth dimention - colormap
# create colormap according to x-value (can use any 50x50 array)
color_dimension = np.squeeze(ls_values2*t_values2) # change to desired fourth dimension
#minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
m.set_array([])
fcolors = m.to_rgba(color_dimension)
# plot
ax.plot_surface(points2[:,:,0], points2[:,:,1], points2[:,:,2], rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

# fourth dimention - colormap
# create colormap according to x-value (can use any 50x50 array)
color_dimension = np.squeeze(ls_values3*t_values3) # change to desired fourth dimension
#minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
m.set_array([])
fcolors = m.to_rgba(color_dimension)
# plot
ax.plot_surface(points3[:,:,0], points3[:,:,1], points3[:,:,2], rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)


ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.colorbar(m)

#plt.show()




fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plot
ax.plot_surface(points1[:,:,0], points1[:,:,1], points1[:,:,2])
ax.plot_surface(points2[:,:,0], points2[:,:,1], points2[:,:,2])
ax.plot_surface(points3[:,:,0], points3[:,:,1], points3[:,:,2])

ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # for i in range(0,nu):
# #     ax.scatter(points[i,:,0], points[i,:,1], points[i,:,2])
# ax.scatter(ls_values[:,:,0], ls_values[:,:,1], ls_values[:,:,2])

# plt.show()