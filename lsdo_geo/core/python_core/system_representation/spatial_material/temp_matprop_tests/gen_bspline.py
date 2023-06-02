import numpy as np
import caddee
import caddee.primitives.bsplines as bs
import caddee.primitives.bsplines.bspline_functions as bsf
import caddee.caddee_core.system_representation.spatial_representation as spatial_representation
import caddee.caddee_core.system_representation.spatial_material.ls_primitive as ls_primitive

nx, ny = (20, 20)
x = np.linspace(-1,1,nx)
y = np.linspace(-1,1,ny)
xv, yv = np.meshgrid(x,y)
zv = xv**2+yv**2

spatial_data = np.dstack((xv,yv,zv))

print(spatial_data.shape)

spatial_bspline = bsf.fit_bspline(spatial_data)

print(spatial_bspline.knots_u)

# ms = spatial_representation.SpatialRepresentation()
# ms.primitives["spatial"] = spatial_bspline
# # ms.plot(plot_type="mesh", show=False)
# ms.plot(show=False)

#spatial_bspline.plot()


nu, nv = (20,20)
u = np.linspace(0,1,nu)
v = np.linspace(0,1,nv)
u_v, v_v = np.meshgrid(u,v)


# print(u_v.shape)
# print(v_v.shape)

points = np.zeros((nu,nv,3))


for i in range(0,nv):
    #print(i)
    points[i,:,:] = spatial_bspline.evaluate_points(u_v[i,:],v_v[i,:])
    #print(points[i,0,0])

# print(points)




# make level set data
r = 0.5
ls_data = np.zeros(zv.shape)

parametric_coords = spatial_bspline.project(spatial_data, return_parametric_coordinates=True)


for i in range(0,nx):
    for j in range(0,ny):
        if xv[i,j]**2 + yv[i,j]**2 <= r:
            ls_data[i,j] = 1
        else:
            ls_data[i,j] = -1

ls_data = np.expand_dims(ls_data, axis=2)


ls_bspline = bsf.fit_bspline(ls_data)

levelset = ls_primitive.LSPrimitive(primitive=ls_bspline)

ls_values = np.zeros((nu,nv,1))
for i in range(0,nv):
    #print(i)
    ls_values[i,:,:] = levelset.evaluate_points(u_v[i,:],v_v[i,:])
    #print(points[i,0,0])

#print(ls_values)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# for i in range(0,nu):
#     ax.scatter(points[i,:,0], points[i,:,1], points[i,:,2])
ax.scatter(points[:,:,0], points[:,:,1], points[:,:,2], c=ls_values)

plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # for i in range(0,nu):
# #     ax.scatter(points[i,:,0], points[i,:,1], points[i,:,2])
# ax.scatter(ls_values[:,:,0], ls_values[:,:,1], ls_values[:,:,2])

# plt.show()