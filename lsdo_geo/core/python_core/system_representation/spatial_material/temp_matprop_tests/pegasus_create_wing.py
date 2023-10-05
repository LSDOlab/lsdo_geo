import pickle
import numpy as np
import caddee
import lsdo_geo.primitives.b_splines as bs
import lsdo_geo.primitives.b_splines.b_spline_functions as bsf
from lsdo_geo.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
from lsdo_geo.caddee_core.system_representation.spatial_representation import SpatialRepresentation
import lsdo_geo.caddee_core.system_representation.spatial_material.ls_primitive as ls_primitive


ms = SpatialRepresentation()

with open('pegasus_cross_section_data.p', 'rb') as f:
    data = pickle.load(f)

ribPoints = data['AirfoilRibsPoints']
# print(ribPoints.shape)

ribPoints = np.swapaxes(ribPoints, 1, 2)
# print(ribPoints.shape)

n_cp = (10,2)
order = (2,)

for i in range(ribPoints.shape[1]-1):
    pointst = np.zeros((10,2,3))
    pointst[:,0,:] = ribPoints[0:10,i,:]
    pointst[:,1,:] = ribPoints[0:10,i+1,:]

    pointsb = np.zeros((10,2,3))
    pointsb[:,0,:] = ribPoints[10:20,i,:]
    pointsb[:,1,:] = ribPoints[10:20,i+1,:]

    pointsf = np.zeros((2,2,3))
    pointsf[0,:,:] = ribPoints[0,(i,i+1),:]
    pointsf[1,:,:] = ribPoints[-1,(i,i+1),:]

    pointsba = np.zeros((2,2,3))
    pointsba[0,:,:] = ribPoints[9,(i,i+1),:]
    pointsba[1,:,:] = ribPoints[10,(i,i+1),:]

    paneltbs = bsf.fit_b_spline(pointst, num_coefficients = n_cp, order=order)
    panelt = SystemPrimitive('panel' + str(i) + 't', paneltbs)
    ms.primitives[panelt.name] = panelt

    panelbbs = bsf.fit_b_spline(pointsb, num_coefficients = n_cp, order=order)
    panelb = SystemPrimitive('panel' + str(i) + 'b', panelbbs)
    ms.primitives[panelb.name] = panelb

    sparfbs = bsf.fit_b_spline(pointsf, num_coefficients = (2,2), order = (2,))
    sparf = SystemPrimitive('spar' + str(i) + 'f', sparfbs)
    ms.primitives[sparf.name] = sparf

    sparbbs = bsf.fit_b_spline(pointsba, num_coefficients = (2,2), order = (2,))
    sparb = SystemPrimitive('spar' + str(i) + 'b', sparbbs)
    ms.primitives[sparb.name] = sparb

n_cp = (10,2)
order = (2,)
for i in range(ribPoints.shape[1]):
    points = np.zeros((10,2,3))
    points[:,0,:] = ribPoints[0:10,i,:]
    points[:,1,:] = np.flip(ribPoints[10:20,i,:],axis=0)
    # if i == 0:
    #     print(points.shape)
    #     print(points)
    
    ribbs = bsf.fit_b_spline(points, num_coefficients = n_cp, order=order)
    rib = SystemPrimitive('rib' + str(i), ribbs)
    ms.primitives[rib.name] = rib



## Thickness info:
# Top surface: 0.226623
# Bottom surface: 0.291276
# Front spar: 0.177914
# Back spar: 0.264536
# Ribs: 0.14852


ms.plot()
# ms.plot(plot_types='wireframe')

ms.write_iges('pegasus_wing.iges')

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# n = 1
# x = ribPoints[0:10,0,n]
# y = ribPoints[0:10,1,n]
# z = ribPoints[0:10,2,n]

# xl = ribPoints[10:20,0,n]
# yl = ribPoints[10:20,1,n]
# zl = ribPoints[10:20,2,n]


# ax.scatter(x,y,z)
# ax.scatter(xl,yl,zl)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

