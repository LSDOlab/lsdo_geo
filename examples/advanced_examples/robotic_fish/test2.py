import pyiges
from pyiges import examples

# load an example impeller
# iges = pyiges.read(examples.impeller)
iges = pyiges.read("lsdo_geo/splines/b_splines/sample_geometries/Robot_no_pump.IGS")

# print an invidiual entity (boring)
# print(iges[0])

# convert all lines to a vtk mesh and plot it
lines = iges.to_vtk(bsplines=True, surfaces=False, merge=True)
lines.plot(color='b', line_width=2)

# convert all surfaces to a vtk mesh and plot it
mesh = iges.to_vtk(bsplines=False, surfaces=True, merge=True, delta=0.05)
mesh.plot(color='b', smooth_shading=True)
# control resolution of the mesh by changing "delta"

# save this surface to file
mesh.save('mesh.ply')  # as ply
mesh.save('mesh.stl')  # as stl
mesh.save('mesh.vtk')  # as vtk