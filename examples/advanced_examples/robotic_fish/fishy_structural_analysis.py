from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
import argparse

from ufl import ln, pi

'''
NOTE: LOOK INTO MESHIO
'''
# from rhino3dm import *
# center = Point3d(1,2,3)
# arc = Arc(center, 10, 1)
# nc = arc.ToNurbsCurve()
# start = nc.PointAtStart
# print(start)

# from lsdo_geo.utils.STEPfileparser.main import convert_step_to_json

# convert_step_to_json('lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules.STEP')


# from steputils import p21

# my_file = p21.readfile('lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules.STEP')
# # my_file = p21.readfile('lsdo_geo/splines/b_splines/sample_geometries/lift_plus_cruise_final.stp')
# print(my_file)

# names_list = {}
# for simple_or_complex_entity in my_file.data[0].instances.values():
#     try:
#         if simple_or_complex_entity.entity.name not in names_list:
#             # names_list.append(simple_or_complex_entity.entity.name)
#             names_list[simple_or_complex_entity.entity.name] = 1
#         else:
#             names_list[simple_or_complex_entity.entity.name] += 1
#     except:
#         for entity in simple_or_complex_entity.entities:
#             if entity.name not in names_list:
#                 # names_list.append(entity.name)
#                 names_list[entity.name] = 1
#             else:
#                 names_list[entity.name] += 1


# import geomdl
# from geomdl import exchange

# exchange.import_vmesh('lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules.STEP')


# import cadquery as cq

# result = cq.importers.importStep("lsdo_geo/splines/b_splines/sample_geometries/fish_gmsh.stp")



# exit()

import lsdo_geo

# geometry = lsdo_geo.import_geometry('lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules.STEP', parallelize=False)
# geometry = lsdo_geo.import_geometry('lsdo_geo/splines/b_splines/sample_geometries/fish_gmsh.stp', parallelize=False)
# geometry.refit(parallelize=True)


# parser = argparse.ArgumentParser()
# parser.add_argument('--nel',dest='nel',default='16',
#                     help='Number of elements')

# args = parser.parse_args()
# num_el = int(args.nel)
# # mesh = createUnitSquareMesh(num_el)
# mesh = createUnitCubeMesh(num_el)

with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/meshes/segment0.xdmf", "r") as xdmf:
# with XDMFFile(MPI.COMM_WORLD, "lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules6.xdmf", "r") as xdmf:
# with XDMFFile(MPI.COMM_WORLD, "lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules6.xdmf", "r") as xdmf:
# with XDMFFile(MPI.COMM_WORLD, "lsdo_geo/splines/b_splines/sample_geometries/cube_mesh.xdmf", "r") as xdmf:
# with XDMFFile(MPI.COMM_WORLD, "lsdo_geo/splines/b_splines/sample_geometries/pneunet1.xdmf", "r") as xdmf:
# with XDMFFile(MPI.COMM_WORLD, "lsdo_geo/splines/b_splines/sample_geometries/simplified_fishy.xdmf", "r") as xdmf:
# with XDMFFile(MPI.COMM_WORLD, "lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules1.xdmf", "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

fea = FEA(mesh)
# Record the function evaluations during optimization process
fea.record = True

# Add state to the PDE problem:
state_name = 'u'
state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
u = Function(state_function_space)
v = TestFunction(state_function_space)
du = Function(state_function_space)
B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary
# B = Constant(domain=mesh, c=(0.0, -0.5, 0.0))  # Body force per unit volume
# T = Constant(domain=mesh, c=(0.1,  0.0, 0.0))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)
# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(domain=mesh, c=E/(2*(1 + nu))), Constant(domain=mesh, c=E*nu/((1 + nu)*(1 - 2*nu)))
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds
# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)
# Compute Jacobian of F
J = derivative(F, u, du)

# output_name = 'dPE_du'
# output_form = F         # residual is F == 0 for equilibrium (minimization of potential energy)

'''
3. Define the boundary conditions
'''
############ Strongly enforced boundary conditions #############
ubc_1 = Function(state_function_space)
ubc_2 = Function(state_function_space)
ubc_1.vector.set(0.)
ubc_2.vector.set(0.)
# ubc.vector.set(0.0)
# ubc_1.vector.set(("0.0", "0.0", "0.0"))
# ubc_2.vector.set(("scale*0.0",
#                 "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
#                 "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
#                 scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3)
locate_BC1 = locate_dofs_geometrical(state_function_space,
                            lambda x: np.isclose(x[0], 0. ,atol=1e-6))  # Want no displacement at x=0
# locate_BC2 = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[0], 1. ,atol=1e-6))  # Want weird displacement at x=1
locate_BC1 = locate_dofs_geometrical(state_function_space,
                            lambda x: np.isclose(x[0], -0.335 ,atol=1e-6))  # Want no displacement at x=0
# locate_BC1 = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[0], 0.0127 ,atol=1e-6))  # Want no displacement at x=0
# locate_BC2 = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[0], -0.0127 ,atol=1e-6))  # Want weird displacement at x=1
# locate_BC1 = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[0], 5.7 ,atol=1e-6))  # Want no displacement at x=0
# locate_BC1 = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[2], 0.09751 ,atol=1e-3))  # Want no displacement at x=0
# locate_BC1 = locate_dofs_geometrical(state_function_space,
#                             lambda x: np.isclose(x[0], 0.2308 ,atol=1.e-3))  # Want no displacement at x=0
print(locate_BC1)


# locate_BC_list_1 = [locate_BC1, locate_BC2]
locate_BC_list_1 = [locate_BC1]
# locate_BC_list_2 = [locate_BC2]
fea.add_strong_bc(ubc_1, locate_BC_list_1)
# fea.add_strong_bc(ubc_1, locate_BC_list_1, state_function_space)
# fea.add_strong_bc(ubc_2, locate_BC_list_2, state_function_space)
body_force = Function(state_function_space)
f_d = -1.
f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# f = ufl.as_vector([0,-f_d,0]) # Body force per unit surface area
# f = ufl.as_vector([-f_d,0,0]) # Body force per unit surface area
# body_force.interpolate(f)
# project(f, body_force)
# body_force.vector.set((1., 0., 0.))
body_term = dot(body_force,v)*dx
residual_form = F - body_term

input_name = 'force_input'
fea.add_input(input_name, body_force)
fea.add_state(name=state_name,
                function=u,
                residual_form=residual_form,
                arguments=[input_name])
# fea.add_output(name=output_name,
#                 type='scalar',
#                 form=output_form,
#                 arguments=[input_name,state_name])


'''
4. Set up the CSDL model
'''
fea.PDE_SOLVER = 'Newton'
# fea.REPORT = True
fea_model = FEAModel(fea=[fea])
fea_model.create_input("{}".format(input_name),
                            shape=fea.inputs_dict[input_name]['shape'],
                            val=0.1*np.ones(fea.inputs_dict[input_name]['shape']) * 0.86)

# fea_model.connect('f','u_state_model.f')
# fea_model.connect('f','l2_functional_output_model.f')
# fea_model.connect('u_state_model.u','l2_functional_output_model.u')

# fea_model.add_design_variable(input_name)
# fea_model.add_objective(output_name, scaler=1e5)

from python_csdl_backend import Simulator
sim = Simulator(fea_model)
# sim = om_simulator(fea_model)
########### Test the forward solve ##############
body_force_input = Function(state_function_space)
f_d = -1.
f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# f = ufl.as_vector([0,-f_d,0]) # Body force per unit surface area
# f = ufl.as_vector([-f_d,0,0]) # Body force per unit surface area
# body_force.interpolate(f)
project(f, body_force_input)

sim[input_name] = getFuncArray(body_force_input)

sim.run()


# # Create mesh and define function space
# mesh = UnitCubeMesh(24, 16, 16)
# V = VectorFunctionSpace(mesh, "Lagrange", 1)

# # Mark boundary subdomians
# left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
# right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

# # Define Dirichlet boundary (x = 0 or x = 1)
# c = Expression(("0.0", "0.0", "0.0"))
# r = Expression(("scale*0.0",
#                 "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
#                 "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
#                 scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3)

# bcl = DirichletBC(V, c, left)
# bcr = DirichletBC(V, r, right)
# bcs = [bcl, bcr]

# # Define functions
# du = TrialFunction(V)            # Incremental displacement
# v  = TestFunction(V)             # Test function
# u  = Function(V)                 # Displacement from previous iteration
# B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
# T  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary

# # Kinematics
# d = u.geometric_dimension()
# I = Identity(d)             # Identity tensor
# F = I + grad(u)             # Deformation gradient
# C = F.T*F                   # Right Cauchy-Green tensor

# # Invariants of deformation tensors
# Ic = tr(C)
# J  = det(F)

# # Elasticity parameters
# E, nu = 10.0, 0.3
# mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# # Stored strain energy density (compressible neo-Hookean model)
# psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# # Total potential energy
# Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# # Compute first variation of Pi (directional derivative about u in the direction of v)
# F = derivative(Pi, u, v)

# # Compute Jacobian of F
# J = derivative(F, u, du)

# # Solve variational problem
# solve(F == 0, u, bcs, J=J,
#       form_compiler_parameters=ffc_options)

# # Save solution in VTK format
# file = File("displacement.pvd");
# file << u;

# # Plot and hold solution
# plot(u, mode = "displacement", interactive = True)