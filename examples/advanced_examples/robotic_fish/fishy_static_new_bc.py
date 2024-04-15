from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
import dolfinx.fem as dolfin_fem
import argparse

from ufl import ln, pi


with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/meshes/segment0.xdmf", "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

# mvc = MeshValueCollection("size_t", mesh, 2) 
# with XDMFFile("mf.xdmf") as infile:
#     infile.read(mvc, "name_to_read")
# mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
from dolfinx import mesh as dolfinx_mesh_module

# with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/meshes/segment0.xdmf", "r") as xdmf:
#        left_chamber_mesh = xdmf.read_mesh(name="Grid")

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/meshes/left_chamber_inner_surfaces.xdmf", "r") as xdmf:
       left_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")
# print(left_chamber_facet_tags.mesh)
# print(left_chamber_facet_tags.dim)
# print(left_chamber_facet_tags.indices)
# print(left_chamber_facet_tags.values)


mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
with XDMFFile(MPI.COMM_WORLD, "examples/advanced_examples/meshes/right_chamber_inner_surfaces.xdmf", "r") as xdmf:
       right_chamber_facet_tags = xdmf.read_meshtags(mesh, name="Grid")


# mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
# with XDMFFile(comm,'.../input_boundary.xdmf','r') as infile:
#     mt = infile.read_meshtags(mesh, "Grid")

# mf = MeshFunction("size_t", mesh, 1, 0)
# LeftOnBoundary().mark(mf, 1)

fea = FEA(mesh)
# Record the function evaluations during optimization process
fea.record = True

# Add state to the PDE problem:
state_name = 'u'
state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
# input_function_space = FunctionSpace(mesh, ('DG', 0))
input_function_space = FunctionSpace(mesh, ('CG', 2))
# test_function_space = FunctionSpace(left_chamber_facet_tags.mesh, ('DG', 0))
u = Function(state_function_space)
v = TestFunction(state_function_space)
du = Function(state_function_space)
B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary
# B = Constant(domain=mesh, c=(0.0, -0.5, 0.0))  # Body force per unit volume
# T = Constant(domain=mesh, c=(0.1,  0.0, 0.0))  # Traction force on the boundary

# pump_max_pressure = 0.
# pump_max_pressure = 1.
# pump_max_pressure = 3.e4
# pump_max_pressure = 3.5e4
# pump_max_pressure = 4.e4
# pump_max_pressure = 5.e4
pump_max_pressure = 1.e5
# pump_max_pressure = 2.e5
# pump_max_pressure = 1.e6
# pump_max_pressure = 1.e7
# pump_vacuum_pressure = -1.e4
# pump_vacuum_pressure = 0.
pump_vacuum_pressure = -3.e4

# Define deformation gradient and Green Deformation tensor
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
# F = I + grad(u)       # Deformation gradient
F = I + grad(u)       # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)
# Elasticity parameters
# E, nu = 1.e6, 0.3
# E, nu = 2.e2, 0.3
# E, nu = 10., 0.3
E = 1.5e6  # Young's modulus of dragon skin 30 silicone rubber (Pa) NOTE: Github copilot made this up (is it right?)
nu = 0.45  # Poisson's ratio of dragon skin 30 silicone rubber  NOTE: Github copilot made this up (is it right?)
mu, lmbda = Constant(domain=mesh, c=E/(2*(1 + nu))), Constant(domain=mesh, c=E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds
# Compute first variation of Pi (directional derivative about u in the direction of v)
strain_energy_term = derivative(Pi, u, v)     
# NOTE: Want derivative wrt u_weighted, but not possible in FEniCSx, so take derivative wrt u instead and multiply by 1/alpha_f to cancel chain rule
# NOTE: This is actually derivative of energy, so it's more like an internal forces term.


# # Compute Jacobian of F
# J = derivative(F, u, du)

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
locate_BC1 = locate_dofs_geometrical(state_function_space,
                            lambda x: np.isclose(x[0], -0.135 ,atol=1e-6))  # Want no displacement at x=0

# locate_BC_list_1 = [locate_BC1, locate_BC2]
locate_BC_list_1 = [locate_BC1]
# locate_BC_list_2 = [locate_BC2]
fea.add_strong_bc(ubc_1, locate_BC_list_1)    # This is the correct way to do it

pressure_input = Function(input_function_space)

left_chamber_facets = left_chamber_facet_tags.find(677)     # 677 is what GMSH GUI assigned it (see in tools --> visibility)
print(left_chamber_facets)
print(mesh.topology.dim-1)
left_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, left_chamber_facets)

right_chamber_facets = right_chamber_facet_tags.find(678)     # 678 is what GMSH GUI assigned it (see in tools --> visibility)
right_chamber_facet_dofs = locate_dofs_topological(input_function_space, mesh.topology.dim - 1, right_chamber_facets)
# NOTE: ONLY WORKS WHEN FUNCTION SPACE IS CG1, AND IT SEEMS TO BE THE WRONG DOF (according to plotting in paraview)

print(left_chamber_facet_dofs)
print(right_chamber_facet_dofs)
# exit()

pressure_input.x.array[left_chamber_facet_dofs] = pump_max_pressure
pressure_input.x.array[right_chamber_facet_dofs] = pump_vacuum_pressure

n = FacetNormal(mesh)

with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/pressure_input.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(pressure_input)

# projectLocalBasis("examples/advanced_examples/temp/")

# internal_pressure_traction = pressure_input*v*n*dx
# internal_pressure_traction = pressure_input*dot(v,n)*dx
internal_pressure_traction = pressure_input*dot(v,n)*ds
# pressure_term = pressure_input*dot(v,n)
# internal_pressure_traction = pressure_term("+")*dS + pressure_term("-")*dS

# left_chamber_pressure_term = left_chamber_pressure[0]*dot(v,n)*ds_LC
# right_chamber_pressure_term = right_chamber_pressure[0]*dot(v,n)*ds_RC

residual_form = strain_energy_term + internal_pressure_traction


# solveNonlinear(residual_form, u, ubc, solver="SNES", report=True, initialize=False)

# path = "examples/advanced_examples/temp"
# xdmf_file = XDMFFile(comm, path+"/u.xdmf", "w")
# xdmf_file.write_mesh(mesh)

# make a function that performs the time stepping (dynamic solution)
# each time step performs the static solve using fea.solve()
input_name = 'force_input'
body_force = Function(state_function_space)
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
# f_d = 10.
# f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# f_d = density_not_fenics*9.81
f_d = 0.
f = ufl.as_vector([0,0,0.]) # Body force per unit surface area
# body_force.interpolate(f)
# project(f, body_force_input)

sim[input_name] = getFuncArray(body_force_input)

sim.run()