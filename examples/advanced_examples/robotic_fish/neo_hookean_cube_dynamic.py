from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
import argparse

from ufl import ln, pi


parser = argparse.ArgumentParser()
parser.add_argument('--nel',dest='nel',default='16',
                    help='Number of elements')

args = parser.parse_args()
num_el = int(args.nel)
# mesh = createUnitSquareMesh(num_el)
mesh = createUnitCubeMesh(16)

fea = FEA(mesh)
# Record the function evaluations during optimization process
fea.record = True

# Add state to the PDE problem:
state_name = 'u'
state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
u_trial = Function(state_function_space)
v = TestFunction(state_function_space)
du = Function(state_function_space)
B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary
# B = Constant(domain=mesh, c=(0.0, -0.5, 0.0))  # Body force per unit volume
# T = Constant(domain=mesh, c=(0.1,  0.0, 0.0))  # Traction force on the boundary

# Time-stepping
t_start = 0.0  # start time
t_end = 15.  # end time
# t_end = 1.0  # end time
# t_steps = 100  # number of time steps
t_steps = 149  # number of time steps

t, dt = np.linspace(t_start, t_end, t_steps, retstep=True)
dt = float(dt)  # time step needs to be converted from class 'numpy.float64' to class 'float' for the .assign() method to work (see below)

# u = Function(state_function_space)
# u_bar = Function(state_function_space)
# du = Function(state_function_space)
# ddu = Function(state_function_space)
# ddu_old = Function(state_function_space)

# alpha_m = Constant(mesh, 0.2)
# alpha_f = Constant(mesh, 0.4)
# rho_inf = 1.0    # asymptotic spectral radius   # NOTE: This is unstable for MBD simulations (alebraic constraints make this unstable)
rho_inf = 0.5    # asymptotic spectral radius   # NOTE: Jiayao used this
# alpha_m = 0.2
# alpha_f = 0.4
alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
alpha_f = rho_inf/(rho_inf + 1)
gamma   = 0.5+alpha_f-alpha_m
beta    = (gamma+0.5)**2/4.


# Test and trial functions
# du = TrialFunction(state_function_space)
# u_ = TestFunction(state_function_space)
# Current (unknown) displacement
# u = Function(state_function_space, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(state_function_space)
v_old = Function(state_function_space)
a_old = Function(state_function_space)


# Generalized alpha method averaging
def linear_combination(start, stop, coefficient):
    return coefficient*start + (1-coefficient)*stop

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)

    # print('u', u[:])
    # # np.savetxt('u.txt', u[:])
    # # np.savetxt()
    # print('u_old', u_old[:])
    # print('dt_', dt_)
    # print('v_old', v_old[:])
    # print('a_old', a_old[:])
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step.""" 

    # # Get vectors (references)
    # u_vec, u0_vec  = u.vector, u_old.vector
    # v0_vec, a0_vec = v_old.vector, a_old.vector 

    u_vec = getFuncArray(u)
    u_old_vec = getFuncArray(u_old)
    v_old_vec = getFuncArray(v_old)
    a_old_vec = getFuncArray(a_old)


    # use update functions using vector arguments
    # a_new = update_a(u, u_old, v_old, a_old, ufl=True)
    # v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

    a_new_vec = update_a(u_vec, u_old_vec, v_old_vec, a_old_vec, ufl=True)
    v_new_vec = update_v(a_new_vec, u_old_vec, v_old_vec, a_old_vec, ufl=True)

    # print('a_new_vec', a_new_vec)
    # print('v_new_vec', v_new_vec)


    # Update (u_old <- u)
    # v_old.vector[:], a_old.vector[:] = v_vec, a_vec
    # u_old.vector[:] = u.vector
    setFuncArray(v_old, v_new_vec)
    setFuncArray(a_old, a_new_vec)
    setFuncArray(u_old, u_vec)

    # print('---------------------------')
    # print(v_old)
    # print(v_new)
    # print('---------------------------')
    # exit()

    # v_old.interpolate(v_new)
    # a_old.interpolate(a_new)
    # u_old.interpolate(u)

    # print('u', u_old.x.array)
    # print('v', v_old.x.array)
    # print('a', a_old.x.array)


a_new = update_a(u_trial, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

a_weighted = linear_combination(a_old, a_new, alpha_m)
v_weighted = linear_combination(v_old, v_new, alpha_f)
u_trial_weighted = linear_combination(u_old, u_trial, alpha_f)

# Define deformation gradient and Green Deformation tensor
d = u_trial.geometric_dimension()
I = Identity(d)             # Identity tensor
# F = I + grad(u_trial)       # Deformation gradient
F = I + grad(u_trial_weighted)       # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)
# Elasticity parameters
E, nu = 1.e5, 0.3
# E, nu = 10., 0.3
mu, lmbda = Constant(domain=mesh, c=E/(2*(1 + nu))), Constant(domain=mesh, c=E*nu/((1 + nu)*(1 - 2*nu)))
# Dynamics parameters
# density = 1.
density = 1.1e5
density = Constant(domain=mesh, c=density)

# Rayleigh damping coefficients
# eta_m = Constant(domain=mesh, c=0.)
eta_m = Constant(domain=mesh, c=1e-3)
eta_k = Constant(domain=mesh, c=0.)

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
# Total potential energy
Pi = psi*dx - dot(B, u_trial)*dx - dot(T, u_trial)*ds
# Compute first variation of Pi (directional derivative about u in the direction of v)
strain_energy_term = 1/alpha_f*derivative(Pi, u_trial, v) 
# NOTE: This is actually derivative of energy, so it's more like an internal forces term.


# # Compute Jacobian of F
# J = derivative(F, u_trial, du)

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
locate_BC2 = locate_dofs_geometrical(state_function_space,
                            lambda x: np.isclose(x[0], 1. ,atol=1e-6))  # Want weird displacement at x=1
locate_BC3 = locate_dofs_geometrical(state_function_space,
                            lambda x: np.isclose(x[2], 1. ,atol=1e-6))  # Want weird displacement at x=1

# locate_BC_list_1 = [locate_BC1, locate_BC2]
# locate_BC_list_1 = [locate_BC1]
locate_BC_list_1 = [locate_BC3]
# locate_BC_list_2 = [locate_BC2]
fea.add_strong_bc(ubc_1, locate_BC_list_1)
# fea.add_strong_bc(ubc_1, locate_BC_list_1, state_function_space)
# fea.add_strong_bc(ubc_2, locate_BC_list_2, state_function_space)
body_force = Function(state_function_space)
# f_d = 1.
# f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# body_force.interpolate(f)
# project(f, body_force)
# body_force.vector.set((1., 0., 0.))

# define dynamic resudual form
body_term = dot(body_force,v)*dx

intertial_term = inner(density*a_weighted,v)*dx
damping_term = eta_m*inner(density*v_weighted,v)*dx#  + eta_k*dot(density*grad(u),grad(v))*dx 
# NOTE: Not clear how to apply stiffness term in rayleigh damping with nonlinearity

residual_form = strain_energy_term - body_term + intertial_term + damping_term
# residual_form = strain_energy_term - body_term
# import ufl
# my_rhs = ufl.rhs(residual_form)

def static_solve(residual_form, u, ubc):
    # u_bar.assign(u + dt*du + 0.25*dt*dt*ddu)
    # u_bar = u + dt*du + 0.25*dt*dt*ddu

    # fea.solve(residual_form, u, ubc)
    # solveNonlinear(residual_form, u, ubc, solver="Newton", report=True, initialize=False)
    solveNonlinear(residual_form, u, ubc, solver="SNES", report=True, initialize=False)

    # np.savetxt('u_vec.txt', u.vector[:])

    # ddu_old.assign(ddu)
    # ddu.assign(4/(dt*dt)*(u - u_bar))
    # du.assign(du + 0.5*dt*(ddu + ddu_old))
    # ddu_old = ddu
    # ddu = 4/(dt*dt)*(u - u_bar)
    # du = du + 0.5*dt*(ddu + ddu_old)

path = "examples/advanced_examples/temp"
xdmf_file = XDMFFile(comm, path+"/u.xdmf", "w")
xdmf_file.write_mesh(mesh)

def dynamic_solve(residual_form, u, ubc, report=False):
    for ti in t:
        print(f't={ti}')
        static_solve(residual_form, u, ubc)

        update_fields(u, u_old, v_old, a_old)

        xdmf_file.write_function(u, ti)


# make a function that performs the time stepping (dynamic solution)
# each time step performs the static solve using fea.solve()
fea.custom_solve = dynamic_solve

input_name = 'force_input'
fea.add_input(input_name, body_force)
fea.add_state(name=state_name,
                function=u_trial,
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
f_d = 10.
f = ufl.as_vector([0,0,-f_d*1.e3]) # Body force per unit surface area
# body_force.interpolate(f)
project(f, body_force_input)

sim[input_name] = getFuncArray(body_force_input)

sim.run()