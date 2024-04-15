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

fea = FEA(mesh)
# Record the function evaluations during optimization process
fea.record = True

# Add state to the PDE problem:
state_name = 'u'
state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
input_function_space = FunctionSpace(mesh, ('DG', 0))
u = Function(state_function_space)
v = TestFunction(state_function_space)
du = Function(state_function_space)
B = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Body force per unit volume
T = Constant(domain=mesh, c=(0.0, 0.0, 0.0))  # Traction force on the boundary
# B = Constant(domain=mesh, c=(0.0, -0.5, 0.0))  # Body force per unit volume
# T = Constant(domain=mesh, c=(0.1,  0.0, 0.0))  # Traction force on the boundary

# Time-stepping
t_start = 0.0  # start time
t_end = 0.5  # end time
# t_end = 4.  # end time
# t_end = 2.  # end time
# t_end = 1.0  # end time       # Good amount of time for one stroke
t_steps = 50  # number of time steps
# t_steps = 100  # number of time steps
# t_steps = 101  # number of time steps
# t_steps = 201  # number of time steps
# t_steps = 301  # number of time steps
# t_steps = 401  # number of time steps
# t_steps = 801  # number of time steps
# t_steps = 51  # number of time steps
# t_steps = 3  # number of time steps

t, dt = np.linspace(t_start, t_end, t_steps, retstep=True)
dt = float(dt)  # time step needs to be converted from class 'numpy.float64' to class 'float' for the .assign() method to work (see below)

p0 = 0.
pump_max_pressure = 0.
# pump_max_pressure = 3.e4
# pump_max_pressure = 3.5e4
# pump_max_pressure = 4.e4
# pump_max_pressure = 5.e4
# pump_max_pressure = 1.e5
# pump_max_pressure = 2.e5
# pump_max_pressure = 1.e6
# pump_vacuum_pressure = -1.e4
# pump_vacuum_pressure = 0.
pump_vacuum_pressure = -3.e4
actuation_frequency = 0.5 # Hz
stroke_period = 1./actuation_frequency/2    # 2 strokes per cycle
num_strokes = int(actuation_frequency*t_end)*2+1   # 2 strokes per cycle
# time_constant = 8*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.
time_constant = 6*actuation_frequency    # in reality this would be a constant from RC circuit analysis, but I want it to be nice for me.

# u = Function(state_function_space)
# u_bar = Function(state_function_space)
# du = Function(state_function_space)
# ddu = Function(state_function_space)
# ddu_old = Function(state_function_space)

# alpha_m = Constant(mesh, 0.2)
# alpha_f = Constant(mesh, 0.4)
rho_inf = 1.0    # asymptotic spectral radius   # NOTE: This is unstable for MBD simulations (alebraic constraints make this unstable)
# rho_inf = 0.5    # asymptotic spectral radius   # NOTE: Jiayao used this
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


a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

a_weighted = linear_combination(a_old, a_new, alpha_m)
v_weighted = linear_combination(v_old, v_new, alpha_f)
u_weighted = linear_combination(u_old, u, alpha_f)

# Define deformation gradient and Green Deformation tensor
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
# F = I + grad(u)       # Deformation gradient
F = I + grad(u_weighted)       # Deformation gradient
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
# Dynamics parameters
density_not_fenics = 1080.  # density of dragon skin 30 silicone rubber (kg/m^3) NOTE: From the data sheet
# density = 1.1e6
density = Constant(domain=mesh, c=density_not_fenics)

# Rayleigh damping coefficients
eta_m = Constant(domain=mesh, c=0.)
# eta_m = Constant(domain=mesh, c=5.e0)
# eta_m = Constant(domain=mesh, c=8.e0)   #Seemed to match visually with gravity, but too much oscillation in pressurized setting.
eta_m = Constant(domain=mesh, c=16.e0)    # Previous was calculated based on frequency, but stiffness term is missing, so multiply by 2.
# eta_k = Constant(domain=mesh, c=0.)

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
# Total potential energy
Pi = psi*dx - dot(B, u_weighted)*dx - dot(T, u_weighted)*ds
# Compute first variation of Pi (directional derivative about u in the direction of v)
strain_energy_term = 1/alpha_f*derivative(Pi, u, v)     
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
# ubc.vector.set(0.0)
# ubc_1.vector.set(("0.0", "0.0", "0.0"))
# ubc_2.vector.set(("scale*0.0",
#                 "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
#                 "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
#                 scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3)
locate_BC1 = locate_dofs_geometrical(state_function_space,
                            # lambda x: np.isclose(x[0], -0.335 ,atol=1e-6))  # Want no displacement at x=0
                            lambda x: np.isclose(x[0], -0.135 ,atol=1e-6))  # Want no displacement at x=0

# locate_BC_list_1 = [locate_BC1, locate_BC2]
locate_BC_list_1 = [locate_BC1]
# locate_BC_list_2 = [locate_BC2]
fea.add_strong_bc(ubc_1, locate_BC_list_1)    # This is the correct way to do it
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

pressure_input = Function(input_function_space)
# locate_left_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] > 0)
# locate_right_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] < 0)

# fish_width = 0.05 # got from paraview
# fish_height = 0.08333333333333333     # got from paraview
fish_width = 0.0465     # played around with to get everything except the outer boundary of the fish skin
fish_height = 0.075     # played around with to get everything except the outer boundary of the fish skin
# locate_left_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] > 0 and (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))     # idea: try x: x[2] > 0 and x[2] < some ellipse
# locate_right_chamber = locate_dofs_geometrical(input_function_space, lambda x: x[2] < 0 and (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))     # idea: try x: x[2] > 0 and x[2] < some ellipse)

locate_left_chamber = locate_dofs_geometrical(input_function_space, 
                                            #   lambda x: #np.logical_and(np.logical_and(x[0] <= -0.15, x[0] >= -0.32),
                                              lambda x: np.logical_and(np.logical_and(x[0] <= -0.148, x[0] >= -0.322),
                                                  np.logical_and(x[2] > 0, 
                                                                       (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))))     # idea: try x: x[2] > 0 and x[2] < some ellipse
locate_right_chamber = locate_dofs_geometrical(input_function_space, 
                                            #    lambda x: #np.logical_and(np.logical_and(x[0] <= -0.15, x[0] >= -0.32),
                                              lambda x: np.logical_and(np.logical_and(x[0] <= -0.148, x[0] >= -0.322),
                                                   np.logical_and(x[2] < 0, 
                                                                        (x[1]**2/fish_height**2 + x[2]**2/fish_width**2 < 1.))))     # idea: try x: x[2] > 0 and x[2] < some ellipse)
# NOTE: Alternative idea is to dot the traction term with [1,0,0] to only apply the x-component of normal tractions.

# def left_chamber_pressure(t):
#     if t < t_end/2:
#         return -(1.e4 - 1.e4*np.exp(-2*t/(t_end/2)))
#     else:
#         return -(-1.e2 + (1.e4+1.e2)*np.exp(-2*(t-t_end/2)/(t_end/2)) - 1.e4*np.exp(-2*t/(t_end/2)))

# def right_chamber_pressure(t):
#     if t < t_end/2:
#         return -(-1.e2 + 1.e2*np.exp(-2*t/(t_end/2)))
#     else:
#         return -(1.e4 - (1.e4+1.e2)*np.exp(-2*(t-t_end/2)/(t_end/2))  + 1.e2*np.exp(-2*t/(t_end/2)))


def compute_chamber_pressure_function(t, pressure_inputs, time_constant, p0, evaluation_t):
    if len(t) != len(pressure_inputs):
        raise ValueError('t and pressure_inputs must have the same length')
    
    # total_t = np.zeros((len(t),len(evaluation_t)))
    chamber_pressure = np.zeros((len(evaluation_t),))
    index = 0
    for i in range(len(t)):   # For each stroke
        # if i < len(t)-1:
        #     evaluation_t = np.linspace(t[i], t[i+1], num_evaluation_points)
        # else:
        #     evaluation_t = np.linspace(t[i], t_end, num_evaluation_points)
        # total_t[i] = evaluation_t
        j = 0
        if i < len(t)-1:
            while evaluation_t[index] < t[i+1]:
                if i == 0:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index], p0, pressure_inputs[i], time_constant)
                else:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index] - t[i],
                                                                        chamber_pressure[index-j-1],
                                                                        pressure_inputs[i],
                                                                        time_constant)
                j += 1
                index += 1
        else:
            while index < len(evaluation_t):
                if i == 0:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index], p0, pressure_inputs[i], time_constant)
                else:
                    chamber_pressure[index] = stroke_pressure(evaluation_t[index] - t[i],
                                                                        chamber_pressure[index-j-1],
                                                                        pressure_inputs[i],
                                                                        time_constant)
                index += 1
                j += 1
        i += 1
            
    return chamber_pressure



def stroke_pressure(t, initial_pressure, final_pressure, time_constant):
    return final_pressure \
        - (final_pressure - initial_pressure)*np.exp(-time_constant*t)\
        # + initial_pressure
    # final pressure is what it should go to. There is a decaying exponential term that goes to zero. Initial pressure is to ensure continuity.
    # t is the time WITHIN THE STROKE. This means that the time previous to the stroke is subtracted off before being passed into this function.

left_chamber_inputs = []
right_chamber_inputs = []
for stroke_index in range(num_strokes):
    if stroke_index % 2 == 0:
        left_chamber_inputs.append(pump_max_pressure)
        right_chamber_inputs.append(pump_vacuum_pressure)
    else:
        left_chamber_inputs.append(pump_vacuum_pressure)
        right_chamber_inputs.append(pump_max_pressure)
# left_chamber_inputs = [pump_max_pressure, pump_vacuum_pressure, pump_max_pressure, pump_vacuum_pressure, pump_max_pressure]
# right_chamber_inputs = [pump_vacuum_pressure, pump_max_pressure, pump_vacuum_pressure, pump_max_pressure, pump_vacuum_pressure]

t_pressure_inputs = np.linspace(0, t_end, int(num_strokes))
t_pressure_inputs[1:] = t_pressure_inputs[1:] - stroke_period/2
left_chamber_pressure = compute_chamber_pressure_function(t_pressure_inputs, left_chamber_inputs, time_constant, p0, t)
right_chamber_pressure = compute_chamber_pressure_function(t_pressure_inputs, right_chamber_inputs, time_constant, p0, t)

# import matplotlib.pyplot as plt
# t_test = np.linspace(0, t_end, 100)
# plt.plot(t_test, left_chamber_pressure(t_test))
# plt.plot(t_test, right_chamber_pressure(t_test))
# plt.show()

# import matplotlib.pyplot as plt
# t_test = np.linspace(0, t_end, 100)
# left_chamber_pressure_vector = np.vectorize(left_chamber_pressure)
# right_chamber_pressure_vector = np.vectorize(right_chamber_pressure)
# plt.plot(t_test, left_chamber_pressure_vector(t_test))
# plt.plot(t_test, right_chamber_pressure_vector(t_test))
# plt.show()

# plt.plot(t_test, left_chamber_pressure(t_test))
# plt.plot(t_test, right_chamber_pressure(t_test))
# plt.show()

# pressure_input.x.array[:] = 1.e6
# pressure_input.x.array[:] = 1.e3
# pressure_input.x.array[locate_left_chamber] = -4.e3
# pressure_input.x.array[locate_left_chamber] = -5.e3
# pressure_input.x.array[locate_left_chamber] = -1.e4
# pressure_input.x.array[locate_right_chamber] = 0.
pressure_input.x.array[locate_left_chamber] = left_chamber_pressure[0]
pressure_input.x.array[locate_right_chamber] = right_chamber_pressure[0]
# pressure_input.x.array[locate_left_chamber] = left_chamber_pressure[-1]
# pressure_input.x.array[locate_right_chamber] = right_chamber_pressure[-1]

# n = ufl.CellNormal(mesh)
n = FacetNormal(mesh)

# def projectLocalBasis(PATH):
#     VT = VectorFunctionSpace(mesh, ("CG", 1), dim=3)
#         #local frame looks good when exported one at a time, paraview doing something funny when all 3 basis vectors included
#     with XDMFFile(MPI.COMM_SELF, PATH+"a0.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         e0 = Function(VT, name="e0")
#         # e0.interpolate(ufl.Expression(n, VT.element.interpolation_points()))
#         e0.interpolate(dolfin_fem.Expression(n, VT.element.interpolation_points()))
#         xdmf.write_function(e0)

with XDMFFile(MPI.COMM_SELF, "examples/advanced_examples/temp/pressure_input.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(pressure_input)
    # exit()

# projectLocalBasis("examples/advanced_examples/temp/")

# internal_pressure_traction = pressure_input*v*n*dx
# internal_pressure_traction = pressure_input*dot(v,n)*dx
# internal_pressure_traction = pressure_input*dot(v,n)*ds
pressure_term = pressure_input*dot(v,n)
internal_pressure_traction = pressure_term("+")*dS + pressure_term("-")*dS

# residual_form = strain_energy_term - body_term + intertial_term + damping_term - internal_pressure_traction
residual_form = strain_energy_term + intertial_term + damping_term - internal_pressure_traction
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
    time_step_index = 0
    for ti in t:
        print(f't={ti}')
        print(f'time_step={time_step_index+1}/{t_steps}')
        pressure_input.x.array[locate_left_chamber] = left_chamber_pressure[time_step_index]
        pressure_input.x.array[locate_right_chamber] = right_chamber_pressure[time_step_index]
        static_solve(residual_form, u, ubc)

        update_fields(u, u_old, v_old, a_old)

        xdmf_file.write_function(u, ti)
        time_step_index += 1


# make a function that performs the time stepping (dynamic solution)
# each time step performs the static solve using fea.solve()
fea.custom_solve = dynamic_solve

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
# f_d = 10.
# f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# f_d = density_not_fenics*9.81
f_d = 0.
f = ufl.as_vector([0,0,-f_d]) # Body force per unit surface area
# body_force.interpolate(f)
# project(f, body_force_input)

sim[input_name] = getFuncArray(body_force_input)

sim.run()