import numpy as np
import scipy.sparse as sps

from caddee.cython.basis_matrix_surface_py import get_basis_surface_matrix
from caddee.cython.get_open_uniform_py import get_open_uniform

from openmdao.api import ExplicitComponent

class BSplineSurfaceComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('n_points_u', types = int)
        self.options.declare('n_points_v', types = int)
        self.options.declare('order_u', types = int)
        self.options.declare('order_v', types = int)
        self.options.declare('u_vec', types = np.ndarray)
        self.options.declare('v_vec', types = np.ndarray)
        self.options.declare('n_control_points_u', types = int)
        self.options.declare('n_control_points_v', types = int)
        self.options.declare('n_t', types = int)
        self.options.declare('preset', types = bool)

    def setup(self):
        n_points_u = self.options['n_points_u']
        n_points_v = self.options['n_points_v']
        order_u = self.options['order_u']
        order_v = self.options['order_v']
        u_vec = self.options['u_vec']
        v_vec = self.options['v_vec']
        n_control_points_u = self.options['n_control_points_u']
        n_control_points_v = self.options['n_control_points_v']
        n_t = self.options['n_t']
        preset = self.options['preset']

        self.ctrl_pts = ctrl_pts = 'ctrl_pts'
        self.pts = pts = 'pts'

        self.add_input(ctrl_pts, shape = (n_control_points_u*n_control_points_v, n_t))
        self.add_output(pts, shape = (n_points_u*n_points_v, n_t))

        nnz = n_points_u * n_points_v * order_u * order_v
        data = np.zeros(nnz)
        row_indices = np.zeros(nnz, np.int32)
        col_indices = np.zeros(nnz, np.int32)

        if preset == False:
            u_vec = np.einsum('i,j->ij', u_vec, np.ones(n_points_v)).flatten()
            v_vec = np.einsum('i,j->ij', np.ones(n_points_u), v_vec).flatten()

        knot_vector_u = np.zeros(n_control_points_u+order_u)
        knot_vector_v = np.zeros(n_control_points_v+order_v)

        get_open_uniform(order_u, n_control_points_u, knot_vector_u)
        get_open_uniform(order_v, n_control_points_v, knot_vector_v)

        get_basis_surface_matrix(
            order_u, n_control_points_u, 0, u_vec, knot_vector_u,
            order_v, n_control_points_v, 0, v_vec, knot_vector_v,
            n_points_u * n_points_v, data, row_indices, col_indices,
        )

        self.basis0 = sps.csc_matrix(
            (data, (row_indices, col_indices)), 
            shape=(n_points_u * n_points_v, n_control_points_u * n_control_points_v),
        )

        val = np.outer(data, np.ones(n_t)).flatten()
        rows = np.outer(n_t * row_indices, np.ones(n_t, np.int32)).flatten() + np.outer(np.ones(n_points_u * n_points_v * order_u * order_v, np.int32), np.arange(n_t)).flatten()
        cols = np.outer(n_t * col_indices, np.ones(n_t, np.int32)).flatten() + np.outer(np.ones(n_points_u * n_points_v * order_u * order_v, np.int32), np.arange(n_t)).flatten()

        basis0 = sps.csc_matrix((val, (rows, cols)), shape=(n_t*n_points_u*n_points_v, n_t*n_control_points_u*n_control_points_v))

        self.declare_partials(pts, ctrl_pts, val=basis0)

    def compute(self, inputs, outputs):
        outputs[self.pts] = self.basis0.dot(inputs[self.ctrl_pts])

    # def compute_partials(self, inputs, partials):

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import matplotlib.pyplot as plt

    order_u = 4
    order_v = 4
    num_control_points_u = 6
    num_control_points_v = 6
    num_points_u = 10
    num_points_v = 10

    cps = np.zeros((num_control_points_u, num_control_points_v, 2))
    cps[:, :, 0] = np.einsum('i,j->ij', np.linspace(0., 1., num_control_points_u), np.ones(num_control_points_v))
    cps[:, :, 1] = np.einsum('i,j->ij', np.ones(num_control_points_u), np.linspace(0., 1., num_control_points_v))
    cps = cps.reshape((num_control_points_u * num_control_points_v, 2))

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output('ctrl_pts', val = cps)

    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])
    prob.model.add_subsystem('bspline_surface', BSplineSurfaceComp(
        n_points_u = num_points_u, n_points_v = num_points_v, order_u = order_u, order_v = order_v,
        u_vec = np.linspace(0., 1., num_points_u), v_vec = np.linspace(0., 1., num_points_v),
        # u_vec = np.array([0]), v_vec = np.linspace(0., 1., num_points_v),
        # u_vec = np.array([0]), v_vec = np.array([1]),
        n_control_points_u = num_control_points_u, n_control_points_v = num_control_points_v, n_t = 2, preset=False),
        promotes=['*']
    )

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    pts = prob['pts']

    plt.plot(pts[:, 0], pts[:, 1], 'ok')
    plt.plot(cps[:, 0], cps[:, 1], 'or')
    # plt.show()