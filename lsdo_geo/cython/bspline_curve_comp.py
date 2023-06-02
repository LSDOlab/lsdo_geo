import numpy as np
import scipy.sparse as sps

from caddee.cython.basis_matrix_curve_py import get_basis_curve_matrix

from openmdao.api import ExplicitComponent

class BSplineCurveComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_points', types = int)
        self.options.declare('order', types = int)
        self.options.declare('n_control_points', types = int)
        self.options.declare('n_t', types = int)

    def setup(self):
        n_points = self.options['n_points']
        order = self.options['order']
        n_control_points = self.options['n_control_points']
        n_t = self.options['n_t']

        self.ctrl_pts = ctrl_pts = 'ctrl_pts'
        self.pts = pts = 'pts'

        self.add_input(ctrl_pts, shape = (n_control_points, n_t))
        self.add_output(pts, shape = (n_points, n_t))

        u_vec = np.linspace(0., 1., n_points)

        data = np.zeros(n_points * order)
        row_indices = np.zeros(n_points * order, np.int32)
        col_indices = np.zeros(n_points * order, np.int32)

        get_basis_curve_matrix(order, n_control_points, 0, u_vec, n_points, data, row_indices, col_indices)
        self.basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(n_points, n_control_points))

        val = np.outer(data, np.ones(n_t)).flatten()
        rows = np.outer(n_t * row_indices, np.ones(n_t, np.int32)).flatten() + np.outer(np.ones(n_points * order, np.int32), np.arange(n_t)).flatten()
        cols = np.outer(n_t * col_indices, np.ones(n_t, np.int32)).flatten() + np.outer(np.ones(n_points * order, np.int32), np.arange(n_t)).flatten()

        basis0 = sps.csc_matrix((val, (rows, cols)), shape=(n_t*n_points, n_t*n_control_points))

        self.declare_partials(pts, ctrl_pts, val=basis0)
        # self.declare_partials(pts, ctrl_pts, val=val, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs[self.pts] = self.basis0.dot(inputs[self.ctrl_pts])

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import matplotlib.pyplot as plt

    cp_array = np.array([
    [ 9.99999786e-01,  1.05001786e-03],
    [ 9.46947154e-01,  1.01412604e-02],
    [ 9.69902937e-01,  7.13838830e-03],
    [ 8.16707526e-01,  3.17302926e-02],
    [ 6.74815046e-01,  4.90392660e-02],
    [ 5.37186135e-01,  6.20090687e-02],
    [ 3.86218473e-01,  7.03822730e-02],
    [ 2.51592704e-01,  6.79346416e-02],
    [ 2.15731492e-01,  6.46922559e-02],
    [ 1.17775273e-01,  5.36165876e-02],
    [ 7.44185581e-02,  4.20951555e-02],
    [ 4.64237698e-02,  3.49922557e-02],
    [ 3.19222973e-03,  1.53880695e-02],
    [ 1.40866665e-02,  1.70158439e-02],
    [-3.50475748e-07, -1.24487638e-07],
    ])

    
    n_points = 50
    order = 4

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output('ctrl_pts', val = cp_array)

    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])
    prob.model.add_subsystem('bspline_curve', BSplineCurveComp(
        n_points = n_points, order = order, n_control_points = cp_array.shape[0], n_t = cp_array.shape[1]),
        promotes=['*']
    )

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    pts = prob['pts']
    # print(pts)
    plt.plot(pts[:, 0], pts[:, 1], 'ok-')
    plt.plot(cp_array[:, 0], cp_array[:, 1], 'or')
    plt.title('Curve')

    plt.show()