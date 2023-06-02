import numpy as np

from openmdao.api import ExplicitComponent

class LinearDistrComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_points', types = int)
        self.options.declare('n_control_points', types = int)
        self.options.declare('n_t', types = int)
    
    def setup(self):
        n_points = self.options['n_points']
        n_control_points = self.options['n_control_points']
        n_t = self.options['n_t']

        self.ctrl_pts = ctrl_pts = 'ctrl_pts'
        self.pts = pts = 'pts'

        self.add_input(ctrl_pts, shape = (n_control_points, n_t))
        self.add_output(pts, shape = (n_points, n_t))

        self.declare_partials(pts, ctrl_pts)
    
    def compute(self, inputs, outputs):
        n_points = self.options['n_points']

        start = inputs['ctrl_pts'][0,:]
        end = inputs['ctrl_pts'][-1,:]

        outputs['pts'] = np.array([np.linspace(i,j,n_points) for i,j in zip(start,end)]).T

    # def compute_partials(self, inputs, partials):

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    n_t = 5
    n_points = 10
    n_control_points = 2

    prob = Problem()
    comp = IndepVarComp()

    # ctrl_pts = np.array([[0],[1],[3]])
    ctrl_pts = np.array([[0,1,2,3,4,], [1,4,5,7,8]])
    comp.add_output('ctrl_pts', ctrl_pts)

    prob.model.add_subsystem('indeps_comp', comp, promotes=['*'])

    comp = LinearDistrComp(n_points=n_points, n_control_points=n_control_points, n_t=n_t)
    prob.model.add_subsystem('LinearDistrComp', comp, promotes=['*'])
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['pts'])
    # prob.check_partials(compact_print=True)