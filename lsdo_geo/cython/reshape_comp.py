import numpy as np

from openmdao.api import ExplicitComponent

class ReshapeComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('shape_input', types = list)
        self.options.declare('shape_output', types = list)

    def setup(self):
        shape_input = self.options['shape_input']
        shape_output = self.options['shape_output']

        self.array_in = array_in = 'array_in'
        self.array_out = array_out = 'array_out'

        self.add_input(array_in, shape = shape_input)
        self.add_output(array_out, shape = shape_output)

        size = np.prod(shape_input)
        arange = np.arange(size)
        self.declare_partials(array_out, array_in, val=1., rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        shape_output = self.options['shape_output']

        outputs[self.array_out] = inputs[self.array_in].reshape(shape_output)

    # def compute_partials(self, inputs, partials):
    
if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    array_in = np.arange(24).reshape(2,4,3)

    shape_input = list(array_in.shape)
    shape_output = shape_input[:-1]
    shape_output[-1] = shape_input[-1]*shape_input[-2]

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output('array_in', val = array_in)

    prob.model.add_subsystem('inputs_comp', comp, promotes = ['*'])
    prob.model.add_subsystem('reshape', ReshapeComp(
        shape_input = shape_input, shape_output = shape_output),
        promotes = ['*'])
    
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print = True)
    print(array_in)
    print(prob['array_out'])