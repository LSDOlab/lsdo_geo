import numpy as np

from openmdao.api import ExplicitComponent

class EinsumReorderComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('subscript_input', types = str)
        self.options.declare('subscript_output', types = str)
        self.options.declare('shape_input', types = list)
        self.options.declare('shape_output', types = list)

    def setup(self):
        subscript_input = self.options['subscript_input']
        subscript_output = self.options['subscript_output']
        shape_input = self.options['shape_input']
        shape_output = self.options['shape_output']

        self.array_in = array_in = 'array_in'
        self.array_out = array_out = 'array_out'

        self.add_input(array_in, shape = shape_input)
        self.add_output(array_out, shape = shape_output)


        size = np.prod(shape_input)
        rows = np.arange(size)
        cols = np.einsum(
            subscript_input + '->' + subscript_output, 
            np.arange(size).reshape(shape_input),
        ).flatten()
        self.declare_partials(array_out, array_in, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        subscript_input = self.options['subscript_input']
        subscript_output = self.options['subscript_output']
        
        outputs[self.array_out] = np.einsum(subscript_input + '->' + subscript_output, inputs[self.array_in])

    # def compute_partials(self, inputs, partials):


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    subscript_input = 'ijk'
    subscript_output = 'jki'
    shape_input = [2, 4, 3]
    shape_output = [4, 3, 2]

    array_in = np.arange(24).reshape(2,4,3)

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output('array_in', val = array_in)

    prob.model.add_subsystem('inputs_comp', comp, promotes = ['*'])
    prob.model.add_subsystem('einsum_reorder', EinsumReorderComp(
        subscript_input = 'ijk', subscript_output = 'jki', shape_input = shape_input, shape_output = shape_output),
        promotes = ['*'])
    
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print = True)
    print(array_in)
    print(prob['array_out'])