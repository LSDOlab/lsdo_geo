import numpy as np

from openmdao.api import ExplicitComponent

class EinsumExpansionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('subscript_input', types = str)
        self.options.declare('subscript_output', types = str)
        self.options.declare('shape_output', types = list)

    def setup(self):
        subscript_input = self.options['subscript_input']
        subscript_output = self.options['subscript_output']

        shape_output = self.options['shape_output']

        subscript_ones = subscript_output
        for c in subscript_input:
            subscript_ones = subscript_ones.replace(c, '')

        self.subscript_ones = subscript_ones

        shape_ones = []
        for c in subscript_ones:
            index = subscript_output.index(c)
            shape_ones.append(shape_output[index])

        shape_input = []
        for c in subscript_input:
            index = subscript_output.index(c)
            shape_input.append(shape_output[index])

        self.ones = ones = np.ones(shape_ones)

        self.array_in = array_in = 'array_in'
        self.array_out = array_out = 'array_out'

        self.add_input(array_in, shape = shape_input)
        self.add_output(array_out, shape = shape_output)

        size_input = np.prod(shape_input)
        size_output = np.prod(shape_output)

        rows = np.arange(size_output)
        cols = np.einsum(
            '{},{}->{}'.format(subscript_input, subscript_ones, subscript_output),
            np.arange(size_input).reshape(shape_input), 
            ones,
        ).flatten()
        self.declare_partials(array_out, array_in, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        subscript_input = self.options['subscript_input']
        subscript_output = self.options['subscript_output']
        subscript_ones = self.subscript_ones

        outputs[self.array_out] = np.einsum(
            '{},{}->{}'.format(subscript_input, subscript_ones, subscript_output),
            inputs[self.array_in], 
            self.ones,
        )

        # outputs[self.array_out] = np.einsum(, np.ones(inputs[self.array_in].shape[0]), inputs[self.array_in])

    # def compute_partials(self, inputs, partials):

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    subscript_input = 'ij'
    subscript_output = 'kij'
    shape_output = [4, 3, 2]

    array_in = np.arange(6).reshape(3, 2)

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output('array_in', val = array_in)

    prob.model.add_subsystem('inputs_comp', comp, promotes = ['*'])
    prob.model.add_subsystem('einsum_expansion_comp', EinsumExpansionComp(
        subscript_input = subscript_input, subscript_output = subscript_output, shape_output = shape_output),
        promotes = ['*'])
    
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print = True)
    print(array_in)
    print(prob['array_out'])

