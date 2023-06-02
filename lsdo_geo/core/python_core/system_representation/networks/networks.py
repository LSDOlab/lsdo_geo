from caddee.utils.caddee_base import CADDEEBase

class power_systems_architecture(CADDEEBase):
    def initialize(self, kwargs):
        self.connections_list = []

    def connect(self, upstream_comp, upstream_vars, downstream_comp, downstream_vars_dict):
        """
        Method to connect components (and models) to specify data transfer. 

        Arguments:
        -----
            upstream_comp : (Component, Model)
                Upstream component or Model
            upstream_vars : str
                Upstream variable(s) contained in instance of VariableGroup
            downstream_comp : (Component, Model)
                Downstream component or Model
            downstream_vars : str
                Downstream variable(s) contained in instance of VariableGroup

        """
        self.connections_list.append((upstream_comp, upstream_vars, downstream_comp, downstream_vars_dict))
        return

class PowerSystemsArchitecture(power_systems_architecture): pass
