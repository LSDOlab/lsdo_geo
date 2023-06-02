import numpy as np
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from caddee.utils.caddee_base import CADDEEBase
# from caddee.caddee_core.system_representation.geometry import Geometry
from caddee.caddee_core.system_representation.spatial_representation import SpatialRepresentation

class Component(CADDEEBase):
    '''
    Groups a component of the system.

    Parameters
    -----------
    name : str
        The name of the component

    spatial_representation : SpatialRepresentation, optional
        The mechanical structure that this component is grouping

    primitive_names : list of strings
        The names of the mechanical structure primitives to be included in the component
    '''
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        # self.parameters.declare('geometry', types=Geometry)
        # self.parameters.declare('geometry_primitive_names', default=[], allow_none=True, types=list)
        self.parameters.declare('component_vars', default=[''], types=list)
        self.parameters.declare('spatial_representation', default=None, types=SpatialRepresentation, allow_none=True)
        self.parameters.declare('primitive_names', default=[], allow_none=True, types=list)

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.spatial_representation = self.parameters['spatial_representation']
        self.primitive_names = self.parameters['primitive_names']
        # self.primitives = self.spatial_representation.get_primitives(search_names=self.primitive_names)

    def get_primitives(self):
        return self.spatial_representation.get_primitives(search_names=self.primitive_names)


    def get_geometry_primitives(self):
        return self.spatial_representation.get_geometry_primitives(search_names=self.primitive_names)
    
    def project(self, points:np.ndarray, direction:np.ndarray=None, grid_search_n:int=25,
                max_iterations=100, offset:np.ndarray=None, plot:bool=False):
        '''
        Projects points onto the system.

        Parameters
        -----------
        points : {np.ndarray, am.MappedArray}
            The points to be projected onto the system.
        targets : list, optional
            The list of primitives to project onto.
        direction : {np.ndarray, am.MappedArray}, optional
            An axis for perfoming projection along an axis. The projection will return the closest point to the axis.
        grid_search_n : int, optional
            The resolution of the grid search prior to the Newton iteration for solving the optimization problem.
        max_iterations : int, optional
            The maximum number of iterations for the Newton iteration.
        properties : list
            The list of properties to be returned (in order) {geometry, parametric_coordinates, (material_name, array_of_properties),...}
        offset : np.ndarray
            An offset to apply after the parametric evaluation of the projection. TODO Fix offset!!
        plot : bool
            A boolean on whether or not to plot the projection result.
        '''
        projected_points = self.spatial_representation.project(points=points, targets=self.primitive_names.copy(), 
                direction=direction, grid_search_n=grid_search_n, max_iterations=max_iterations, offset=offset, plot=plot)

        return projected_points

    def plot(self):
        self.spatial_representation.plot(primitives=self.primitive_names)

    # def set_module_input(self, name, val, design_condition, csdl_var=True, units='', computed_upstream=False, dv_flag=False, lower=None, upper=None, scaler=None):
        
    #     """
    #     Method for specifying operational variables on components.
    #     Same method as set_module_input() in CADDEEBase, except here 
    #     there is an additional, non-optional keyword 'design_condition='.
    #     This is because when variables are added to components, CADDEE needs
    #     to know what design condition is concerned. 
    #     """

    #     if name not in self.variables_metadata:
    #         raise ValueError("Unknown variable '{}'. "
    #                          "Acceptable variables are {}.".format(name,list(self.variables_metadata.__dict__['_dict'].keys())))       
    #     else:
    #         # self.variables_metadata[name] = [val, csdl_var, design_condition, computed_upstream, dv_flag, lower, upper, scaler]
    #         if design_condition:
    #             comp_name = self.parameters['name']
    #             cond_name = design_condition.parameters['name']
    #             var_name = '{}_{}'.format(comp_name, name)
    #             self.variables_metadata.__setitem__(name, val, csdl_var, design_condition, units, computed_upstream,
    #                                             dv_flag, lower, upper, scaler)
    #             design_condition.variables_metadata.declare(name=var_name)
    #             design_condition.variables_metadata.__setitem__(var_name, val, csdl_var, design_condition, units, computed_upstream,
    #                                             dv_flag, lower, upper, scaler)
    #         else:
    #             self.variables_metadata.__setitem__(name, val, csdl_var, design_condition, units, computed_upstream,
    #                                             dv_flag, lower, upper, scaler)
        


class LiftingSurface(Component):
    pass

class Rotor(Component):
    def initialize(self, kwargs):
        super().initialize(kwargs)
        self.parameters.declare('component_vars', default=['rpm'])
        # return 

    # def _assemble_csdl(self):
    #     if self.set_module_input.has_been_called:
    #         name = self.parameters['name']
            
    #         csdl_model = ModuleCSDL(module=self, name=name, prepend=name)
    #         R = csdl_model.register_module_input(f'{name}_radius', shape=(1, ), val=1,  computed_upstream=False, promotes=True)
    #         to = csdl_model.register_module_input(f'{name}_thrust_origin', val=np.array([0, 0, 0]), shape=(3, ), computed_upstream=False, promotes=True)
    #         tv = csdl_model.register_module_input(f'{name}_thrust_vector', val=np.array([1, 0, 0]), shape=(3, ), computed_upstream=False, promotes=True)
    #         return csdl_model
    #     else:
    #         print('set_module_input_not called')
    #         exit()

class MotorComp(Component):
    def initialize(self, kwargs):
        super().initialize(kwargs)
        # Parameters
        self.parameters.declare('component_vars', default=['rpm'])

class BatteryComp(Component):
    def initialize(self, kwargs):
        super().initialize(kwargs)
        # Parameters
        self.parameters.declare('component_vars', default=[''])































    # def define_custom_geometry(self, name, val, csdl_var=True, computed_upstream=True, dv_flag=False, lower=None, upper=None, scaler=None):
    #     """
    #     Method to define custom geometry of a component.
    #     Examples: rotor radius, wing corner points, etc.

    #     This method declares a new variable in the variable metadata_dictionary
    #     of the component.
        
    #     Will be depreciated in the future!
    #     """
        
    #     return
    
    # def add_component_state(name, val, design_condition, csdl_var=True, computed_upstream=False, dv_flg=False, upper=None, lower=None, scaler=None):
    #     """
    #     Method for setting a component state.

    #     For this method we again have the keyword "computed_upstream". In addition, 
    #     we have a required "design_condition" argument because CADDEE needs to know
    #     to which condition the component state belongs. 

    #     This method also would live in the Component class. It would "add" the 
    #     states of component-specific variables like 'rpm' or 'elevator_deflection'.
    #     This method would declare a new input to component's variables_metadata
    #     dictionary 

    #     The reason we're using add here is that it is more extensible. It will be 
    #     difficult to predict what kind of variables components will have. 
        
    #     Caddee will later on check whether this variable also exists as an entry 
    #     in the variable_metadata dictionary of the model/solver associated with 
    #     the component and throw an error if it doesn't exist in there. 
    #     """


    #     return