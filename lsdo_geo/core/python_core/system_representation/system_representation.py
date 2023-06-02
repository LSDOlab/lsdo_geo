import numpy as np

from caddee.utils.caddee_base import CADDEEBase

from caddee.caddee_core.system_representation.spatial_representation import SpatialRepresentation
from caddee.caddee_core.system_representation.component.component import Component, LiftingSurface, Rotor

# Type checking imports
import array_mapper as am


class SystemRepresentation(CADDEEBase):
    '''
    A SystemRepresentation object is the description of the phyiscal system.
    This description includes all description required to perform the desired analysis.

    Parameters
    -----------
    spatial_representation: SpatialRepresentation = None
        The spatial representation of the physical system.
    power_systems_architecture: list = None
        The power_systems_architecture reprsentation of the physical system.
    components: list = None
        The list of Component objects. Components are user-defined groupings of the system.
    '''

    def initialize(self, kwargs):
        self.parameters.declare(name='spatial_representation', default=None, allow_none=True, types=SpatialRepresentation)
        # self.parameters.declare(name='power_systems_architecture', default=None, allow_none=True, types=power_systems_architectureRepresentation)
        self.parameters.declare(name='power_systems_architecture', default=None, allow_none=True, types=list)  # temporarily leaving this here so no error is thrown.
        self.parameters.declare(name='components', default=None, allow_none=True, types=list)
        self.power_systems_architecture = None
        self.components = {}
        self.configurations = {}

    def assign_attributes(self):
        self.spatial_representation = self.parameters['spatial_representation']
        self.components = self.parameters['components']
        if self.components is None:
            self.components = {}
        self.power_systems_architecture = self.parameters['power_systems_architecture']
        if self.power_systems_architecture is None:
            self.power_systems_architecture = {}

        if self.spatial_representation is None:
            self.spatial_representation = SpatialRepresentation()

    def set_spatial_representation(self, spatial_representation:SpatialRepresentation):
        self.spatial_representation = spatial_representation

    # def add_component(self, component):
    #     self.components[component.name] = component
    def add_component(self, component):
        # print(component.parameters.__dict__['_dict'])
        component_name = component.parameters['name']
        if component_name in self.components:
            raise Exception(f"Component with name '{component_name}' already exists.")
        else:
            if component_name == 'motor_comp':
                print(component_name)
                exit()
            self.components[component_name] = component

    '''
    Defines a connection between two components at a location or region on the respective components.
    '''
    def connect(self, component1:Component, component2:Component, 
                region_on_component1:am.MappedArray=None, region_on_component2:am.MappedArray=None,
                type='mechanical'):
        # NOTE: The regions can also be level set functions isntead of Mapped Arrays.
        pass
        

    def import_geometry(self, file_name : str):
        '''
        Imports geometry primitives from a file.

        Parameters
        ----------
        file_name : str
            The name of the file (with path) that containts the geometric information.
        '''
        self.spatial_representation.import_file(file_name=file_name)
        return self.spatial_representation

    
    def project(self, points:np.ndarray, targets:tuple=None, direction:np.ndarray=None,
                grid_search_n:int=25, max_iterations=100, offset:np.ndarray=None, plot:bool=False):
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
        return self.spatial_representation.project(points=points, targets=targets, direction=direction,
            grid_search_n=grid_search_n, max_iterations=max_iterations, offset=offset, plot=plot)

    def add_input(self, function, connection_name=None, val=None):
        pass
    
    def add_output(self, name, quantity):
        '''
        Adds an output to the system configuration.
        '''
        self.spatial_representation.add_output(name=name, quantity=quantity)


    def create_instances(self, names:list):
        '''
        Create new configurations based on the design configuration.
        '''
        # NOTE: This should return pointers to some sort of dummy objects that can store their additional information.
        #   -- These dummy objects must have methods for taking in the new information like transform or whatever its long term name is.
        for name in names:
            configuration = SystemConfiguration(system_representation=self)
            self.configurations[name] = configuration    # TODO replace name with dummy return object!!
        return self.configurations
    

    def assemble_csdl(self):
        '''
        Constructs and returns the CADDEE model.
        '''
        from caddee.csdl_core.system_representation_csdl.system_representation_csdl import SystemRepresentationCSDL
        return SystemRepresentationCSDL(system_representation = self)


from caddee.caddee_core.system_representation.prescribed_actuations import PrescribedActuation

class SystemConfiguration(CADDEEBase):
    '''
    A SystemRepresentation object is the description of the phyiscal system.
    This description includes all description required to perform the desired analysis.

    Parameters
    -----------
    system_representation: SystemRepresentation
        The system representation that this is a configuration of.
    '''

    def initialize(self, kwargs):
        self.parameters.declare(name='system_representation', allow_none=False, types=SystemRepresentation)

        self.transformations = []

    def assign_attributes(self):
        self.system_representation = self.parameters['system_representation']

    def transform(self, transformation:PrescribedActuation):
        self.transformations.append(transformation)
        
