from csdl import Model
# from lsdo_geo.caddee_core.caddee import CADDEE
from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
from lsdo_geo.caddee_core.system_parameterization.system_parameterization import SystemParameterization
from lsdo_geo.caddee_core.system_model.system_model import SystemModel
from lsdo_geo.csdl_core.system_model_csdl.system_model_csdl import SystemModelCSDL
from lsdo_geo.csdl_core.system_parameterization_csdl.system_parameterization_csdl import SystemParameterizationCSDL


class CADDEECSDL(Model):
    """
    Top-level caddee csdl class

    There are three parameters that contain the three 
    python classes contained in the CADDEE class
        1) SystemRepresentation
        2) SystemParameterization
        3) SystemModel
    """
    
    def initialize(self):
        self.parameters.declare('caddee') #, types=CADDEE)
        # self.parameters.declare('system_representation', types=SystemRepresentation)
        # self.parameters.declare('system_parameterization')# , types=(SystemParameterization, None))
        # self.parameters.declare('system_model', types=SystemModel)
        # establish a pattern where the pure python instances corresponding to 
        # csdl object are declared as parameters (or their contained classes)

    
    def define(self):
        # caddee
        caddee = self.parameters['caddee']
        # system configuration & parameterization
        system_representation = lsdo_geo.system_representation
        psa_connections = system_representation.power_systems_architecture.connections_list
        system_parameterization = lsdo_geo.system_parameterization
        system_parameterization_csdl = SystemParameterizationCSDL(
            system_representation=system_representation,
            system_parameterization=system_parameterization,
        )
        self.add(system_parameterization_csdl, 'system_parameterization')

        # system model
        system_model = lsdo_geo.system_model
        system_model_csdl = SystemModelCSDL(
            system_model=system_model,
            psa_connections=psa_connections,    
        )
        self.add(system_model_csdl, 'system_model')
        
        
        # NOTE: previously we would suppress promotions but now, objects like meshes 
        # that live in system_representation_csdl need to be known downstream in 
        # system_model_csdl, so here, it is ok to promote
        
                
       


# from csdl import Model
# from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
# from lsdo_geo.caddee_core.system_parameterization.system_parameterization import SystemParameterization
# from lsdo_geo.caddee_core.system_model.system_model import SystemModel
# from lsdo_geo.csdl_core.system_model_csdl.system_model_csdl import SystemModelCSDL
# from lsdo_geo.csdl_core.system_representation_csdl.system_representation_csdl import SystemRepresentationCSDL


# class CADDEECSDL(Model):
#     """
#     Top-level caddee csdl class

#     There are three parameters that contain the three 
#     python classes contained in the CADDEE class
#         1) SystemRepresentation
#         2) SystemParameterization
#         3) SystemModel
#     """
    
#     def initialize(self):
#         self.parameters.declare('caddee', types=CADDEE)
#         self.parameters.declare('system_representation', types=SystemRepresentation)
#         self.parameters.declare('system_parameterization')# , types=(SystemParameterization, None))
#         self.parameters.declare('system_model', types=SystemModel)
#         # establish a pattern where the pure python instances corresponding to 
#         # csdl object are declared as parameters (or their contained classes)

#         self.system_representation_csdl = None
#         self.system_model_csdl = None
    
#     def define(self):
#         # system configuration & parameterization
#         system_representation = self.parameters['system_representation']
#         system_parameterization = self.parameters['system_parameterization']
#         system_representation_csdl = SystemRepresentationCSDL(
#             system_representation=system_representation,
#             system_parameterization=system_parameterization,
#         )
#         self.add(system_representation_csdl, 'system_representation')
#         self.system_representation_csdl = system_representation_csdl

#         # system model
#         system_model = self.parameters['system_model']
#         system_model_csdl = SystemModelCSDL(system_model=system_model)
#         self.add(system_model_csdl, 'system_model')
#         self.system_model_csdl = system_model_csdl
        
        
#         # NOTE: previously we would suppress promotions but now, objects like meshes 
#         # that live in system_representation_csdl need to be known downstream in 
#         # system_model_csdl, so here, it is ok to promote
        

#         test_input = self.declare_variable('test_csdl_input', 0.)
#         self.register_output('caddee_csdl_test_output', test_input + 1)
                
       