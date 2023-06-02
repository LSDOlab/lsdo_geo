import array_mapper as am
import numpy as np
from caddee.caddee_core.system_representation.component.component import Component


class PrescribedActuation:
    '''
    Defines a "solver" for defining actuation within a component. This is the simplest solver
    which is just prescribing the value of the actuation. This is just a parent class and does
    not provide functionality.
    '''

    def __init__(self, component:Component, axis:am.MappedArray, value:np.ndarray=None) -> None:
        self.component = component
        self.axis = axis
        self.value = value


class PrescribedRotation(PrescribedActuation):
    '''
    Defines a "solver" for defining rotational actuation within a component. This is the simplest solver
    which is just prescribing the value of the rotation.
    '''

    def __init__(self, component: Component, axis: am.MappedArray, value:np.ndarray=None) -> None:
        super().__init__(component, axis, value)

        self.value = 0.
        self.units = 'radians'

    def set_rotation(self, name:str, value:np.ndarray, units:str='radians'):
        self.value = value
        self.units = units

    def assemble_csdl(self):
        '''
        Assembles the CSDL model to perform this operation.
        '''
        from caddee.csdl_core.system_representation_csdl.prescribed_rotation_csdl import PrescribedRotationCSDL
        return PrescribedRotationCSDL(prescribed_rotation = self)


class PrescribedTranslation(PrescribedActuation):
    '''
    Defines a "solver" for defining translational actuation within a component. This is the simplest solver
    which is just prescribing the value of the translation.
    '''
    
    def assemble_csdl(self):
        '''
        Assembles the CSDL model to perform this operation.
        '''
        from caddee.csdl_core.system_representation_csdl.system_representation_csdl import PrescribedTranslationCSDL
        return PrescribedTranslationCSDL(prescribed_translation = self)


# from caddee.utils.caddee_base import CADDEEBase

# class PrescribedActuation(CADDEEBase):
#     '''
#     Defines a "solver" for defining actuation within a component. This is the simplest solver
#     which is just prescribing the value of the actuation.
#     '''

#     def initialize(self, kwargs):
#         self.parameters.declare(name='component', allow_none=False, types=Component)
#         self.parameters.declare(name='axis', allow_none=False, types=am.MappedArray)

#     def assign_attributes(self):
#         self.component = self.parameters['component']
#         self.axis = self.parameters['axis']


# class PrescribedRotation(PrescribedActuation):
#     '''
#     Defines a "solver" for defining rotational actuation within a component. This is the simplest solver
#     which is just prescribing the value of the rotation.
#     '''

#     def initialize(self, kwargs):
#         self.parameters.declare(name='units', default='degrees', allow_none=True, types=np.ndarray)
#         return super().initialize(kwargs)


#     def assign_attributes(self):
#         self.units = self.parameters['units']
#         return super().assign_attributes()
    

#     def assemble_csdl(self):
#         '''
#         Assembles the CSDL model to perform this operation.
#         '''
#         from caddee.csdl_core.system_representation_csdl.system_representation_csdl import SystemRepresentationCSDL
#         return SystemRepresentationCSDL(system_representation = self)


# class PrescribedTranslation(PrescribedActuation):
#     '''
#     Defines a "solver" for defining translational actuation within a component. This is the simplest solver
#     which is just prescribing the value of the translation.
#     '''
#     pass
