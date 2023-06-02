from caddee.utils.caddee_base import CADDEEBase
from caddee.caddee_core.system_representation.spatial_representation.component.component import Component

class PowerNode(CADDEEBase): pass

class RotorNode(PowerNode): 
    def initialize(self, kwargs):
        self.parameters.declare(name='component', default=None, types=Component, allow_none=True)

class MotorNode(PowerNode): 
    def initialize(self, kwargs):
        self.parameters.declare(name='component', default=None, types=Component, allow_none=True)

class BatteryNode(PowerNode):
    def initialize(self, kwargs):
        self.parameters.declare(name='component', default=None, types=Component, allow_none=True)