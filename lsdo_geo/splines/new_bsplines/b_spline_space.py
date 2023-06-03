import m3l
from dataclasses import dataclass

@dataclass
class BSplineSpace(m3l.FunctionSpace):
    name : str
    order : tuple
    knots : tuple

