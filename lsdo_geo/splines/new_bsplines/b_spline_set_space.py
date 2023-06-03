import m3l
from dataclasses import dataclass

@dataclass
class BSplineSetSpace(m3l.FunctionSpace):
    name : str
    order : tuple
    knots : tuple
    b_spline_spaces : dict