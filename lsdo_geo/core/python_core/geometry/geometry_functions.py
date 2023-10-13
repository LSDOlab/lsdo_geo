from lsdo_geo.splines.b_splines.b_spline_functions import import_file, create_b_spline_set
from lsdo_geo.core.python_core.geometry.geometry import Geometry

def import_geometry(file_name:str, name:str='geometry', parallelize:bool=True) -> Geometry:
    '''
    Imports geometry from a file.

    Parameters
    ----------
    file_name : str
        The name of the file (with path) that containts the geometric information.
    '''
    b_splines = import_file(file_name, parallelize=parallelize)
    b_spline_set = create_b_spline_set(name, b_splines)
    geometry = Geometry(name, b_spline_set.space, b_spline_set.coefficients, b_spline_set.num_physical_dimensions,
                        b_spline_set.coefficient_indices, b_spline_set.connections)
    return geometry
