from lsdo_geo.splines.b_splines.b_spline_functions import import_file, create_b_spline_set
from lsdo_geo.core.geometry.geometry import Geometry
import os
from pathlib import Path
import pickle


def import_geometry(file_name:str, name:str='geometry', parallelize:bool=True) -> Geometry:
    '''
    Imports geometry from a file.

    Parameters
    ----------
    file_name : str
        The name of the file (with path) that containts the geometric information.
    '''
    from lsdo_geo import IMPORT_FOLDER
    fn = os.path.basename(file_name)
    fn_wo_ext = fn[:fn.rindex('.')]

    saved_geometry = IMPORT_FOLDER / f'{fn_wo_ext}_stored_import_dict.pickle'

    if name == 'geometry':
        name = fn_wo_ext

    saved_geometry_file = Path(saved_geometry) 

    if saved_geometry_file.is_file():
        with open(saved_geometry, 'rb') as handle:
            import_dict = pickle.load(handle)
            b_splines = import_dict['b_splines']

    else:
        import_dict = {}
        b_splines = import_file(file_name, parallelize=parallelize)
        import_dict['b_splines'] = b_splines
        with open(IMPORT_FOLDER / f'{fn_wo_ext}_stored_import_dict.pickle', 'wb+') as handle:
            pickle.dump(import_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    b_spline_set = create_b_spline_set(name, b_splines)
    geometry = Geometry(name, b_spline_set.space, b_spline_set.coefficients, b_spline_set.num_physical_dimensions,
                        b_spline_set.coefficient_indices, b_spline_set.connections)
    return geometry
