from lsdo_geo.splines.b_splines.b_spline_functions import import_file, create_b_spline_set
from lsdo_geo.core.geometry.geometry import Geometry
import os
from pathlib import Path
import pickle
import m3l


def import_geometry(file_name:str, name:str='geometry', parallelize:bool=True) -> Geometry:
    '''
    Imports geometry from a file.

    Parameters
    ----------
    file_name : str
        The name of the file (with path) that containts the geometric information.
    '''
    fn = os.path.basename(file_name)
    fn_wo_ext = fn[:fn.rindex('.')]

    file_path = f"stored_files/imports/{fn_wo_ext}_stored_import.pickle"
    path = Path(file_path)

    if path.is_file():
        with open(file_path, 'rb') as handle:
            b_splines = pickle.load(handle)
    else:
        b_splines = import_file(file_name, parallelize=parallelize)
        # Since we can't pickle csdl variables, convert them back to numpy arrays
        for b_spline_name, b_spline in b_splines.items():
            b_spline.coefficients = b_spline.coefficients.value

        Path("stored_files/imports").mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb+') as handle:
            pickle.dump(b_splines, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Since we can't pickle csdl variables, convert them back to csdl variables
    for b_spline_name, b_spline in b_splines.items():
        b_spline.coefficients = m3l.Variable(
            shape=b_spline.coefficients.shape,
            value=b_spline.coefficients,
            name=b_spline_name+'_coefficients')

    b_spline_set = create_b_spline_set(name, b_splines)
    geometry = Geometry(name, b_spline_set.space, b_spline_set.coefficients, b_spline_set.num_physical_dimensions,
                        b_spline_set.coefficient_indices, b_spline_set.connections)
    return geometry
