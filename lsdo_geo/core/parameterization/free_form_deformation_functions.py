from lsdo_geo.core.parameterization.ffd_block import FFDBlock

from typing import Union
import numpy as np
import m3l
from lsdo_geo.splines.b_splines.b_spline import BSpline
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
from lsdo_geo.splines.b_splines.b_spline_sub_set import BSplineSubSet
from lsdo_geo.core.geometry.geometry import Geometry
import time

def construct_ffd_block_around_entities(name:str, entities:list[Union[np.ndarray, m3l.Variable, BSpline, BSplineSet, BSplineSubSet]],
                                        num_coefficients:tuple[int]=5, order:tuple[int]=2, num_physical_dimensions:int=3):
    '''
    Constructs an FFD block around the given entities and embeds them within.
    '''
    if type(entities) is not list:
        entities = [entities]

    from lsdo_geo.splines.b_splines.b_spline_functions import create_cartesian_enclosure_block
    # loop over entities and get points to create enclosure block
    enclosed_points = []
    t1 =  time.time()
    for entity in entities:
        if type(entity) is np.ndarray:
            enclosed_points.append(entity)
        elif type(entity) is m3l.Variable:
            enclosed_points.append(entity.value)
        elif type(entity) is BSpline or type(entity) is BSplineSet or type(entity) is Geometry:
            enclosed_points.append(entity.coefficients.value)
        elif type(entity) is BSplineSubSet:
            enclosed_points.append(entity.get_coefficients().value)
        else:
            raise Exception("Please pass in a valid entity type.")
    enclosed_points = np.hstack(enclosed_points)
    t2 =  time.time()
    # print('Time for looping over entities: ', t2-t1)
    t3 = time.time()
    b_spline_hyper_volume = create_cartesian_enclosure_block(name='b_spline_hyper_volume', points=enclosed_points,
                                                             num_coefficients=num_coefficients, order=order, knot_vectors=None,
                                                             num_parametric_dimensions=num_physical_dimensions, # NOTE: THIS MIGHT NOT WORK?
                                                             num_physical_dimensions=num_physical_dimensions)
    t4 = time.time()
    # print('time for creating enclosure volume: ', t4-t3)

    b_spline_hyper_volume.coefficients.name = name + '_coefficients'
    
    t5 = time.time()
    ffd_block = FFDBlock(name=name, space=b_spline_hyper_volume.space, coefficients=b_spline_hyper_volume.coefficients,
                         num_physical_dimensions=num_physical_dimensions, embedded_entities=entities)
    t6 = time.time()
    # print('time for creating FFD block ', t6-t5)
    return ffd_block