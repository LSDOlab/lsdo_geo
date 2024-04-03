from lsdo_geo.core.parameterization.ffd_block import FFDBlock

from typing import Union
import numpy as np
import m3l
from lsdo_geo.splines.b_splines.b_spline import BSpline
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
from lsdo_geo.splines.b_splines.b_spline_sub_set import BSplineSubSet
from lsdo_geo.core.geometry.geometry import Geometry

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

    b_spline_hyper_volume = create_cartesian_enclosure_block(name='b_spline_hyper_volume', points=enclosed_points,
                                                             num_coefficients=num_coefficients, order=order, knot_vectors=None,
                                                             num_parametric_dimensions=num_physical_dimensions, # NOTE: THIS MIGHT NOT WORK?
                                                             num_physical_dimensions=num_physical_dimensions)
    b_spline_hyper_volume.coefficients.name = entities[0].name + '_ffd_block_coefficients'
    
    ffd_block = FFDBlock(name=name, space=b_spline_hyper_volume.space, coefficients=b_spline_hyper_volume.coefficients,
                         num_physical_dimensions=num_physical_dimensions, embedded_entities=entities)
    
    return ffd_block


def construct_ffd_block_from_corners(name:str, entities:list[Union[np.ndarray, m3l.Variable, BSpline, BSplineSet, BSplineSubSet]],
                                     corners:np.ndarray,
                                        num_coefficients:tuple[int]=5, order:tuple[int]=2, num_physical_dimensions:int=3):
    '''
    Constructs an FFD block around the given entities and embeds them within.
    '''
    if type(entities) is not list:
        entities = [entities]

    from lsdo_geo.splines.b_splines.b_spline_functions import create_cartesian_enclosure_block
    from lsdo_geo.splines.b_splines.b_spline_functions import create_b_spline_from_corners
    # loop over entities and get points to create enclosure block
    enclosed_points = []
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

    b_spline_hyper_volume = create_b_spline_from_corners(name, corners, order=order, num_coefficients=num_coefficients, knot_vectors=None)
    b_spline_hyper_volume.coefficients.name = entities[0].name + '_ffd_block_coefficients'
    
    ffd_block = FFDBlock(name=name, space=b_spline_hyper_volume.space, coefficients=b_spline_hyper_volume.coefficients,
                         num_physical_dimensions=num_physical_dimensions, embedded_entities=entities)
    
    return ffd_block


def construct_tight_fit_ffd_block(name:str, entities:list[Union[np.ndarray, m3l.Variable, BSpline, BSplineSet, BSplineSubSet]],
                                        num_coefficients:tuple[int]=5, order:tuple[int]=2, num_physical_dimensions:int=3):
    '''
    Constructs an FFD block around the given entities and embeds them within.
    '''
    if type(entities) is not list:
        entities = [entities]

    from lsdo_geo.splines.b_splines.b_spline_functions import create_cartesian_enclosure_block
    # loop over entities and get points to create enclosure block
    enclosed_points = []
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
    # print('Time for looping over entities: ', t2-t1)
    
    # Steps for currently non-standard FFD blocks
    # 0) Cartesian enclosure volume (line 39) 
    b_spline_hyper_volume = create_cartesian_enclosure_block(name='b_spline_hyper_volume', points=enclosed_points,
                                                             num_coefficients=num_coefficients, order=order, knot_vectors=None,
                                                             num_parametric_dimensions=num_physical_dimensions, # NOTE: THIS MIGHT NOT WORK?
                                                             num_physical_dimensions=num_physical_dimensions)
    b_spline_hyper_volume.coefficients.name = entities[0].name + '_ffd_block_coefficients'

    # b_spline_hyper_volume.plot()
    
    # 1) Generate set of parametric coordinates
    num_spanwise_sampling = 20
    sampling_projection_direction = np.array([0., 0., 1.])
    parametric_coordinates_top = np.linspace(np.array([0.5, 0., 1.]), np.array([0.5, 1., 1.]), num_spanwise_sampling)
    # parametric_coordinates_bot = np.linspace(np.array([0.5, 0., -1.]), np.array([0.5. 1.. -1.]), num_spanwise_sampling)

    top_evaluation_matrix = b_spline_hyper_volume.compute_evaluation_map(parametric_coordinates=parametric_coordinates_top)
    # bot_evaluation_matrix = b_spline_hyper_volume.compute_evaluation_map(parametric_coordinates=parametric_coordinates_bot)

    ffd_top_points = top_evaluation_matrix.dot(b_spline_hyper_volume.coefficients.value).reshape((num_spanwise_sampling, 3))
    # ffd_bot_points = bot_evaluation_matrix.dot(b_spline_hyper_volume.coefficients.value)

    top_points_on_wing = entities[0].project(ffd_top_points, direction=sampling_projection_direction)
    # bot_points_on_wing = entities[0].project(ffd_bot_points, direction=sampling_projection_direction)

    # identify key surfaces
    entity_counters = {}
    for coordinate in top_points_on_wing:
        entity_name = coordinate[0]
        if entity_name in entity_counters:
            entity_counters[entity_name] += 1
        else:
            entity_counters[entity_name] = 0

    key_entities = []
    for entity_name, entity_count in entity_counters.items():
        if entity_count >= 2:
            key_entities.append(entity_name)

    # Evaluate to get corners of key entities
    corner_points = {}
    for i, key_entity in enumerate(key_entities):
        if i <= (len(key_entities)-1)//2:
            parametric_coordinate_00 = [(key_entity, np.array([1., 1.]))]   # u,v are swapped/inverted between wing and ffd
            parametric_coordinate_01 = [(key_entity, np.array([0., 1.]))]   # u,v are swapped/inverted between wing and ffd
            parametric_coordinate_10 = [(key_entity, np.array([1., 0.]))]   # u,v are swapped/inverted between wing and ffd
            parametric_coordinate_11 = [(key_entity, np.array([0., 0.]))]   # u,v are swapped/inverted between wing and ffd
        else:
            parametric_coordinate_00 = [(key_entity, np.array([0., 0.]))]   # u,v are swapped/inverted between wing and ffd
            parametric_coordinate_01 = [(key_entity, np.array([1., 0.]))]   # u,v are swapped/inverted between wing and ffd
            parametric_coordinate_10 = [(key_entity, np.array([0., 1.]))]   # u,v are swapped/inverted between wing and ffd
            parametric_coordinate_11 = [(key_entity, np.array([1., 1.]))]   # u,v are swapped/inverted between wing and ffd

        corner_0i0 = entities[0].evaluate(parametric_coordinate_00).value
        corner_0ip10 = entities[0].evaluate(parametric_coordinate_01).value
        corner_1i0 = entities[0].evaluate(parametric_coordinate_10).value
        corner_1ip10 = entities[0].evaluate(parametric_coordinate_11).value

        corner_0i1 = entities[0].evaluate(parametric_coordinate_00).value
        corner_0ip11 = entities[0].evaluate(parametric_coordinate_01).value
        corner_1i1 = entities[0].evaluate(parametric_coordinate_10).value
        corner_1ip11 = entities[0].evaluate(parametric_coordinate_11).value

        corner_0i0[2] = b_spline_hyper_volume.coefficients.value[2]  # z value of first coordinate
        corner_0ip10[2] = b_spline_hyper_volume.coefficients.value[2]  # z value of first coordinate
        corner_1i0[2] = b_spline_hyper_volume.coefficients.value[2]  # z value of first coordinate
        corner_1ip10[2] = b_spline_hyper_volume.coefficients.value[2]  # z value of first coordinate

        corner_0i1[2] = b_spline_hyper_volume.coefficients.value[5]  # z value of first coordinate
        corner_0ip11[2] = b_spline_hyper_volume.coefficients.value[5]  # z value of first coordinate
        corner_1i1[2] = b_spline_hyper_volume.coefficients.value[5]  # z value of first coordinate
        corner_1ip11[2] = b_spline_hyper_volume.coefficients.value[5]  # z value of first coordinate

        if i == 0:
            corner_points[f'0{i}0'] = corner_0i0
            corner_points[f'0{i}1'] = corner_0i1
            corner_points[f'0{i+1}0'] = corner_0ip10
            corner_points[f'0{i+1}1'] = corner_0ip11
            corner_points[f'1{i}0'] = corner_1i0
            corner_points[f'1{i}1'] = corner_1i1
            corner_points[f'1{i+1}0'] = corner_1ip10
            corner_points[f'1{i+1}1'] = corner_1ip11
        else:
            corner_points[f'0{i+1}0'] = corner_0ip10
            corner_points[f'0{i+1}1'] = corner_0ip11
            corner_points[f'1{i+1}0'] = corner_1ip10
            corner_points[f'1{i+1}1'] = corner_1ip11

    # turn corner points into lines
    corners = np.zeros((2,i+2,2,3))
    for index, corner_point in corner_points.items():
        u = int(index[0])
        v = int(index[1])
        w = int(index[2])
        corners[u,v,w,:] = corner_point

    ffd_block = construct_ffd_block_from_corners(name=name, entities=entities, corners=corners, num_coefficients=num_coefficients, order=order, num_physical_dimensions=num_physical_dimensions)    
    
    
    return ffd_block