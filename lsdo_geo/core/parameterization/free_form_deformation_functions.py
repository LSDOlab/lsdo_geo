from lsdo_geo.core.parameterization.ffd_block import FFDBlock

from typing import Union
import numpy as np
import lsdo_function_spaces as lfs
import csdl_alpha as csdl
from lsdo_geo.core.geometry.geometry import Geometry

def construct_ffd_block_around_entities(entities:list[Union[np.ndarray, csdl.Variable, lfs.Function, lfs.FunctionSet]],
                                        num_coefficients:tuple[int]=2, degree:tuple[int]=1, name:str='ffd_block') -> FFDBlock:
    '''
    Constructs an FFD block around the given entities and embeds them within.

    Parameters
    ----------
    entities : list[Union[np.ndarray, csdl.Variable, lfs.Function, lfs.FunctionSet]]
        List of entities to be enclosed by the FFD block.
    num_coefficients : tuple[int], optional = 2
        Number of coefficients in each direction of the FFD block, by default 2.
    degree : tuple[int], optional = 2
        Degree of the FFD block, by default 2.
    name : str, optional = 'ffd_block'
        Name of the FFD block, by default 'ffd_block'.

    Returns
    -------
    FFDBlock
        FFD block enclosing the given entities.
    '''
    if not isinstance(entities, list):
        entities = [entities]

    # loop over entities and get points to create enclosure block
    enclosed_points = []
    for entity in entities:
        if isinstance(entity, np.ndarray):
            entity = entity
            enclosed_points.append(entity.reshape(-1, entity.shape[-1]))
        elif isinstance(entity, csdl.Variable):
            entity = entity.value
            enclosed_points.append(entity.reshape(-1, entity.shape[-1]))
        elif isinstance(entity, lfs.Function):
            entity = entity.coefficients.value
            enclosed_points.append(entity.reshape(-1, entity.shape[-1]))
        elif isinstance(entity, lfs.FunctionSet):
            for function in entity.functions.values():
                entity = function.coefficients.value
                enclosed_points.append(entity.reshape(-1, entity.shape[-1]))
        else:
            raise Exception("Please pass in a valid entity type.")

    enclosed_points = np.vstack(enclosed_points)

    num_physical_dimensions = enclosed_points.shape[-1]

    b_spline_hyper_volume = lfs.create_enclosure_block(points=enclosed_points,
                                                             num_coefficients=num_coefficients, degree=degree, knot_vectors=None,
                                                             num_parametric_dimensions=num_physical_dimensions,
                                                             name='b_spline_hyper_volume')
    
    ffd_block = FFDBlock(space=b_spline_hyper_volume.space, coefficients=b_spline_hyper_volume.coefficients, name=name, embedded_entities=entities)
    
    return ffd_block


def construct_ffd_block_from_corners(entities:list[Union[np.ndarray, csdl.Variable, lfs.Function, lfs.FunctionSet, Geometry]],
                                     corners:np.ndarray,
                                        num_coefficients:tuple[int]=2, degree:tuple[int]=1, name:str='ffd_block') -> FFDBlock:
    '''
    Constructs an FFD block around the given entities and embeds them within.
    '''
    if not isinstance(entities, list):
        entities = [entities]

    b_spline_hyper_volume = lfs.create_b_spline_from_corners(corners, degree=degree, num_coefficients=num_coefficients, knot_vectors=None,
                                                             name='b_spline_hyper_volume_for_ffd_block')
    # Just piggybacks on the Function to make the FFD Block
    ffd_block = FFDBlock(space=b_spline_hyper_volume.space, coefficients=b_spline_hyper_volume.coefficients,
                         name=name, embedded_entities=entities)
    
    return ffd_block


def construct_tight_fit_ffd_block(entities:list[Union[np.ndarray, csdl.Variable, lfs.Function, lfs.FunctionSet, Geometry]],
                                        num_coefficients:tuple[int]=5, degree:tuple[int]=2, name:str='ffd_block') -> FFDBlock:
    '''
    Constructs an FFD block around the given entities and embeds them within.
    '''
    if not isinstance(entities, list):
        entities = [entities]

    # loop over entities and get points to create enclosure block

    
    # Steps for currently non-standard FFD blocks
    # 0) Cartesian enclosure volume (line 39) 
    enclosure_ffd_block = construct_ffd_block_around_entities(entities=entities, num_coefficients=num_coefficients,
                                                              degree=degree, name='helper_volume')

    # b_spline_hyper_volume.plot()
    
    # 1) Generate set of parametric coordinates
    num_spanwise_sampling = 20  # THIS MUST BE EVEN TO MAKE SURE WE DON'T GET AN AIRFOIL CROSS SECTION
    sampling_projection_direction = np.array([0., 0., 1.])
    parametric_coordinates_top = np.linspace(np.array([0.5, 0., 1.]), np.array([0.5, 1., 1.]), num_spanwise_sampling)
    # parametric_coordinates_bot = np.linspace(np.array([0.5, 0., -1.]), np.array([0.5. 1.. -1.]), num_spanwise_sampling)

    enclosure_top_points = enclosure_ffd_block.evaluate(parametric_coordinates_top).value
    # ffd_bot_points = bot_evaluation_matrix.dot(b_spline_hyper_volume.coefficients.value)

    top_points_on_wing = entities[0].project(enclosure_top_points, direction=sampling_projection_direction, plot=False)
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

        corner_0i0[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[2]  # z value of first coordinate
        corner_0ip10[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[2]  # z value of first coordinate
        corner_1i0[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[2]  # z value of first coordinate
        corner_1ip10[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[2]  # z value of first coordinate

        corner_0i1[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[5]  # z value of second coordinate
        corner_0ip11[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[5]  # z value of second coordinate
        corner_1i1[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[5]  # z value of second coordinate
        corner_1ip11[2] = enclosure_ffd_block.coefficients.value.reshape((-1,))[5]  # z value of second coordinate

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
        corners[u,v,w,:] = corner_point.reshape((3,))

    ffd_block = construct_ffd_block_from_corners(entities=entities, corners=corners, num_coefficients=num_coefficients, degree=degree, name=name)
    
    
    return ffd_block