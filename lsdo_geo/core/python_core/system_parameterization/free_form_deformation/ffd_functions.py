'''
This file is for functions associated with the FFD "Package"
'''

from caddee.primitives.bsplines.bspline_surface import BSplineSurface
from caddee.primitives.bsplines.bspline_volume import BSplineVolume
from caddee.primitives.bsplines.bspline_functions import generate_open_uniform_knot_vector, create_bspline_from_corners

import numpy as np

# TODO to work for 2d as well (hyper-volume or nd volume)
def create_cartesian_enclosure_volume(enclosed_entities:list, num_control_points:tuple, order:tuple, knot_vectors:tuple=None, xyz_to_uvw_indices:tuple=(0,1,2), volume_type:str='BSplineVolume'):
    '''
    Creates a volume that tightly fits around a set of entities.
    '''
    if type(enclosed_entities) is dict:
        enclosed_entities = list(enclosed_entities.values())

    if knot_vectors is None:
        knot_vectors=()
        for i in range(len(num_control_points)):
            axis_knots = generate_open_uniform_knot_vector(num_control_points[i], order[i])
            knot_vectors = knot_vectors + (axis_knots,)      # append knots

    embedded_points = None

    for enclosed_entity in enclosed_entities:
        if type(enclosed_entity) is BSplineSurface:
            entity_num_control_points = np.cumprod(enclosed_entity.shape[:-1])[-1]
            points = enclosed_entity.control_points.reshape((entity_num_control_points, -1))
        elif type(enclosed_entity) is np.ndarray:
            points = enclosed_entity
        else:
            raise Exception("Please input a BSplineSurface or np.ndarray as the enclosed entities")
        
        if embedded_points is None:
            embedded_points = points
        else:
            embedded_points = np.vstack((embedded_points, points))

    mins = np.min(embedded_points, axis=0)
    maxs = np.max(embedded_points, axis=0)

    # Can probably automate this to nd using a for loop
    points = np.zeros((2,2,2,3))
    points[0,0,0,:] = np.array([mins[0], mins[1], mins[2]])
    points[0,0,1,:] = np.array([mins[0], mins[1], maxs[2]])
    points[0,1,0,:] = np.array([mins[0], maxs[1], mins[2]])
    points[0,1,1,:] = np.array([mins[0], maxs[1], maxs[2]])
    points[1,0,0,:] = np.array([maxs[0], mins[1], mins[2]])
    points[1,0,1,:] = np.array([maxs[0], mins[1], maxs[2]])
    points[1,1,0,:] = np.array([maxs[0], maxs[1], mins[2]])
    points[1,1,1,:] = np.array([maxs[0], maxs[1], maxs[2]])

    xyz_to_uvw_indices = list(xyz_to_uvw_indices)
    for uvw_index in range(len(xyz_to_uvw_indices)-1):
        xyz_index = xyz_to_uvw_indices[uvw_index]
        points = np.swapaxes(points, xyz_index, uvw_index)
        swapped_xyz_index = xyz_to_uvw_indices.index(uvw_index)
        new_xyz_to_uvw_indices = xyz_to_uvw_indices.copy()
        new_xyz_to_uvw_indices[uvw_index] = uvw_index
        new_xyz_to_uvw_indices[swapped_xyz_index] = xyz_index
        xyz_to_uvw_indices = new_xyz_to_uvw_indices

    volume = create_bspline_from_corners(corners=points, order=order, num_control_points=num_control_points,
            knot_vectors=knot_vectors, name=None)

    return volume

