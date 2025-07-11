import lsdo_geo
import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt
from typing import Union, Optional
import lsdo_function_spaces as lfs


def import_geometry(file_name:str, name:str='geometry', parallelize:bool=False, scale:int=1.0) -> lsdo_geo.Geometry:
    '''
    Imports geometry from a file.

    Parameters
    ----------
    file_name : str
        The name of the file (with path) that containts the geometric information.
    '''
    
    function_set = lfs.import_file(file_name, parallelize=parallelize)
    if scale != 1.0:
        for function in function_set.functions.values():
            function.coefficients = csdl.Variable(value=function.coefficients.value * scale)
    geometry = lsdo_geo.Geometry(functions=function_set.functions, function_names=function_set.function_names, name=name, space=function_set.space)
    return geometry


def rotate(points:Union[csdl.Variable,npt.NDArray[np.float64]], axis_origin:Union[csdl.Variable,npt.NDArray[np.float64]], 
           axis_vector:Union[csdl.Variable,npt.NDArray[np.float64]], angles:Union[csdl.Variable,npt.NDArray[np.float64],float],
           units:str='radians') -> csdl.Variable:
    points_out_shape = None
    if len(points.shape) == 1:
        # print("Rotating points is in vector format, so rotation is assuming 3d and reshaping into (-1,3)")
        points = points.reshape((points.size//3,3))
    if len(points.shape) > 2:
        points_out_shape = points.shape
        points = points.reshape((points.size//points.shape[-1], points.shape[-1]))
        
    if type(points) is np.ndarray:
        points = csdl.Variable(shape=points.shape, value=points)

    if type(axis_origin) is np.ndarray:
        axis_origin = csdl.Variable(shape=axis_origin.shape, value=axis_origin)
    
    # If axis vector is aligned with x, y, or z axis, then instead using rotation matrix (more efficient)
    if isinstance(axis_vector, np.ndarray):
        origin_expanded = csdl.expand(axis_origin, points.shape, 'i->ji')
        if np.allclose(axis_vector, np.array([1,0,0])) or np.allclose(axis_vector, np.array([-1,0,0])):
            if np.allclose(axis_vector, np.array([-1,0,0])):
                angles = -angles
            # rotation_matrix = np.array([[1, 0, 0],
            #                             [0, np.cos(angles), -np.sin(angles)],
            #                             [0, np.sin(angles), np.cos(angles)]])
            # rotated_points = np.dot(points, rotation_matrix)
            cos_angle = csdl.cos(angles)
            sin_angle = csdl.sin(angles)
            rotation_matrix = csdl.Variable(shape=(3,3), value=0.)
            rotation_matrix = rotation_matrix.set(csdl.slice[0,0], 1)
            rotation_matrix = rotation_matrix.set(csdl.slice[1,1], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1,2], sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2,1], -sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2,2], cos_angle)
            # rotated_points = csdl.tensordot(points, rotation_matrix, axes=([-1], [0]))
            rotated_points = csdl.matmat(points - origin_expanded, rotation_matrix)
            rotated_points = rotated_points + origin_expanded
            return rotated_points
        elif np.allclose(axis_vector, np.array([0,1,0])) or np.allclose(axis_vector, np.array([0,-1,0])):
            if np.allclose(axis_vector, np.array([0,-1,0])):
                angles = -angles
            # rotation_matrix = np.array([[np.cos(angles), 0, np.sin(angles)],
            #                             [0, 1, 0],
            #                             [-np.sin(angles), 0, np.cos(angles)]])
            # rotated_points = np.dot(points, rotation_matrix)
            cos_angle = csdl.cos(angles)
            sin_angle = csdl.sin(angles)
            rotation_matrix = csdl.Variable(shape=(3,3), value=0.)
            rotation_matrix = rotation_matrix.set(csdl.slice[0,0], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[0,2], -sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1,1], 1)
            rotation_matrix = rotation_matrix.set(csdl.slice[2,0], sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2,2], cos_angle)
            # rotated_points = csdl.tensordot(points, rotation_matrix, axes=([-1], [0]))
            rotated_points = csdl.matmat(points - origin_expanded, rotation_matrix)
            rotated_points = rotated_points + origin_expanded
            return rotated_points
        elif np.allclose(axis_vector, np.array([0,0,1])) or np.allclose(axis_vector, np.array([0,0,-1])):
            if np.allclose(axis_vector, np.array([0,-1,0])):
                angles = -angles
            # rotation_matrix = np.array([[np.cos(angles), -np.sin(angles), 0],
            #                             [np.sin(angles), np.cos(angles), 0],
            #                             [0, 0, 1]])
            # rotated_points = np.dot(points, rotation_matrix)
            cos_angle = csdl.cos(angles)
            sin_angle = csdl.sin(angles)
            rotation_matrix = csdl.Variable(shape=(3,3), value=0.)
            rotation_matrix = rotation_matrix.set(csdl.slice[0,0], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[0,1], sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1,0], -sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1,1], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2,2], 1)
            # rotated_points = csdl.tensordot(points, rotation_matrix, axes=([-1], [0]))
            rotated_points = csdl.matmat(points - origin_expanded, rotation_matrix)
            rotated_points = rotated_points + origin_expanded
            return rotated_points


    if isinstance(axis_vector, np.ndarray):
        axis_vector = csdl.Variable(shape=axis_vector.shape, value=axis_vector)

    if isinstance(angles, (float, int)):
        angles = csdl.Variable(shape=(1,), value=angles)
    elif isinstance(angles, np.ndarray):
        angles = csdl.Variable(shape=angles.shape, value=angles)
    if units == 'degrees':
        angles = angles * np.pi / 180

    points_wrt_rotation_origin = points - csdl.expand(axis_origin, points.shape, 'i->ji')

    output_shape = (points.shape[0], 4)

    points_wrt_rotation_origin_quaternion = csdl.Variable(shape=output_shape, name='points_wrt_rotation_origin_quaternion',
                                               value=0.)
    points_wrt_rotation_origin_quaternion = points_wrt_rotation_origin_quaternion.set(csdl.slice[:,1:], points_wrt_rotation_origin)
    points_wrt_rotation_origin_quaternion = points_wrt_rotation_origin_quaternion.set(csdl.slice[:,0], 0)

    if angles.shape[0] > 1:
        output_shape = (angles.shape[0],) + output_shape
    else:
        output_shape = (1,) + output_shape

    rotated_points_quaternion = csdl.Variable(shape=output_shape, name='rotated_points', value=0.)
    # for i in csdl.frange(angles.shape[0]):
    for i in range(angles.shape[0]):
        angle = angles[i]

        angle_divide_by_2 = angle / 2
        sin_theta_divide_by_2 = csdl.sin(angle_divide_by_2)

        quaternion_bottom = sin_theta_divide_by_2 * axis_vector
        quaternion = csdl.concatenate((csdl.cos(angle_divide_by_2), quaternion_bottom))

        quaternion = quaternion / csdl.norm(quaternion)
        
        quaternion_conjugate = quaternion
        quaternion_conjugate = quaternion_conjugate.set(csdl.slice[1:], -quaternion_conjugate[1:])

        # for j in csdl.frange(points_wrt_rotation_origin.shape[0]):
        # for j in range(points_wrt_rotation_origin.shape[0]):
            # rotated_points_quaternion = rotated_points_quaternion.set(csdl.slice[i,j,:], 
            #                     hamiltonion_product(quaternion, 
            #                     hamiltonion_product(points_wrt_rotation_origin_quaternion[j],
            #                                         quaternion_conjugate)))
        
        # # baseline
        # rotated_points =  vectorized_hamiltonion_product_2(quaternion, 
        #                     vectorized_hamiltonion_product_1(points_wrt_rotation_origin_quaternion,
        #                                         quaternion_conjugate))
        
        # # csdl function
        # self.vectorized_hamiltonion_product_2 = csdl.Function(vectorized_hamiltonion_product_2)
        # rotated_points =  self.vectorized_hamiltonion_product_2(quaternion, points_wrt_rotation_origin_quaternion,quaternion_conjugate)

        # old
        # rotated_points_quaternion = rotated_points_quaternion.set(csdl.slice[i,:,:], 
        #                     vectorized_hamiltonion_product_2(quaternion, 
        #                     vectorized_hamiltonion_product_1(points_wrt_rotation_origin_quaternion,
        #                                         quaternion_conjugate)))
        rotated_points_quaternion = rotated_points_quaternion.set(csdl.slice[i,:,:],
                                                                  apply_quaternion_rotation(points_wrt_rotation_origin, quaternion))

    if angles.shape[0] == 1:
        rotated_points_wrt_rotation_origin = rotated_points_quaternion[0,:,1:]
        rotated_points = rotated_points_wrt_rotation_origin + csdl.expand(axis_origin, points.shape, 'i->ji')
    else:
        rotated_points_wrt_rotation_origin = rotated_points_quaternion[:,:,1:]
        output_shape = (angles.shape[0], points.shape[0], points.shape[1])
        rotated_points = rotated_points_wrt_rotation_origin + csdl.expand(axis_origin, output_shape, 'i->kij')

    if points_out_shape is not None:
        rotated_points = rotated_points.reshape(points_out_shape)
    return rotated_points


def apply_quaternion_rotation(points:Union[csdl.Variable,npt.NDArray[np.float64]],
                                quaternion:Union[csdl.Variable,npt.NDArray[np.float64]],
                                rotation_origin:Optional[Union[csdl.Variable,npt.NDArray[np.float64]]]=None) -> csdl.Variable:
        '''
        Applies a quaternion rotation to a set of points.
    
        Parameters
        ----------
        points : Union[csdl.Variable, npt.NDArray[np.float64]]
            The points to be rotated.
        quaternion : Union[csdl.Variable, npt.NDArray[np.float64]]
            The quaternion representing the rotation.
    
        Returns
        -------
        csdl.Variable
            The rotated points.
        '''
        
        # if type(points) is np.ndarray:
        #     points = csdl.Variable(shape=points.shape, value=points)
        
        # if type(quaternion) is np.ndarray:
        #     quaternion = csdl.Variable(shape=quaternion.shape, value=quaternion)

        if len(points.shape) == 1:
            # print("Rotating points is in vector format, so rotation is assuming 3d and reshaping into (-1,3)")
            points = points.reshape((points.size//3,3))
        if len(points.shape) > 2:
            points_out_shape = points.shape
            points = points.reshape((points.size//points.shape[-1], points.shape[-1]))
        else:
            points_out_shape = None

        if rotation_origin is not None:
            if type(rotation_origin) is np.ndarray:
                rotation_origin = csdl.Variable(shape=rotation_origin.shape, value=rotation_origin)
            points = points - csdl.expand(rotation_origin, points.shape, 'i->ji')
    
        points_quaternion = csdl.Variable(shape=(points.shape[0], 4), name='points_quaternion', value=0.)
        points_quaternion = points_quaternion.set(csdl.slice[:,1:], points)
        points_quaternion = points_quaternion.set(csdl.slice[:,0], 0)
        quaternion_conjugate = quaternion
        quaternion_conjugate = quaternion_conjugate.set(csdl.slice[1:], -quaternion_conjugate[1:])

        rotated_points_quaternion = vectorized_hamiltonion_product_2(quaternion, 
                                    vectorized_hamiltonion_product_1(points_quaternion, quaternion_conjugate))
            
        if rotation_origin is not None:
            rotated_points_quaternion = rotated_points_quaternion + csdl.expand(rotation_origin, points.shape, 'i->ji')

        if points_out_shape is not None:
            rotated_points_quaternion = rotated_points_quaternion.reshape(points_out_shape)
        return rotated_points_quaternion


def vectorized_hamiltonion_product_1(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    # q1 = q1.reshape((4,))
    # q2 = q2.reshape((4,))

    q1_0 = q1[:,0]
    q1_1 = q1[:,1]
    q1_2 = q1[:,2]
    q1_3 = q1[:,3]

    q2_0 = q2[0]
    q2_1 = q2[1]
    q2_2 = q2[2]
    q2_3 = q2[3]

    # q = csdl.Variable(shape=q1.shape, name='quaternion_product', value=0.)
    # q = q.set(csdl.slice[:,0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
    # q = q.set(csdl.slice[:,1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
    # q = q.set(csdl.slice[:,2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
    # q = q.set(csdl.slice[:,3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

    q_1 = q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3
    q_2 = q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2
    q_3 = q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1
    q_4 = q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0

    q = csdl.vstack((q_1, q_2, q_3, q_4))
    q = q.T()

    return q

def vectorized_hamiltonion_product_2(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    q1_0 = q1[0]
    q1_1 = q1[1]
    q1_2 = q1[2]
    q1_3 = q1[3]

    q2_0 = q2[:,0]
    q2_1 = q2[:,1]
    q2_2 = q2[:,2]
    q2_3 = q2[:,3]

    # q = csdl.Variable(shape=q2.shape, name='quaternion_product', value=0.)
    # q = q.set(csdl.slice[:,0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
    # q = q.set(csdl.slice[:,1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
    # q = q.set(csdl.slice[:,2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
    # q = q.set(csdl.slice[:,3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

    q_1 = q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3
    q_2 = q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2
    q_3 = q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1
    q_4 = q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0

    q = csdl.vstack((q_1, q_2, q_3, q_4))
    q = q.T()

    return q





def hamiltonion_product(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    q1 = q1.reshape((4,))
    q2 = q2.reshape((4,))

    q1_0 = q1[0]
    q1_1 = q1[1]
    q1_2 = q1[2]
    q1_3 = q1[3]

    q2_0 = q2[0]
    q2_1 = q2[1]
    q2_2 = q2[2]
    q2_3 = q2[3]

    q = csdl.Variable(shape=(4,), name='quaternion_product', value=0.)
    q = q.set(csdl.slice[0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
    q = q.set(csdl.slice[1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
    q = q.set(csdl.slice[2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
    q = q.set(csdl.slice[3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

    return q
