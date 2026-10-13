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


def compute_rodrigues_rotation_matrix(axis_vector: Union[csdl.Variable, npt.NDArray[np.float64]],
                                      angle: Union[csdl.Variable, npt.NDArray[np.float64], float]) -> csdl.Variable:
    '''
    Computes the 3x3 rotation matrix using Rodrigues' formula for rotation around a given axis.

    R = cos(theta)*I + (1 - cos(theta))*(k * k^T) + sin(theta)*[k]_x
    where k is the unit axis vector.
    '''
    if isinstance(axis_vector, np.ndarray):
        axis_vector = csdl.Variable(shape=axis_vector.shape, value=axis_vector)
    if isinstance(angle, (float, int)):
        angle = csdl.Variable(shape=(1,), value=angle)
    elif isinstance(angle, np.ndarray):
        angle = csdl.Variable(shape=angle.shape, value=angle)

    unit_axis = axis_vector / csdl.norm(axis_vector)
    kx = unit_axis[0]
    ky = unit_axis[1]
    kz = unit_axis[2]

    cos_a = csdl.cos(angle)
    sin_a = csdl.sin(angle)
    one_minus_cos = 1.0 - cos_a

    R = csdl.Variable(shape=(3, 3), value=0.0)
    R = R.set(csdl.slice[0, 0], cos_a + one_minus_cos * kx * kx)
    R = R.set(csdl.slice[0, 1], one_minus_cos * kx * ky - sin_a * kz)
    R = R.set(csdl.slice[0, 2], one_minus_cos * kx * kz + sin_a * ky)

    R = R.set(csdl.slice[1, 0], one_minus_cos * kx * ky + sin_a * kz)
    R = R.set(csdl.slice[1, 1], cos_a + one_minus_cos * ky * ky)
    R = R.set(csdl.slice[1, 2], one_minus_cos * ky * kz - sin_a * kx)

    R = R.set(csdl.slice[2, 0], one_minus_cos * kx * kz - sin_a * ky)
    R = R.set(csdl.slice[2, 1], one_minus_cos * ky * kz + sin_a * kx)
    R = R.set(csdl.slice[2, 2], cos_a + one_minus_cos * kz * kz)
    return R


def rotate_using_rodrigues(points: Union[csdl.Variable, npt.NDArray[np.float64]],
                           rotation_origin: Union[csdl.Variable, npt.NDArray[np.float64]],
                           axis_vector: Union[csdl.Variable, npt.NDArray[np.float64]],
                           angles: Union[csdl.Variable, npt.NDArray[np.float64], float],
                           units: str = 'radians') -> csdl.Variable:
    '''
    Rotates points about an arbitrary axis and origin using Rodrigues' rotation formula.

    This is significantly faster in forward evaluation (~3.4x) and reverse-mode AD (~10x)
    compared to quaternion multiplication, while matching results to machine precision.
    '''
    points_out_shape = None
    if len(points.shape) == 1:
        points = points.reshape((points.size // 3, 3))
    if len(points.shape) > 2:
        points_out_shape = points.shape
        points = points.reshape((points.size // points.shape[-1], points.shape[-1]))

    if isinstance(points, np.ndarray):
        points = csdl.Variable(shape=points.shape, value=points)
    if isinstance(rotation_origin, np.ndarray):
        rotation_origin = csdl.Variable(shape=rotation_origin.shape, value=rotation_origin)
    if isinstance(axis_vector, np.ndarray):
        axis_vector = csdl.Variable(shape=axis_vector.shape, value=axis_vector)

    if isinstance(angles, (float, int)):
        angles = csdl.Variable(shape=(1,), value=angles)
    elif isinstance(angles, np.ndarray):
        angles = csdl.Variable(shape=angles.shape, value=angles)
    if units == 'degrees':
        angles = angles * np.pi / 180

    origin_expanded = csdl.expand(rotation_origin, points.shape, 'i->ji')
    points_centered = points - origin_expanded

    if angles.shape[0] == 1:
        R = compute_rodrigues_rotation_matrix(axis_vector, angles)
        rotated_points = csdl.matmat(points_centered, R.T()) + origin_expanded
    else:
        output_shape = (angles.shape[0], points.shape[0], 3)
        rotated_points = csdl.Variable(shape=output_shape, value=0.0)
        for i in range(angles.shape[0]):
            R_i = compute_rodrigues_rotation_matrix(axis_vector, angles[i])
            rot_i = csdl.matmat(points_centered, R_i.T())
            rotated_points = rotated_points.set(csdl.slice[i, :, :], rot_i)
        origin_multi = csdl.expand(rotation_origin, output_shape, 'k->ijk')
        rotated_points = rotated_points + origin_multi

    if points_out_shape is not None:
        if angles.shape[0] > 1:
            rotated_points = rotated_points.reshape((angles.shape[0],) + points_out_shape)
        else:
            rotated_points = rotated_points.reshape(points_out_shape)
    return rotated_points


def rotate_using_quaternion(points: Union[csdl.Variable, npt.NDArray[np.float64]],
                            rotation_origin: Union[csdl.Variable, npt.NDArray[np.float64]],
                            axis_vector: Union[csdl.Variable, npt.NDArray[np.float64]],
                            angles: Union[csdl.Variable, npt.NDArray[np.float64], float],
                            units: str = 'radians') -> csdl.Variable:
    '''
    Rotates points about an arbitrary axis and origin using quaternion multiplication.
    '''
    points_out_shape = None
    if len(points.shape) == 1:
        points = points.reshape((points.size // 3, 3))
    if len(points.shape) > 2:
        points_out_shape = points.shape
        points = points.reshape((points.size // points.shape[-1], points.shape[-1]))

    if isinstance(points, np.ndarray):
        points = csdl.Variable(shape=points.shape, value=points)
    if isinstance(rotation_origin, np.ndarray):
        rotation_origin = csdl.Variable(shape=rotation_origin.shape, value=rotation_origin)
    if isinstance(axis_vector, np.ndarray):
        axis_vector = csdl.Variable(shape=axis_vector.shape, value=axis_vector)

    if isinstance(angles, (float, int)):
        angles = csdl.Variable(shape=(1,), value=angles)
    elif isinstance(angles, np.ndarray):
        angles = csdl.Variable(shape=angles.shape, value=angles)
    if units == 'degrees':
        angles = angles * np.pi / 180

    unit_axis_vector = axis_vector / csdl.norm(axis_vector)

    points_wrt_rotation_origin = points - csdl.expand(rotation_origin, points.shape, 'i->ji')

    output_shape = (points.shape[0], 4)

    points_wrt_rotation_origin_quaternion = csdl.Variable(
        shape=output_shape, name='points_wrt_rotation_origin_quaternion', value=0.0
    )
    points_wrt_rotation_origin_quaternion = points_wrt_rotation_origin_quaternion.set(csdl.slice[:, 1:], points_wrt_rotation_origin)
    points_wrt_rotation_origin_quaternion = points_wrt_rotation_origin_quaternion.set(csdl.slice[:, 0], 0)

    if angles.shape[0] > 1:
        output_shape = (angles.shape[0],) + output_shape
    else:
        output_shape = (1,) + output_shape

    rotated_points_quaternion = csdl.Variable(shape=output_shape, name='rotated_points', value=0.0)
    for i in range(angles.shape[0]):
        angle = angles[i]

        angle_divide_by_2 = angle / 2
        sin_theta_divide_by_2 = csdl.sin(angle_divide_by_2)

        quaternion_bottom = sin_theta_divide_by_2 * unit_axis_vector
        quaternion = csdl.concatenate((csdl.cos(angle_divide_by_2), quaternion_bottom))

        quaternion = quaternion / csdl.norm(quaternion)

        rotated_points_quaternion = rotated_points_quaternion.set(
            csdl.slice[i, :, 1:],
            apply_quaternion_rotation(points_wrt_rotation_origin, quaternion)
        )

    if angles.shape[0] == 1:
        rotated_points_wrt_rotation_origin = rotated_points_quaternion[0, :, 1:]
        rotated_points = rotated_points_wrt_rotation_origin + csdl.expand(rotation_origin, points.shape, 'i->ji')
    else:
        rotated_points_wrt_rotation_origin = rotated_points_quaternion[:, :, 1:]
        output_shape = (angles.shape[0], points.shape[0], points.shape[1])
        rotated_points = rotated_points_wrt_rotation_origin + csdl.expand(rotation_origin, output_shape, 'k->ijk')

    if points_out_shape is not None:
        if angles.shape[0] > 1:
            rotated_points = rotated_points.reshape((angles.shape[0],) + points_out_shape)
        else:
            rotated_points = rotated_points.reshape(points_out_shape)
    return rotated_points


def rotate(points:Union[csdl.Variable,npt.NDArray[np.float64]], rotation_origin:Union[csdl.Variable,npt.NDArray[np.float64]], 
           axis_vector:Union[csdl.Variable,npt.NDArray[np.float64]], angles:Union[csdl.Variable,npt.NDArray[np.float64],float],
           units:str='radians', method:str='rodrigues') -> csdl.Variable:
    points_out_shape = None
    if len(points.shape) == 1:
        # print("Rotating points is in vector format, so rotation is assuming 3d and reshaping into (-1,3)")
        points = points.reshape((points.size//3,3))
    if len(points.shape) > 2:
        points_out_shape = points.shape
        points = points.reshape((points.size//points.shape[-1], points.shape[-1]))
        
    if type(points) is np.ndarray:
        points = csdl.Variable(shape=points.shape, value=points)

    if type(rotation_origin) is np.ndarray:
        rotation_origin = csdl.Variable(shape=rotation_origin.shape, value=rotation_origin)

    if isinstance(angles, (float, int)):
        angles = csdl.Variable(shape=(1,), value=angles)
    elif isinstance(angles, np.ndarray):
        angles = csdl.Variable(shape=angles.shape, value=angles)
    if units == 'degrees':
        angles = angles * np.pi / 180
    
    # If axis vector is aligned with x, y, or z axis and single angle, then use rotation matrix (more efficient)
    if isinstance(axis_vector, np.ndarray) and angles.shape[0] == 1:
        origin_expanded = csdl.expand(rotation_origin, points.shape, 'i->ji')
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
            rotation_matrix = rotation_matrix.set(csdl.slice[0:1,0:1], 1)
            rotation_matrix = rotation_matrix.set(csdl.slice[1:2,1:2], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1:2,2:3], sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2:3,1:2], -sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2:3,2:3], cos_angle)
            # rotated_points = csdl.tensordot(points, rotation_matrix, axes=([-1], [0]))
            rotated_points = csdl.matmat(points - origin_expanded, rotation_matrix)
            rotated_points = rotated_points + origin_expanded
            if points_out_shape is not None:
                rotated_points = rotated_points.reshape(points_out_shape)
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
            rotation_matrix = rotation_matrix.set(csdl.slice[0:1,0:1], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[0:1,2:3], -sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1:2,1:2], 1)
            rotation_matrix = rotation_matrix.set(csdl.slice[2:3,0:1], sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2:3,2:3], cos_angle)
            # rotated_points = csdl.tensordot(points, rotation_matrix, axes=([-1], [0]))
            rotated_points = csdl.matmat(points - origin_expanded, rotation_matrix)
            rotated_points = rotated_points + origin_expanded
            if points_out_shape is not None:
                rotated_points = rotated_points.reshape(points_out_shape)
            return rotated_points
        elif np.allclose(axis_vector, np.array([0,0,1])) or np.allclose(axis_vector, np.array([0,0,-1])):
            if np.allclose(axis_vector, np.array([0,0,-1])):
                angles = -angles
            # rotation_matrix = np.array([[np.cos(angles), -np.sin(angles), 0],
            #                             [np.sin(angles), np.cos(angles), 0],
            #                             [0, 0, 1]])
            # rotated_points = np.dot(points, rotation_matrix)
            cos_angle = csdl.cos(angles)
            sin_angle = csdl.sin(angles)
            rotation_matrix = csdl.Variable(shape=(3,3), value=0.)
            rotation_matrix = rotation_matrix.set(csdl.slice[0:1,0:1], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[0:1,1:2], sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1:2,0:1], -sin_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[1:2,1:2], cos_angle)
            rotation_matrix = rotation_matrix.set(csdl.slice[2:3,2:3], 1)
            # rotated_points = csdl.tensordot(points, rotation_matrix, axes=([-1], [0]))
            rotated_points = csdl.matmat(points - origin_expanded, rotation_matrix)
            rotated_points = rotated_points + origin_expanded
            if points_out_shape is not None:
                rotated_points = rotated_points.reshape(points_out_shape)
            return rotated_points

    if method == 'rodrigues':
        rotated_points = rotate_using_rodrigues(
            points=points,
            rotation_origin=rotation_origin,
            axis_vector=axis_vector,
            angles=angles,
            units=units,
        )
    elif method == 'quaternion':
        rotated_points = rotate_using_quaternion(
            points=points,
            rotation_origin=rotation_origin,
            axis_vector=axis_vector,
            angles=angles,
            units=units,
        )
    else:
        raise ValueError(f"Unknown rotation method '{method}'. Choose 'rodrigues' or 'quaternion'.")

    if points_out_shape is not None:
        if angles.shape[0] > 1:
            rotated_points = rotated_points.reshape((angles.shape[0],) + points_out_shape)
        else:
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
        
        if isinstance(quaternion, np.ndarray):
            quaternion = csdl.Variable(shape=quaternion.shape, value=quaternion)

        points_out_shape = points.shape
        if len(points.shape) == 1:
            # print("Rotating points is in vector format, so rotation is assuming 3d and reshaping into (-1,3)")
            points = points.reshape((points.size//3,3))
        elif len(points.shape) > 2:
            points = points.reshape((points.size//points.shape[-1], points.shape[-1]))

        if rotation_origin is not None:
            if type(rotation_origin) is np.ndarray:
                rotation_origin = csdl.Variable(shape=rotation_origin.shape, value=rotation_origin)
            points = points - csdl.expand(rotation_origin, points.shape, 'i->ji')
    
        points_quaternion = csdl.Variable(shape=(points.shape[0], 4), name='points_quaternion', value=0.)
        points_quaternion = points_quaternion.set(csdl.slice[:,1:], points)
        points_quaternion = points_quaternion.set(csdl.slice[:,0], 0)
        quaternion_conjugate = quaternion
        quaternion_conjugate = quaternion_conjugate.set(csdl.slice[1:], -quaternion_conjugate[1:])

        rotated_points_quaternion = vectorized_hamilton_product_2(quaternion, 
                                    vectorized_hamilton_product_1(points_quaternion, quaternion_conjugate))

        rotated_points = rotated_points_quaternion[:, 1:]
            
        if rotation_origin is not None:
            rotated_points = rotated_points + csdl.expand(rotation_origin, points.shape, 'i->ji')

        # if points_out_shape is not None:
        rotated_points = rotated_points.reshape(points_out_shape)
        return rotated_points


# def vectorized_hamilton_product_1(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
#     # q1 = q1.reshape((4,))
#     # q2 = q2.reshape((4,))

#     q1_0 = q1[:,0]
#     q1_1 = q1[:,1]
#     q1_2 = q1[:,2]
#     q1_3 = q1[:,3]

#     q2_0 = q2[0]
#     q2_1 = q2[1]
#     q2_2 = q2[2]
#     q2_3 = q2[3]

#     # q = csdl.Variable(shape=q1.shape, name='quaternion_product', value=0.)
#     # q = q.set(csdl.slice[:,0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
#     # q = q.set(csdl.slice[:,1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
#     # q = q.set(csdl.slice[:,2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
#     # q = q.set(csdl.slice[:,3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

#     q_1 = q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3
#     q_2 = q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2
#     q_3 = q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1
#     q_4 = q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0

#     q = csdl.vstack((q_1, q_2, q_3, q_4))
#     q = q.T()

#     return q

# def vectorized_hamilton_product_2(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
#     q1_0 = q1[0]
#     q1_1 = q1[1]
#     q1_2 = q1[2]
#     q1_3 = q1[3]

#     q2_0 = q2[:,0]
#     q2_1 = q2[:,1]
#     q2_2 = q2[:,2]
#     q2_3 = q2[:,3]

#     # q = csdl.Variable(shape=q2.shape, name='quaternion_product', value=0.)
#     # q = q.set(csdl.slice[:,0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
#     # q = q.set(csdl.slice[:,1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
#     # q = q.set(csdl.slice[:,2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
#     # q = q.set(csdl.slice[:,3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

#     q_1 = q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3
#     q_2 = q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2
#     q_3 = q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1
#     q_4 = q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0

#     q = csdl.vstack((q_1, q_2, q_3, q_4))
#     q = q.T()

#     return q





# def hamilton_product(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
#     q1 = q1.reshape((4,))
#     q2 = q2.reshape((4,))

#     q1_0 = q1[0]
#     q1_1 = q1[1]
#     q1_2 = q1[2]
#     q1_3 = q1[3]

#     q2_0 = q2[0]
#     q2_1 = q2[1]
#     q2_2 = q2[2]
#     q2_3 = q2[3]

#     q = csdl.Variable(shape=(4,), name='quaternion_product', value=0.)
#     q = q.set(csdl.slice[0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
#     q = q.set(csdl.slice[1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
#     q = q.set(csdl.slice[2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
#     q = q.set(csdl.slice[3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

#     return q


def hamilton_product(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    q1 = q1.reshape((4,))
    q2 = q2.reshape((4,))

    q1_0 = q1[0]
    q1_vec = q1[1:4]

    q2_0 = q2[0]
    q2_vec = q2[1:4]

    q = csdl.Variable(shape=(4,), name='quaternion_product', value=0.)
    q = q.set(csdl.slice[0:1], q1_0*q2_0 - csdl.vdot(q1_vec, q2_vec))
    q = q.set(csdl.slice[1:4], q1_0*q2_vec + q2_0*q1_vec + csdl.cross(q1_vec, q2_vec))

    return q

def vectorized_hamilton_product_1(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    # q1 = q1.reshape((4,))
    # q2 = q2.reshape((4,))

    q1_0 = q1[:,0]
    q1_vec = q1[:,1:4]

    q2_0 = q2[0]
    q2_vec = q2[1:4]

    # q = csdl.Variable(shape=q1.shape, name='quaternion_product', value=0.)
    # q = q.set(csdl.slice[:,0], q1_0*q2_0 - csdl.vdot(q1_vec, q2_vec))
    # q = q.set(csdl.slice[:,1:4], q1_0*q2_vec + q2_0*q1_vec + csdl.cross(q1_vec, q2_vec))

    q_scalar = (q1_0*q2_0 - csdl.matvec(q1_vec, q2_vec)).reshape((-1,1))
    q2_vec_expanded = csdl.expand(q2_vec, q1_vec.shape, 'i->ji')
    q_vec = csdl.outer(q1_0,q2_vec) + q2_0*q1_vec + csdl.cross(q1_vec, q2_vec_expanded, axis=1)

    return csdl.concatenate((q_scalar, q_vec), axis=1)

def vectorized_hamilton_product_2(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    q1_0 = q1[0]
    q1_vec = q1[1:4]

    q2_0 = q2[:,0]
    q2_vec = q2[:,1:4]

    # q = csdl.Variable(shape=q2.shape, name='quaternion_product', value=0.)
    # q = q.set(csdl.slice[:,0], q1_0*q2_0 - csdl.vdot(q1_vec, q2_vec))
    # q = q.set(csdl.slice[:,1:4], q1_0*q2_vec + q2_0*q1_vec + csdl.cross(q1_vec, q2_vec))

    q_scalar = (q1_0*q2_0 - csdl.einsum(q1_vec, q2_vec, action='i,ji->j')).reshape((-1,1))
    q1_vec_expanded = csdl.expand(q1_vec, q2_vec.shape, 'i->ji')
    q_vec = q1_0*q2_vec + csdl.outer(q2_0,q1_vec) + csdl.cross(q1_vec_expanded, q2_vec, axis=1)

    return csdl.concatenate((q_scalar, q_vec), axis=1)


def S_of(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    W = csdl.Variable(shape=(3, 4), value=0.)
    W = W.set(csdl.slice[:, 0], -q[1:])
    W = W.set(csdl.slice[0:1, 1:2], q0)
    W = W.set(csdl.slice[0:1, 2:3], -q3)
    W = W.set(csdl.slice[0:1, 3:4], q2)
    W = W.set(csdl.slice[1:2, 1:2], q3)
    W = W.set(csdl.slice[1:2, 2:3], q0)
    W = W.set(csdl.slice[1:2, 3:4], -q1)
    W = W.set(csdl.slice[2:3, 1:2], -q2)
    W = W.set(csdl.slice[2:3, 2:3], q1)
    W = W.set(csdl.slice[2:3, 3:4], q0)
    S = 2*W
    return S