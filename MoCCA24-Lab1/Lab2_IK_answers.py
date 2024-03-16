##############
# 姓名：
# 学号：
##############
import numpy as np
from scipy.spatial.transform import Rotation as R
from Lab1_FK_answers import *
from copy import deepcopy

def rotation_to_align_vectors(A, B):
    """
    Calculates the rotation matrix to align vector A to vector B.
    
    Parameters:
        A (numpy.ndarray): The source vector.
        B (numpy.ndarray): The target vector.
        
    Returns:
        scipy.spatial.transform.Rotation: Rotation object representing the transformation.
    """
    # Normalize vectors
    A_normalized = A / (np.linalg.norm(A) + 1e-6)
    B_normalized = B / (np.linalg.norm(B) + 1e-6)

    # set angle to 0 if the vectors are the same
    if np.allclose(A_normalized, B_normalized):
        return R.from_rotvec([0, 0, 0])

    # Find axis of rotation
    axis_of_rotation = np.cross(A_normalized, B_normalized)
    axis_of_rotation /= np.linalg.norm(axis_of_rotation)

    # Find angle of rotation
    angle = np.arccos(np.dot(A_normalized, B_normalized))

    # Create rotation object
    rotation = R.from_rotvec(angle * axis_of_rotation)

    return rotation



def part1_inverse_kinematics(meta_data, input_joint_positions, input_joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """    

    joint_positions = input_joint_positions
    joint_orientations = input_joint_orientations

    length = joint_positions.shape[0]


    path,_,_,_ = meta_data.get_path_from_root_to_end()
    # precompute all the children and grandchildren of the joints in the path
    list_children = []
    for i in range(len(path)):
        children = [path[i]]
        for _ in range(length):
            for j in range(length):
                if meta_data.joint_parent[j] in children and j not in children:
                    children.append(j)
        list_children.append(children)

    # set max steps to 100 to ensure that part2 will have a fps of 60
    for step in range(100):
        # check if the end joint is close enough to the target
        if np.linalg.norm(joint_positions[path[-1]]-target_pose) < 0.1:
            break

        for i in range(len(path)-1,0,-1):
            cur_joint = path[i]

            # skip the end joint
            if i == len(path)-1:
                continue
            
            # CCD step
            rotation = rotation_to_align_vectors(joint_positions[path[-1]]-joint_positions[cur_joint],target_pose-joint_positions[cur_joint])

            # get all the children and grandchildren of the current joint
            children = list_children[i]

            # apply the rotation to the current joint and all its children and grandchildren
            for child in children:
                joint_orientations[child] = R.as_quat(rotation*R.from_quat(joint_orientations[child]))
                joint_positions[child] = joint_positions[cur_joint] + rotation.apply(joint_positions[child]-joint_positions[cur_joint])
            
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    path,_,_,_ = meta_data.get_path_from_root_to_end()

    lWrist_end = path[-1]
    lWrist = path[-2]
    lelbow = path[-3]


    upright_rot = R.from_euler('XYZ', [0, 0, 90], degrees=True)

    lWrist_offset = upright_rot.apply(meta_data.joint_initial_position[lWrist_end] - meta_data.joint_initial_position[lWrist])

    # equalvalent to the IK problem for the position of lWrist
    target_pos = np.array([relative_x + joint_positions[0][0], target_height, relative_z + joint_positions[0][2]]) - lWrist_offset
    
    meta_data_copy = deepcopy(meta_data)
    meta_data_copy.end_joint = 'lWrist'
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data_copy, joint_positions, joint_orientations, target_pos)

    # force the orientation of lWrist to be upright
    joint_orientations[lWrist] = R.from_euler('XYZ',[0,0,90],degrees=True).as_quat()


    # update the position and orientation of lWrist_end
    joint_positions[lWrist_end] = joint_positions[lWrist] + R.from_quat(joint_orientations[lWrist]).apply(meta_data.joint_initial_position[lWrist_end] - meta_data.joint_initial_position[lWrist])
    joint_orientations[lWrist_end] = joint_orientations[lWrist]

    return joint_positions, joint_orientations
    
