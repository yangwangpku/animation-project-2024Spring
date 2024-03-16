##############
# 姓名：
# 学号：
##############
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    with open(bvh_file_path, 'r') as file:
        lines = file.readlines()

    joint_name = []
    joint_parent = []
    joint_offset = []

    joint_stack = []

    for line in lines:
        line = line.strip()
        if line.startswith('ROOT') or line.startswith('JOINT') or line.startswith('End Site'):

            # update parent
            if joint_stack:
                joint_parent.append(joint_name.index(joint_stack[-1]))
            else:
                joint_parent.append(-1)

            # get name
            if line.splitlines()[0] == 'End Site':
                name = joint_stack[-1] + '_end'
            else:
                name = line.split()[1]
            joint_name.append(name)
            joint_stack.append(name)

        elif line.startswith('OFFSET'):
            offset = [float(x) for x in line.split()[1:]]
            joint_offset.append(offset)
        elif line.startswith('}'):
            joint_stack.pop()
    
    return joint_name, joint_parent,joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = []

    motion_data_frame = motion_data[frame_id]
    motion_data_ptr = 0

    length = len(joint_name)
    for i in range(length):
        if joint_parent[i] == -1:
            joint_positions[i] = motion_data_frame[motion_data_ptr:motion_data_ptr+3]
            motion_data_ptr += 3
            joint_orientations.append(R.from_euler('XYZ', motion_data_frame[motion_data_ptr:motion_data_ptr+3] , degrees=True))
            motion_data_ptr += 3
        elif joint_name[i].endswith('_end'):
            joint_positions[i] = joint_positions[joint_parent[i]] + joint_orientations[joint_parent[i]].apply(joint_offset[i])
            joint_orientations.append(joint_orientations[joint_parent[i]])
        else:
            joint_positions[i] = joint_positions[joint_parent[i]] + joint_orientations[joint_parent[i]].apply(joint_offset[i])
            joint_orientations.append(joint_orientations[joint_parent[i]] * R.from_euler('XYZ', motion_data_frame[motion_data_ptr:motion_data_ptr+3], degrees=True))
            motion_data_ptr += 3

    joint_orientations = np.array([x.as_quat() for x in joint_orientations])
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_name_T, joint_parent_T,joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A,joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)

    # joint name 顺序可能不一致
    assert(joint_name_T == joint_name_A)
    assert(joint_parent_T == joint_parent_A)

    joint_name = joint_name_T
    joint_parent = joint_parent_T

    # get one child for each joint
    joint_child = [-1] * len(joint_name)
    for i in range(len(joint_name)):
        for j in range(len(joint_name)):
            if joint_parent[j] == i:
                joint_child[i] = j
                break

    motion_data_A = load_motion_data(A_pose_bvh_path)
    motion_data_T = np.zeros_like(motion_data_A)


    for frame in range(motion_data_A.shape[0]):
        motion_data_T_frame = motion_data_T[frame]
        motion_data_A_frame = motion_data_A[frame]
        motion_data_ptr = 0
        length = len(joint_name)
        for i in range(length):
            if joint_parent[i] == -1:
                motion_data_T_frame[motion_data_ptr:motion_data_ptr+6] = motion_data_A_frame[motion_data_ptr:motion_data_ptr+6]
                motion_data_ptr += 6
            elif joint_name[i].endswith('_end'):
                continue
            else:
                parent = joint_parent[i]
                child = joint_child[i]

                # one rotation that transforms joint_offset_T[i] to joint_offset_A[i]
                R_i = rotation_to_align_vectors(joint_offset_T[i], joint_offset_A[i])
                R_ci = rotation_to_align_vectors(joint_offset_T[child], joint_offset_A[child])

                motion_data_T_frame[motion_data_ptr:motion_data_ptr+3] = (R_i.inv()*R.from_euler("XYZ",motion_data_A_frame[motion_data_ptr:motion_data_ptr+3],degrees=True) *R_ci).as_euler("XYZ",degrees=True)
                # motion_data_T_frame[motion_data_ptr:motion_data_ptr+3] = (R_pi.inv()*R.from_euler("XYZ",motion_data_A_frame[motion_data_ptr:motion_data_ptr+3],degrees=True) *R_i).as_euler("XYZ",degrees=True)

                motion_data_ptr += 3

    
    return motion_data_T