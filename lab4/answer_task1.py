from bvh_utils import *
import numpy as np
from scipy.spatial.transform import Rotation as R
#---------------你的代码------------------#
def compute_transform_matrix(translation, orientation):
    """
    计算变换矩阵
    输入：
        translation: (3,) ndarray, 平移向量
        orientation: (4,) ndarray, 四元数表示的旋转
    输出：
        transform_matrix: (4,4) ndarray, 变换矩阵
    """
    rot_matrix = R.from_quat(orientation).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    N = T_pose_vertex_translation.shape[0]
    M = joint_translation.shape[0]
    
    # 计算每个关节的变换矩阵
    T_matrices = np.zeros((M, 4, 4))
    T_pose_matrices = np.zeros((M, 4, 4))
    
    for i in range(M):
        T_matrices[i] = compute_transform_matrix(joint_translation[i], joint_orientation[i])
        T_pose_matrices[i] = compute_transform_matrix(T_pose_joint_translation[i], [0, 0, 0, 1])  # T pose下的旋转是单位四元数
    
    # 计算T pose到目标姿态的变换矩阵
    transform_matrices = np.matmul(T_matrices, np.linalg.inv(T_pose_matrices))
    
    # 将顶点转换为齐次坐标
    T_pose_vertex_translation_h = np.hstack((T_pose_vertex_translation, np.ones((N, 1))))
    
    # 批量处理每个顶点
    vertex_translation = np.zeros((N, 3))
    for i in range(4):
        joint_idx = skinning_idx[:, i]
        weight = skinning_weight[:, i][:, np.newaxis]

        if np.any(weight > 0):
            transform = transform_matrices[joint_idx]
            transformed_vertex = np.einsum('ijk,ik->ij', transform, T_pose_vertex_translation_h)
            
            vertex_translation += weight * transformed_vertex[:, :3] / transformed_vertex[:, 3][:, np.newaxis]
    
    return vertex_translation
