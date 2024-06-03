from bvh_utils import *
from scipy.spatial.transform import Rotation as R

#---------------你的代码------------------#

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    return q / norm

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_to_dual_quaternion(translation, quaternion):
    dual_part = 0.5 * quaternion_multiply(np.array([0, *translation]), quaternion)
    return np.concatenate([quaternion, dual_part])

def dual_quaternion_dot(dq1, dq2):
    real1, dual1 = dq1[:4], dq1[4:]
    real2, dual2 = dq2[:4], dq2[4:]
    return np.dot(real1, real2)

def dual_quaternion_normalize(dq):
    real, dual = dq[:4], dq[4:]
    real = normalize_quaternion(real)
    dual = dual - np.dot(real, dual) * real
    return np.concatenate([real, dual])

def dual_quaternion_multiply(dq1, dq2):
    real1, dual1 = dq1[:4], dq1[4:]
    real2, dual2 = dq2[:4], dq2[4:]
    real = quaternion_multiply(real1, real2)
    dual = quaternion_multiply(real1, dual2) + quaternion_multiply(dual1, real2)
    return np.concatenate([real, dual])

def dual_quaternion_conjugate(dq):
    real, dual = dq[:4], dq[4:]
    real_conj = quaternion_conjugate(real)
    dual_conj = quaternion_conjugate(dual)
    return np.concatenate([real_conj, -dual_conj])

def dual_quaternion_apply(dq, p):
    p_dq = np.array([1,0,0,0,0, p[0], p[1], p[2]])
    applied_dq = dual_quaternion_multiply(dual_quaternion_multiply(dq, p_dq),dual_quaternion_conjugate(dq)) 
    return applied_dq[5:]

def dual_quaternion_interpolation(dq1, dq2, t):
    if dual_quaternion_dot(dq1, dq2) < 0.0:
        dq2 = -dq2

    interpolated_dq = (1 - t) * dq1 + t * dq2
    return dual_quaternion_normalize(interpolated_dq)
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

# translation 和 orientation 都是全局的
def dq_skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
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

    transform_dq = np.zeros((M, 8))
    for i in range(M):
        translation = transform_matrices[i][:3, 3]
        rotation = R.from_matrix(transform_matrices[i][:3, :3]).as_quat()
        # [x, y, z, w] -> [w, x, y, z]
        rotation = np.array([rotation[3], rotation[0], rotation[1], rotation[2]])
        transform_dq[i] = (quaternion_to_dual_quaternion(translation, rotation))
        assert(np.allclose(dual_quaternion_normalize(transform_dq[i]), transform_dq[i]))
    # 计算每个顶点上的dq
    dq = np.zeros((N, 8))
    for i in range(N):
        accumulated_weight = 0
        dq[i] = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        for j in range(4):
            joint_idx = skinning_idx[i, j]
            weight = skinning_weight[i, j]

            # if joint_idx == 61 and weight > 0.3:
            #     breakpoint()

            if weight > 0:
                t = weight / (accumulated_weight + weight)
                dq[i] = dual_quaternion_interpolation(dq[i], transform_dq[joint_idx], t)
                accumulated_weight += weight

    # 在dq下计算顶点位置
    vertex_translation = np.zeros((N, 3))
    for i in range(N):
        vertex_translation[i] = dual_quaternion_apply(dq[i], T_pose_vertex_translation[i])

    return vertex_translation