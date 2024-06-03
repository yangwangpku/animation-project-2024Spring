import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose: (20, 4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs: 指定参数，可能包含kp,kd
    输出： global_torque: (20, 3)的numpy数组，表示每个关节的全局坐标下的目标力矩，因为不需要控制方向，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    #首先根据每个关节到跟关节的距离来确定kp和kd
    parent_index = physics_info.parent_index # len(parent_index) =len(joint_name) = 20
    joint_name = physics_info.joint_name
    kp = np.zeros(len(joint_name))
    kd = np.zeros(len(joint_name))

    kp = kargs.get('kp', 500)  # 如果没有传入kp，默认为500
    kd = kargs.get('kd', 20)   # 如果没有传入kd，默认为20

    kp = np.ones(len(joint_name)) * kp
    kd = np.ones(len(joint_name)) * kd

    # 一组效果不错的kp和kd值

    depth2parent = np.zeros(len(parent_index))
    for i in range(len(parent_index)):
        j = i
        cnt = 0
        while parent_index[j] != -1:
            j = parent_index[j]
            cnt += 1
        depth2parent[i] = cnt

    kp_ref = np.array([600, 800, 800, 800, 600, 400, 400])
    kd_ref = np.array([100, 30, 15, 10, 8, 5, 5]) * 1.1
    for i in range(len(joint_name)):
        kp[i] = kp_ref[int(depth2parent[i])]
        kd[i] = kd_ref[int(depth2parent[i])]


    # 注意关节没有自己的朝向和角速度，这里用子body的朝向和角速度表示此时关节的信息
    joint_orientation = physics_info.get_body_orientation()
    # print(physics_info.get_root_pos_and_vel())
    parent_index = physics_info.parent_index
    joint_avel = physics_info.get_body_angular_velocity()

    global_torque = np.zeros((20,3))

    for i in range(0,len(joint_orientation)):  # 跳过根节点
        if i != physics_info.root_idx:
            parent_orientation = R.from_quat(joint_orientation[parent_index[i]])
        else:
            # set parent_orientation to identity
            parent_orientation = R.from_quat([0, 0, 0, 1])
        # 当前关节朝向（四元数）
        current_orientation = parent_orientation.inv() * R.from_quat(joint_orientation[i])
        # 目标关节朝向（四元数）
        target_orientation = R.from_quat(pose[i])

        # 计算当前与目标的旋转误差（四元数）
        error_orientation = target_orientation * current_orientation.inv()
        # 将旋转误差转换为旋转矢量
        error_angle = error_orientation.as_rotvec()

        # 计算PD控制力矩
        torque = parent_orientation.apply(error_angle) * kp[i] + kd[i] * (-joint_avel[i])

        global_torque[i] += torque

    # 对力矩进行剪裁以避免过大的力矩
    max_torque = 200  # 根据需要调整最大力矩值
    global_torque = np.clip(global_torque, -max_torque, max_torque)

    return global_torque


def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力，在后续仿真中只会保留y方向的力
           global_root_torque: (3,)的numpy数组，表示根节点的全局坐标下的辅助力矩，用来控制角色的朝向，实际上就是global_torque的第0项
           global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
        3. 可以适当将根节点目标位置上提以产生更大的辅助力，使角色走得更自然
    '''

    
    # ------一些提示代码，你可以随意修改------------#
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 4000) # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 200)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    global_root_force = kp * (target_position - root_position) - kd * root_velocity
    global_root_torque = global_torque[0]
    return global_root_force, global_root_torque, global_torque

frame_cnt = 0


def calculate_com(physics_info):
    '''计算重心位置和速度'''
    body_positions = physics_info.get_body_position()
    body_velocities = physics_info.get_body_velocity()
    body_masses = physics_info.get_body_mass()

    total_mass = 0
    com = np.zeros(3)
    com_velocity = np.zeros(3)
    for i in range(len(body_positions)):
        mass = body_masses[i]
        total_mass += mass
        com += mass * body_positions[i]
        com_velocity += mass * body_velocities[i]
    com /= total_mass
    com_velocity /= total_mass
    return com, com_velocity

def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    输出： 带反馈的global_torque: (20, 3)的numpy数组，因为不需要控制方向，根节点力矩会被无视
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均，但要注意角色还会收到一个从背后推他的外力
        可以定义一个全局的frame_count变量来标记当前的帧数，在站稳后根据帧数使角色进行周期性左右摇晃，如果效果好可以加分（0-20），可以考虑让角色站稳后再摇晃
    '''
    # ------一些提示代码，你可以随意修改------------#
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    
    joint_positions = physics_info.get_joint_translation()
    # 适当前移，这里的权重需要你自己调整
    tar_pos = joint_positions[7] * 0.5 + joint_positions[8] * 0.5
    global frame_cnt
    # # 修改根节点目标位置让角色摇摆起来，可以角色先站稳之后再周期性摇摆
    # tar_pos += 0.1 * np.sin((frame_cnt)/400)
    # frame_cnt += 1
    # # 抹去根节点力矩
    # torque = np.zeros((20,3))

    # 获取当前的重心位置和速度
    joint_velocity = physics_info.get_body_velocity()
    joint_mass = physics_info.get_body_mass()
    com = np.zeros(3)
    com_velocity = np.zeros(3)
    mass = 0
    for i in range(len(joint_mass)):
        com += joint_mass[i] * joint_positions[i]
        com_velocity = joint_mass[i] * joint_velocity[i]
        mass += joint_mass[i]
    com /= mass
    com_velocity /= mass

    # 设置PD控制器参数
    kp = 50
    kd = 10

    # 计算重心位置和目标位置之间的差异，以及虚拟力
    com_displacement = tar_pos - com
    virtual_force = kp * com_displacement - kd * com_velocity

    # 初始化总力矩数组
    torque = part1_cal_torque(pose, physics_info)

    # 使用Jacobian转置来计算每个关节的力矩
    for i in range(3, 8):  # 跳过根节点
        # 获取该关节的Jacobian转置
        # J_transpose = physics_info.get_joint_jacobian_transpose(i)

        # 计算并更新关节的力矩
        # torque[i] += np.dot(com - joint_positions[i], virtual_force)
        torque [i] -= np.cross(com - joint_positions[i], virtual_force)

    # 根据帧计数添加周期性摇摆
    tar_pos += 0.1 * np.sin(frame_cnt / 400)
    frame_cnt += 1

    max_torque = 200
    
    return np.clip(torque, -max_torque, max_torque)