##############
# 姓名：
# 学号：
##############
"""
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
"""
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from bvh_motion import BVHMotion
from smooth_utils import *

def slerp(q1, q2, alpha):
    '''
    input: q1, q2 two quaternions, shape is (N,4)
              alpha, the interpolation coefficient
    output: the interpolated quaternion, shape is (N,4)
    '''
    # Ensure q1 and q2 are numpy arrays
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    # Initialize an array to store the interpolated quaternions
    interpolated_quaternions = np.empty_like(q1)

    # Interpolate each quaternion individually
    for i in range(q1.shape[0]):
        # Create a times array for the Slerp function
        times = [0, 1]

        # Stack the quaternions to create a (2, 4) array
        quaternions = np.vstack([q1[i], q2[i]])

        # Create a Rotation object from the stacked quaternions
        r = R.from_quat(quaternions)

        # Create a Slerp object for interpolation
        slerp_interpolator = Slerp(times, r)

        # Interpolate using the given alpha value
        interpolated_rotation = slerp_interpolator([alpha])

        # Store the interpolated quaternion
        interpolated_quaternions[i] = interpolated_rotation.as_quat()[0]

    # Return the interpolated quaternions
    return interpolated_quaternions


# part1
def blend_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, v:float=None, input_alpha:np.ndarray=None, target_fps=60) -> BVHMotion:
    '''
    输入: 两个将要blend的动作，类型为BVHMotion
          将要生成的BVH的速度v
          如果给出插值的系数alpha就不需要再计算了
          target_fps,将要生成BVH的fps
    输出: blend两个BVH动作后的动作，类型为BVHMotion
    假设两个动作的帧数分别为n1, n2
    首先需要制作blend 的权重适量 alpha
    插值系数alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    Tips:
        1. 计算速度，两个BVH已经将Root Joint挪到(0.0, 0.0)的XOZ位置上了
        2. 利用v计算插值系数alpha
        3. 线性插值以及Slerp
        4. 可能输入的两个BVH的fps不同，需要考虑
    '''
    # 首先，通过输入bvh的初末hiposition来计算两个BVH的平均v，并根据公式得到新动作帧数n3，混合权重alpha即恒为w1和w2
    n1=bvh_motion1.joint_position.shape[0]
    n2=bvh_motion2.joint_position.shape[0]
    s1=np.linalg.norm(bvh_motion1.joint_position[-1,0]-bvh_motion1.joint_position[0,0])
    s2=np.linalg.norm(bvh_motion2.joint_position[-1,0]-bvh_motion2.joint_position[0,0])
    v1=s1/(n1*bvh_motion1.frame_time)
    v2=s2/(n2*bvh_motion2.frame_time)
    w1=np.clip((v2-v)/(v2-v1),0,1)
    w2=1.0-w1
    
    
    if input_alpha is None:
        n3=int((w1*v1*n1+w2*v2*n2)/v)
        alpha = np.full((n3,), w1) 
        
    else:
        alpha = input_alpha
        n3 = alpha.shape[0] 
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((n3,res.joint_position.shape[1],res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((n3,res.joint_rotation.shape[1],res.joint_rotation.shape[2]))
    res.joint_rotation[...,3] = 1.0
    res.frame_time = 1.0 / target_fps

    for i in range(n3):
        # 对position做线性插值，对rotation做Slerp插值
        res.joint_position[i,:] = alpha[i] * bvh_motion1.joint_position[int(i*n1/n3),:] + (1-alpha[i]) * bvh_motion2.joint_position[int(i*n2/n3),:]
        
        for j in range(bvh_motion1.joint_rotation.shape[1]):
            rot1_joint=R.from_quat(bvh_motion1.joint_rotation[int(i*n1/n3),j])
            rot2_joint=R.from_quat(bvh_motion2.joint_rotation[int(i*n2/n3),j])
            concatenated_rot = R.concatenate([rot1_joint,rot2_joint])
            slerp_joint = Slerp([0, 1], concatenated_rot)
            res.joint_rotation[i][j]=slerp_joint(1-alpha[i]).as_quat()
    
    return res

# part2
def build_loop_motion(bvh_motion:BVHMotion, ratio:float, half_life:float) -> BVHMotion:
    '''
    输入: 将要loop化的动作，类型为BVHMotion
          damping在前在后的比例ratio, ratio介于[0,1]
          弹簧振子damping效果的半衰期 half_life
          如果你使用的方法不含上面两个参数，就忽视就可以了，因接口统一保留
    输出: loop化后的动作，类型为BVHMotion
    
    Tips:
        1. 计算第一帧和最后一帧的旋转差、Root Joint位置差 (不用考虑X和Z的位置差)
        2. 如果使用"inertialization"，可以利用`smooth_utils.py`的
        `quat_to_avel`函数计算对应角速度的差距，对应速度的差距请自己填写
        3. 逐帧计算Rotations和Postions的变化
        4. 注意 BVH的fps需要考虑，因为需要算对应时间
        5. 可以参考`smooth_utils.py`的注释或者 https://theorangeduck.com/page/creating-looping-animations-motion-capture
    
    '''
    res = bvh_motion.raw_copy()
    frames = res.joint_position.shape[0]

    # 计算Root Joint位置差
    root_pos_diff = res.joint_position[-1,0,1] - res.joint_position[0,0,1]   # 只考虑Root Joint Y轴
    
    # 计算Root Joint速度差
    root_vel = (res.joint_position[1:,0] - res.joint_position[:-1,0]) / res.frame_time  # Root Joint 速度
    root_vel_diff = root_vel[-1] - root_vel[0]

    # 计算第一帧和最后一帧的旋转差
    rot_diff = R.from_quat(res.joint_rotation[-1]).as_rotvec() - R.from_quat(res.joint_rotation[0]).as_rotvec()

    # 计算角速度的差距
    avel = quat_to_avel(res.joint_rotation, res.frame_time)
    avel_diff = avel[-1] - avel[0]

    for frame in range(frames):
        # 位置
        pos_front_res, vel_front_res = decay_spring_implicit_damping_pos(root_pos_diff * ratio, root_vel_diff * ratio, half_life, res.frame_time * frame)
        res.joint_position[frame, 0] += pos_front_res

        # 旋转
        rot_front_res, avel_front_res = decay_spring_implicit_damping_rot(rot_diff * ratio, avel_diff * ratio, half_life, res.frame_time * frame)
        res.joint_rotation[frame] = R.from_rotvec(R.from_quat(res.joint_rotation[frame]).as_rotvec() + rot_front_res).as_quat()

    # 末尾
    for frame in range(frames):
        # 位置
        pos_back_res, vel_back_res = decay_spring_implicit_damping_pos(root_pos_diff * (1 - ratio), -root_vel_diff * (1 - ratio), half_life, res.frame_time * frame)
        res.joint_position[frames - frame - 1, 0] -= pos_back_res

        # 旋转
        rot_back_res, avel_back_res = decay_spring_implicit_damping_rot(rot_diff * (1 - ratio), -avel_diff * (1 - ratio), half_life, res.frame_time * frame)
        res.joint_rotation[frames - frame - 1] = R.from_rotvec(R.from_quat(res.joint_rotation[frames - frame - 1]).as_rotvec() - rot_back_res).as_quat()

    return res

# part3
def concatenate_two_motions(bvh_motion1:BVHMotion, bvh_motion2:BVHMotion, mix_frame1:int, mix_time:int, ratio = 0.5):
    '''
    将两个bvh动作平滑地连接起来
    输入: 将要连接的两个动作，类型为BVHMotion
          混合开始时间是第一个动作的第mix_frame1帧
          mix_time表示用于混合的帧数
    输出: 平滑地连接后的动作，类型为BVHMotion
    
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    motion1 = bvh_motion1.sub_sequence(0, mix_frame1)
    # align the facing direction and position of bvh_motion2
    pos = motion1.joint_position[-1,0,[0,2]]
    rot = motion1.joint_rotation[-1,0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    motion2 = bvh_motion2.translation_and_rotation(0, pos, facing_axis)

    # 计算Root Joint位置差
    root_pos_diff = motion1.joint_position[-1,0,1] - motion2.joint_position[0,0,1]   # 只考虑Root Joint Y轴


    # 计算Root Joint速度差
    motion1_vel = (motion1.joint_position[1:,0] - motion1.joint_position[:-1,0]) / motion1.frame_time  # Root Joint 速度
    motion2_vel = (motion2.joint_position[1:,0] - motion2.joint_position[:-1,0]) / motion2.frame_time  # Root Joint 速度
    root_vel_diff = (motion1_vel[-1] - motion2_vel[0])

    # 计算第一帧和最后一帧的旋转差
    rot_diff = R.from_quat(motion1.joint_rotation[-1]).as_rotvec() - R.from_quat(motion2.joint_rotation[0]).as_rotvec()

    # 计算角速度的差距
    motion1_avel = quat_to_avel(motion1.joint_rotation, motion1.frame_time)
    motion2_avel = quat_to_avel(motion2.joint_rotation, motion2.frame_time)
    avel_diff = motion1_avel[-1] - motion2_avel[0]

    motion1_frames = motion1.joint_position.shape[0]
    motion2_frames = motion2.joint_position.shape[0]

    half_life = 0.2


    for frame in range(motion2_frames):
        # 位置
        pos_front_res, vel_front_res = decay_spring_implicit_damping_pos(root_pos_diff * ratio, root_vel_diff * ratio, half_life, motion2.frame_time * frame)
        motion2.joint_position[frame, 0] += pos_front_res
        # 旋转
        rot_front_res, avel_front_res = decay_spring_implicit_damping_rot(rot_diff * ratio, avel_diff * ratio, half_life, motion2.frame_time * frame)
        motion2.joint_rotation[frame] = R.from_rotvec(R.from_quat(motion2.joint_rotation[frame]).as_rotvec() + rot_front_res).as_quat()

    # 末尾
    for frame in range(motion1_frames):
        # 位置
        pos_back_res, vel_back_res = decay_spring_implicit_damping_pos(root_pos_diff * (1 - ratio), -root_vel_diff * (1 - ratio), half_life, motion1.frame_time * (motion1_frames - frame - 1))
        motion1.joint_position[frame, 0] -= pos_back_res

        # 旋转
        rot_back_res, avel_back_res = decay_spring_implicit_damping_rot(rot_diff * (1 - ratio), -avel_diff * (1 - ratio), half_life, motion1.frame_time * (motion1_frames - frame - 1))
        motion1.joint_rotation[frame] = R.from_rotvec(R.from_quat(motion1.joint_rotation[frame]).as_rotvec() - rot_back_res).as_quat()
    

    # 计算Root Joint速度差
    motion1_vel = (motion1.joint_position[1:,0] - motion1.joint_position[:-1,0]) / motion1.frame_time  # Root Joint 速度
    motion2_vel = (motion2.joint_position[1:,0] - motion2.joint_position[:-1,0]) / motion2.frame_time  # Root Joint 速度
    root_vel_diff = (motion1_vel[-1] - motion2_vel[0])

    motion1.append(motion2)
    
    return motion1

