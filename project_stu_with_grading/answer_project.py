# 以下部分均为可更改部分，你可以把需要的数据结构定义进来
from typing import List
from bvh_loader import BVHMotion
from scipy.spatial.transform import Rotation as R
from physics_warpper import PhysicsInfo
import numpy as np
import smooth_utils

def calc_root_state(motion):
    '''
        Compute the root position, root rotation, root velocity, root angular velocity
    '''
    root_pos = motion.joint_position[:,0,:]
    root_rotation = motion.joint_rotation[:,0,:]
    root_vel = (root_pos[1:,:] - root_pos[:-1,:])/(1.0/60.0)
    root_avel = smooth_utils.quat_to_avel(motion.joint_rotation[:,0,:], 1.0/60.0)
    return root_pos, root_rotation, root_vel, root_avel

def decompose_rotation_with_yaxis(rotations):
    '''
    输入: rotations 形状为(..., 4)的ndarray, 四元数旋转
    输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
    '''
    rot = R.from_quat(rotations)
    matrices = rot.as_matrix()
    y_axes = matrices[..., :, 1]
    global_y = np.array([0, 1, 0])
    dot_product = np.einsum('...i,i->...', y_axes, global_y)
    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
    axes = np.cross(y_axes, global_y)
    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    axes = axes / np.where(norms == 0, 1, norms)  # 避免除以零
    rot_vecs = axes * angles[..., np.newaxis]
    rot_inv = R.from_rotvec(rot_vecs)
    Ry = (rot_inv * rot).inv()
    Rxz = Ry.inv() * rot
    
    return Ry.as_quat(), Rxz.as_quat()

def cal_pos_offset(source,target):
    """计算从source到target的XoZ平面位置偏移

    Args:
        source : 三维位置
        target : 三维位置
        返回的是（-1，3）的数组
    """
    ret = target - source
    ret[...,1] = 0
    return ret

def cal_rot_offset(source, target):
    """计算从source到target的Y轴旋转

    Args:
        source : (..., 4) 旋转quat数组
        target : (..., 4) 旋转quat数组

    返回的是 (..., 4) 的数组
    """
    # 保留输入的形状以便返回时恢复
    original_shape = source.shape
    batch_size = source.shape[:-1]

    # 将输入reshape为二维数组，方便批量处理
    source_reshape = source.reshape(-1, 4)
    target_reshape = target.reshape(-1, 4)

    # 计算 source 和 target 的朝向
    source_facing_direction_xz = R.from_quat(source_reshape).apply(np.array([0, 0, 1]))[:, [0, 2]]
    target_facing_direction_xz = R.from_quat(target_reshape).apply(np.array([0, 0, 1]))[:, [0, 2]]
    
    # 计算绕Y轴的旋转角度
    source_angle = np.arctan2(source_facing_direction_xz[:, 0], source_facing_direction_xz[:, 1])
    target_angle = np.arctan2(target_facing_direction_xz[:, 0], target_facing_direction_xz[:, 1])
    
    # 生成绕Y轴的旋转矩阵
    axis = np.array([0, 1, 0])

    original_rot = R.from_rotvec(axis[np.newaxis] * source_angle[:, np.newaxis])
    desired_rot = R.from_rotvec(axis[np.newaxis] * target_angle[:, np.newaxis])
    
    # 计算旋转差异
    rot_diff = desired_rot * original_rot.inv()
    
    # 恢复原始形状
    result = rot_diff.as_quat().reshape(*batch_size, 4)
    
    return result

def joint_shift_cost(joint_positions, joint_rotations, joint_velocities,joint_angular_velocities, cur_joint_positions, cur_joint_rotations, cur_joint_velocities, cur_joint_angular_velocities):
    weight = 1
    pos_cost = np.mean(np.linalg.norm(joint_positions[...,1:,:] - cur_joint_positions[...,1:,:], axis=-1), axis=-1)    # 不考虑根节点
    # rotation_cost = np.mean(np.linalg.norm(joint_rotations[...,1:,:] - cur_joint_rotations[...,1:,:], axis=-1), axis=-1)    # 不考虑根节点
    rotation_cost = np.max(np.linalg.norm(joint_rotations[...,1:,:] - cur_joint_rotations[...,1:,:], axis=-1), axis=-1)    # 不考虑根节点
    
    velocity_cost = np.mean(np.linalg.norm(joint_velocities[...,1:,:] - cur_joint_velocities[...,1:,:], axis=-1), axis=-1)    # 不考虑根节点
    angular_velocity_cost = np.max(np.linalg.norm(joint_angular_velocities[...,1:,:] - cur_joint_angular_velocities[...,1:,:], axis=-1), axis=-1)    # 不考虑根节点
    
    return pos_cost[:-1] + rotation_cost[:-1] * 10 + velocity_cost + angular_velocity_cost * 0.8

def velocity_cost(root_pos, root_rotation, root_vel, root_avel, cur_root_pos, cur_root_rotation, cur_root_vel,cur_root_avel):
    rot_offset = cal_rot_offset(root_rotation,cur_root_rotation)
    vel_aligned = R.from_quat(rot_offset[:-1]).apply(root_vel)

    vel_cost = np.linalg.norm(vel_aligned - cur_root_vel,axis=-1)

    avel_aligned = R.from_quat(rot_offset[:-1]).apply(root_avel)

    avel_cost = np.linalg.norm(avel_aligned - cur_root_avel,axis=-1)
    return vel_cost + avel_cost


def desire_cost(root_pos, root_rotation, root_vel, root_avel,desired_pos_list,desired_rot_list,desired_vel_list,desired_avel_list):
    
    frames_considered = root_pos.shape[0] - 101
    key_frame_num = 6

    pos_offset = desired_pos_list[0] - root_pos
    pos_offset[...,1] = 0   # 只考虑XoZ平面

    rot_offset = cal_rot_offset(root_rotation,desired_rot_list[0])
    vel_aligned = R.from_quat(rot_offset[:-1]).apply(root_vel)

    vel_cost = np.zeros_like(vel_aligned[...,:frames_considered,0])

    for i in range(key_frame_num):
        vel_cost += np.linalg.norm(vel_aligned[(20*i):] - desired_vel_list[i],axis=-1)[:frames_considered]
    vel_cost /= key_frame_num

    avel_aligned = R.from_quat(rot_offset[:-1]).apply(root_avel)

    avel_cost = np.zeros_like(avel_aligned[...,:frames_considered,0])
    for i in range(key_frame_num):
        avel_cost += np.linalg.norm(avel_aligned[(20*i):] - desired_avel_list[i],axis=-1)[:frames_considered]
    avel_cost /= key_frame_num
    
    return vel_cost + avel_cost


class CharacterController():
    def __init__(self, viewer, controller, pd_controller) -> None:
        # viewer 类，封装physics
        self.viewer = viewer
        # 手柄/键盘控制器
        self.controller = controller
        # pd controller
        self.pd_controller = pd_controller
        # motion
        self.motions = []
        # 添加motion
        # self.motions.append(BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh'))
        # self.motions.append(BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh').sub_sequence(300,30000))
        run_motion = BVHMotion(bvh_file_name='./motion_material/long_motion/long_run.bvh').sub_sequence(300,5000)
        walk_motion = BVHMotion(bvh_file_name='./motion_material/long_motion/long_walk.bvh').sub_sequence(300,5000)
        
        self.run_frames = run_motion.motion_length
        run_motion.append(walk_motion)
        self.motions.append(run_motion)
        self.raw_motion = run_motion.raw_copy()
        self.smooth_motion = self.raw_motion.raw_copy()  # 用于存储平滑后的motion

        self.raw_motion.adjust_joint_name(self.viewer.joint_name)
        self.smooth_motion.adjust_joint_name(self.viewer.joint_name)
        # self.motions.append(BVHMotion(bvh_file_name='./motion_material/idle.bvh'))
        # self.motions.append(BVHMotion(bvh_file_name='./motion_material/run.bvh'))
        # self.motions.append(BVHMotion(bvh_file_name='./motion_material/walk.bvh'))
        
        # 下面是你可能会需要的成员变量，只是一个例子形式
        # 当然，你可以任意编辑，来符合你的要求
        self.cur_pos = None
        self.cur_rot = None

        self.cur_root_pos = None
        
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        
        self.raw_motion_velocity = self.raw_motion.joint_position[1:] - self.raw_motion.joint_position[:-1]
        self.raw_motion_angular_velocity = smooth_utils.quat_to_avel(self.raw_motion.joint_rotation, 1/60)

        self.root_pos, self.root_rotation, self.root_vel, self.root_avel = calc_root_state(self.raw_motion)

    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait=None
                     ):
        '''
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态
            当然我们只是提供了速度和角速度的输入，如果通过pos和rot已经很好选择下一个动作了，可以不必须使用速度和角速度
            desired_pos_list: 期望位置, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望位置(XoZ平面)， 期望位置可以用来拟合根节点位置
            desired_rot_list: 期望旋转, 6x4的矩阵, 四元数, 每一行对应0，20，40...帧的期望旋转(Y旋转), 期望旋转可以用来拟合根节点旋转
            desired_vel_list: 期望速度, 6x3的矩阵, [x, 0, z], 每一行对应0，20，40...帧的期望速度(XoZ平面), 期望速度可以用来拟合根节点速度
            desired_avel_list: 期望角速度, 6x3的矩阵, [0, y, 0], 每一行对应0，20，40...帧的期望角速度(Y旋转), 期望角速度可以用来拟合根节点角速度
        
        Output: 输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            1. 注意应该利用的期望位置和期望速度应该都是在XoZ平面内，期望旋转和期望角速度都是绕Y轴的旋转。其他的项没有意义

        '''
        motion_length = self.raw_motion.motion_length

        joint_name = self.raw_motion.joint_name

        if current_gait:
            desired_vel_list *= 0.3

        if (self.cur_frame+1) % 15 == 0:
            # pre_pos = self.cur_pos
            # cur_pos = self.smooth_motion.joint_position[self.cur_frame]

            # pre_rot = self.cur_rot
            # cur_rot = self.smooth_motion.joint_rotation[self.cur_frame]

            pre_pos = self.raw_motion.joint_position[self.cur_frame-1].copy()
            cur_pos = self.raw_motion.joint_position[self.cur_frame].copy()

            pre_rot = self.raw_motion.joint_rotation[self.cur_frame-1].copy()
            cur_rot = self.raw_motion.joint_rotation[self.cur_frame].copy()


            cur_vel = (cur_pos - pre_pos) / (1/60)
            cur_avel = smooth_utils.quat_to_avel([pre_rot,cur_rot],1/60)[0]



            sh_cost = joint_shift_cost(self.raw_motion.joint_position, self.raw_motion.joint_rotation,self.raw_motion_velocity,self.raw_motion_angular_velocity, cur_pos, cur_rot, cur_vel, cur_avel)
            de_cost = desire_cost(self.root_pos, self.root_rotation, self.root_vel, self.root_avel,desired_pos_list,desired_rot_list,desired_vel_list,desired_avel_list) 
            ve_cost = velocity_cost(self.root_pos, self.root_rotation, self.root_vel, self.root_avel, cur_pos[0], cur_rot[0], cur_vel[0], cur_avel[0])
            
            frames_considered = self.raw_motion.joint_position.shape[0] - 101

            min_de_cost = np.min(de_cost[:frames_considered])

            # select the frame with the lowest sh_cost[:frames_considered] + ve_cost[:frames_considered] with de_cost[:frames_considered] < 1.2 * min_de_cost
            good_frames = np.where(de_cost[:frames_considered] < 1.5 * min_de_cost)[0]
            
            if current_gait:
                # select the frame id > run_frames
                good_frames = good_frames[good_frames > self.run_frames]
                
            
            cost = sh_cost[good_frames] + ve_cost[good_frames]

            best_frame_index = np.argmin(cost)
            best_frame = good_frames[best_frame_index]

            best_cost = cost[best_frame_index]

            # if best_cost > 50:
            #     breakpoint()

            if np.abs(best_frame - self.cur_frame) > 60:

                new_pos = self.raw_motion.joint_position[best_frame].copy()
                new_rot = self.raw_motion.joint_rotation[best_frame].copy()

                new_vel = (self.raw_motion.joint_position[best_frame+1] - self.raw_motion.joint_position[best_frame]) / (1/60)
                new_avel = smooth_utils.quat_to_avel(self.raw_motion.joint_rotation[best_frame:best_frame+2],1/60)[0]

                # align cur_frame to best_frame
                cur_pos[0][[0,2]] = new_pos[0][[0,2]] # align root position
                y_rot_offset = cal_rot_offset(cur_rot[0],new_rot[0])
                cur_rot[0] = (R.from_quat(y_rot_offset)*R.from_quat(cur_rot[0])).as_quat()  # align root rotation
                cur_vel[0] = R.from_quat(y_rot_offset).apply(cur_vel[0])  # align root velocity
                cur_avel[0] = R.from_quat(y_rot_offset).apply(cur_avel[0])  # align root angular velocity
                d_pos = cur_pos - new_pos
                d_rot = R.from_quat(cur_rot).as_rotvec()-R.from_quat(new_rot).as_rotvec()
                d_vel = cur_vel - new_vel
                d_avel = cur_avel - new_avel

                frames_smoothed = 100

                self.smooth_motion.joint_position[best_frame:best_frame+frames_smoothed] = self.raw_motion.joint_position[best_frame:best_frame+frames_smoothed]
                self.smooth_motion.joint_rotation[best_frame:best_frame+frames_smoothed] = self.raw_motion.joint_rotation[best_frame:best_frame+frames_smoothed]

                half_life = 0.2
                for i in range(frames_smoothed):
                    offset_pos,_=smooth_utils.decay_spring_implicit_damping_pos(d_pos,d_vel,half_life,i*(1/60))
                    offset_rot,_=smooth_utils.decay_spring_implicit_damping_rot(d_rot,d_avel,half_life,i*(1/60))


                    self.smooth_motion.joint_position[best_frame+i] += offset_pos                
                    self.smooth_motion.joint_rotation[best_frame+i] = R.from_rotvec(R.from_quat(self.smooth_motion.joint_rotation[best_frame+i]).as_rotvec()+offset_rot).as_quat()

                self.cur_frame = best_frame
                print("change frame to ", best_frame)

            # joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list=[self.cur_frame],root_pos=desired_pos_list[0], root_quat=desired_rot_list[0])
        
        # if (self.cur_frame+1) % 15 == 0:
        #     print("position_diff:",(self.smooth_motion.joint_position[self.cur_frame][1:]- self.cur_pos[1:]).mean())
        #     print("rot_diff:",(self.smooth_motion.joint_rotation[self.cur_frame][1:]- self.cur_rot[1:]).mean())
        self.cur_pos = self.smooth_motion.joint_position[self.cur_frame]
        self.cur_rot = self.smooth_motion.joint_rotation[self.cur_frame]

        motion_root_positon = self.smooth_motion.joint_position[self.cur_frame][0]
        motion_root_rotation = self.smooth_motion.joint_rotation[self.cur_frame][0]

        pos_offset = cal_pos_offset(motion_root_positon,desired_pos_list[0])
        rot_offset = cal_rot_offset(motion_root_rotation,desired_rot_list[0])

        joint_translation, joint_orientation = self.smooth_motion.batch_forward_kinematics(frame_id_list=[self.cur_frame],root_pos=motion_root_positon+pos_offset, root_quat=(R.from_quat(rot_offset) *R.from_quat(motion_root_rotation)).as_quat() )
        # joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list=[self.cur_frame],root_pos=motion_root_positon+pos_offset, root_quat=desired_rot_list[0] )
        
        joint_translation = joint_translation[0]
        joint_orientation = joint_orientation[0]
        old_root_pos = self.cur_root_pos

        if current_gait:
            joint_translation[0] = (joint_translation[0]-old_root_pos)*0.3+old_root_pos

        self.cur_root_pos = joint_translation[0]
        
        self.cur_frame = (self.cur_frame + 1) % motion_length
        if self.cur_frame >= self.raw_motion.num_frames:
            self.cur_frame = 0
        return joint_name, joint_translation, joint_orientation
    

    def sync_controller_and_character(self, character_state):
        '''
        这一部分用于同步手柄和你的角色的状态
        更新后很有可能会出现手柄和角色的位置不一致，你可以按需调整
        '''
        controller_pos = character_state[1][0]
        self.controller.set_pos(controller_pos)
    




class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pass

    def apply_root_force_and_torque(self):
        pass
    
    def apply_static_torque(self):
        pass