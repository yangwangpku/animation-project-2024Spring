##############
# 姓名：
# 学号：
##############
# 以下部分均为可更改部分，你可以把需要的数据结构定义进来，可以继承自Graph class
from graph import *
from answer_task1 import *
from typing import List
from bvh_motion import BVHMotion
from scipy.spatial.transform import Rotation as R

class CharacterController():
    def __init__(self, controller) -> None:
        # 手柄/键盘控制器
        self.controller = controller
        # 读取graph结构
        self.graph = Graph('./nodes.npy')
        self.graph.load_from_file()
        # node name组成的List
        self.node_names = [nd.name for nd in self.graph.nodes]
        # edge name组成的List
        self.edge_names = []
        for nd in self.graph.nodes:
            for eg in nd.edges:
                self.edge_names.append(eg.label)

        # 下面是你可能会需要的成员变量，只是一个例子形式
        # 当然，你可以任意编辑，来符合你的要求
        # 当前角色的参考root位置
        self.cur_root_pos = None
        # 当前角色的参考root旋转
        self.cur_root_rot = None

        # 当前角色处于Graph的哪一个节点
        self.cur_node : Node = None
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge : Edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = -1

        # 当前角色正在进行的motion
        self.cur_motion : BVHMotion = None

        # 初始化上述参数
        self.initialize()
        
    def initialize(self):
        # 当前角色处于Graph的哪一个节点
        self.cur_node = self.graph.nodes[0]
        # 当前角色选择了哪一条边(选择要切换成哪一个新的动作)
        self.cur_edge = None
        # 当前角色处于正在跑的BVH的第几帧
        self.cur_frame = 0
        # 当前角色对应的BVH的结束的帧数
        self.cur_end_frame = self.cur_node.motion.motion_length
        
        # 当前角色的参考root位置
        self.cur_root_pos = self.cur_node.motion.joint_position[0,0,:].copy()
        self.cur_root_pos[1] = 0 # 忽略竖直方向，即y方向的位移
        
        # 当前角色的参考root旋转
        self.cur_root_rot, _ = BVHMotion.decompose_rotation_with_yaxis(self.cur_node.motion.joint_rotation[0, 0])
    
        self.cur_motion = self.cur_node.motion

    def select_next_motion(self, desired_pos_list, desired_rot_list):
        # compute the difference between the current root rotation and the desired rotation
        cur_rot = R.from_quat(self.cur_root_rot)
        desired_rot = R.from_quat(desired_rot_list[-1])
        diff_rot = cur_rot.inv() * desired_rot

        # compute the angle to rotate to the desired rotation
        angle = diff_rot.as_rotvec()[1]

        # > 135 degree, spin clockwise
        if angle > 2.356:
            return self.graph.nodes[4]
        # 45 - 135 degree, turn right
        elif angle > 0.785:
            return self.graph.nodes[1]
        # -45 - 45 degree, go straight
        elif angle > -0.785:
            return self.graph.nodes[0]
        # -135 - -45 degree, turn left
        elif angle > -2.356:
            return self.graph.nodes[2]
        # < -135 degree, spin counterclockwise
        else:
            return self.graph.nodes[3]


    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list
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
        # continue the current motion if the current motion is not finished
        if self.cur_frame < self.cur_end_frame:
            joint_name = self.cur_motion.joint_name
            joint_translation, joint_orientation = self.cur_motion.batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            
            # update your representation of the character
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
            self.cur_frame += 1
            return joint_name, joint_translation, joint_orientation
        
        else:
            # select the next motion
            next_node = self.select_next_motion(desired_pos_list, desired_rot_list)
            self.cur_node = next_node
            self.cur_frame = 0
            self.cur_end_frame = next_node.motion.motion_length
            joint_name = self.cur_node.motion.joint_name

            facing_axis = R.from_quat(self.cur_root_rot).apply(np.array([0,0,1])).flatten()[[0,2]]
            next_motion = self.cur_node.motion.translation_and_rotation(0, self.cur_root_pos[[0,2]], facing_axis)

            smoothed_motion = concatenate_two_motions(self.cur_motion, next_motion, self.cur_motion.motion_length, 10, 1.0)
            smoothed_motion.joint_position = smoothed_motion.joint_position[self.cur_motion.motion_length:,:,:]
            smoothed_motion.joint_rotation = smoothed_motion.joint_rotation[self.cur_motion.motion_length:,:,:]
            self.cur_motion = smoothed_motion

            joint_translation, joint_orientation = self.cur_motion.batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            
            # 更新你的表示角色的信息
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]


        return joint_name, joint_translation, joint_orientation
