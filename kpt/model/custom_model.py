import copy

import numpy as np
import torch
from kpt.model.kinematic_model import KinematicModel
from pymo.parsers import BVHParser
from pytorch3d.transforms import euler_angles_to_matrix
from scipy.spatial.transform import Rotation


class CustomModel(KinematicModel):
    """Custom Model Class"""

    def __init__(self, skeleton, motion_data, channel_name):
        self.model_type = 'custom'
        self.skeleton = copy.deepcopy(skeleton)
        self.joints = list(self.skeleton.keys())
        self.motion_data = motion_data
        self.channel_name = channel_name
        self.root_name = self.joints[0]
        self.kinematic_chain = self._build_kinematic_chain()

    def _build_kinematic_chain(self):   
        kinematic_chain = {} 
        for joint_name in self.joints:
            joint_info = self.skeleton[joint_name]
            joint_info['offsets'] = torch.Tensor(np.expand_dims(joint_info['offsets'], 1)) # Offsets will have a shape of (3,1)
            if joint_name is self.root_name:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.skeleton[joint_name]['channels']])[-3:] # Extract rotation channel only.
            else:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.skeleton[joint_name]['channels']])
            kinematic_chain[joint_name] = joint_info
        return kinematic_chain

    def get_root_pos_rot(self, frame_id: int):
        root_position = torch.unsqueeze(self.motion_data[:3], 1)
        channel_order = self.kinematic_chain[self.root_name]['channel_order']
        root_rotation = self._euler_to_rotation_matrix(frame_id, self.root_name, channel_order)
        return root_position, root_rotation      
    
    def get_rotation_matrix(self, frame_id: int, joint_name: str):
        if not self.skeleton[joint_name]['children']:
            return torch.eye(3)
        channel_order = self.kinematic_chain[joint_name]['channel_order']
        rot_mat = self._euler_to_rotation_matrix(frame_id, joint_name, channel_order)
        return rot_mat

    def _euler_to_rotation_matrix(self, frame_id: int, joint_name: str, channel_order: str):
        """Return rotation matrix from given joint_name and frame_id by using self.motion_data

        Args:
            frame_id (int): frame id. Index starts from 0.
            joint_name (str): joint name should be defined in self.kinematic_chain
            channel_order (str): multiplication order. If 'ZYX' is given, it assumes transformed vector will be: [ZYX \times v]

        Raises:
            ValueError: Check channel_order. e.g.) 'XYZ', 'ZYX'...

        Returns:
            rot_mat: transformed rotation matrix (3,3) from euler angle
        """        
        rot_cols = [joint_name + '_' + channel for channel in ['Xrotation', 'Yrotation', 'Zrotation']]
        
        col_idx = []
        for col in rot_cols:
            col_idx.append(self.channel_name.index(col))
        from pytorch3d.transforms import euler_angles_to_matrix
        base = torch.zeros((3,3))
        base[0][0] = self.motion_data[col_idx[0]]
        base[1][1] = self.motion_data[col_idx[1]]
        base[2][2] = self.motion_data[col_idx[2]]
        x_rot = euler_angles_to_matrix(base[0], convention='XYZ')
        y_rot = euler_angles_to_matrix(base[1], convention='XYZ')
        z_rot = euler_angles_to_matrix(base[2], convention='XYZ')

        rot_mat = torch.eye(3)
        for axis in channel_order:
            if axis == 'X' :
                rot_mat = torch.matmul(rot_mat, x_rot)
            elif axis == 'Y':
                rot_mat = torch.matmul(rot_mat, y_rot)
            elif axis == 'Z' :
                rot_mat = torch.matmul(rot_mat, z_rot)
            else:
                raise ValueError(f'Wrong channel order given (Capital Only): {channel_order}')
        return rot_mat
