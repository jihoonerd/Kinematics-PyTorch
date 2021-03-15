import copy

import numpy as np
import torch
from kpt.model.kinematic_model import KinematicModel
from pymo.parsers import BVHParser
from pytorch3d.transforms import euler_angles_to_matrix


class BVHModel(KinematicModel):
    """Class for parsing BVH and retrieving kinematic information.
    
    # Note that `values` of parsed data is using degree, not radian!
    """

    def __init__(self, bvh_path):
        self.model_type = 'bvh'
        self.bvh_path = bvh_path
        self.parsed = BVHParser().parse(bvh_path)
        self.motion_data = self.parsed.values
        self.framerate = self.parsed.framerate
        self.skeleton = self.parsed.skeleton
        self.joints = list(self.parsed.skeleton.keys())
        self.root_name = self.parsed.root_name
        self.kinematic_chain = self._build_kinematic_chain()

    def _build_kinematic_chain(self):
        """This builds a kinematic chain from skeleton data from parsed BVH.
        
        Returns:
            dict: keys: joint_name names, values: processed skeleton dictionary from parsed BVH
        """        
        kinematic_chain = {}

        for joint_name in self.joints:
            joint_info = copy.deepcopy(self.parsed.skeleton[joint_name]) # Should use deepcopy to preserve original parsing data
            joint_info['offsets'] = torch.Tensor(np.expand_dims(joint_info['offsets'], 1)) # Offsets will have a shape of (3,1)
            if joint_name is self.root_name:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.parsed.skeleton[joint_name]['channels']])[-3:] # Extract rotation channel only.
            else:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.parsed.skeleton[joint_name]['channels']])
            kinematic_chain[joint_name] = joint_info
        return kinematic_chain

    def get_root_pos_rot(self, frame_id: int):
        """This returns root's world coordinate and rotation matrix by given frame_id.

        Args:
            frame_id (int): frame id. Index starts from 0.
        Returns:
            root_position: Position offset (3,1) in world coordinates.
            root_rotation: Rotation matrix (3,3) in world coordinates.
        """
        
        cur_frame = self.motion_data.iloc[frame_id]
        root_position_sr = cur_frame[[self.root_name + '_' + pos for pos in self.skeleton[self.root_name]['channels'][:3]]]
        root_position = torch.Tensor(np.expand_dims(root_position_sr.values, axis=0)).T
        channel_order = self.kinematic_chain[self.root_name]['channel_order']
        root_rotation = self._euler_to_rotation_matrix(frame_id, self.root_name, channel_order, degrees=True)
        return root_position, root_rotation
    
    def get_rotation_matrix(self, frame_id: int, joint_name: str):
        """Returns rotation matrix from given joint at provided frame_id

        Args:
            frame_id (int): frame id. Index starts from 0.
            joint_name (str): joint name should be defined in self.kinematic_chain

        Returns:
            rot_mat: Rotation matrix (3,3) in local coordinates.
        """        
        if not self.skeleton[joint_name]['children']:
            return torch.eye(3)
        channel_order = self.kinematic_chain[joint_name]['channel_order']
        rot_mat = self._euler_to_rotation_matrix(frame_id, joint_name, channel_order, degrees=True)
        return rot_mat

    def _euler_to_rotation_matrix(self, frame_id: int, joint_name: str, channel_order: str, degrees: bool):
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
        cur_frame = self.motion_data.iloc[frame_id]
        rot_cols = [joint_name + '_' + channel for channel in ['Xrotation', 'Yrotation', 'Zrotation']]

        base = torch.zeros((3,3))
        base[0][0] = cur_frame[rot_cols].values[0]
        base[1][1] = cur_frame[rot_cols].values[1]
        base[2][2] = cur_frame[rot_cols].values[2]

        if degrees:
            base = torch.deg2rad(base)

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
