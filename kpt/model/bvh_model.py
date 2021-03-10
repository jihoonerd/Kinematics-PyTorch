import copy

import numpy as np
import torch
from kpt.model.kinematic_model import KinematicModel
from pymo.parsers import BVHParser
from pytorch3d.transforms import euler_angles_to_matrix


class BVHModel(KinematicModel):
    """Class for parsing BVH and retrieving kinematic information."""    

    def __init__(self, bvh_path):
        self.model_type = 'bvh'
        self.bvh_path = bvh_path
        self.parsed = BVHParser().parse(bvh_path)
        self.motion_data = self.parsed.values
        self.framerate = self.parsed.framerate
        self.skeleton = self.parsed.skeleton
        self.joints = list(self.parsed.skeleton.keys())
        self.root_name = self.parsed.root_name

    def get_kinematic_chain(self):
        """This builds a kinematic chain from skeleton data from parsed BVH.
        
        Returns:
            dict: keys: joint names, values: processed skeleton dictionary from parsed BVH
        """        
        kinematic_chain = {}

        for joint in self.joints:
            joint_info = copy.deepcopy(self.parsed.skeleton[joint]) # Should use deepcopy to preserve original parsing data
            joint_info['offsets'] = torch.Tensor(np.expand_dims(joint_info['offsets'], 1)) # Offsets will have a shape of (3,1)
            if joint is self.root_name:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.parsed.skeleton[joint]['channels']])[-3:] # Extract rotation channel only.
            else:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.parsed.skeleton[joint]['channels']])
            kinematic_chain[joint] = joint_info
        return kinematic_chain
    
    def get_root_pos_rot(self, frame_id: int, channel_order: str):
        """This returns frame_id's world coordinate and rotation matrix in Tensor.

        Args:
            frame_id (int): frame id. Index starts from 0.
        Returns:
            root_position: Position offset (3,1) in world coordinates.
            root_rotation: Rotation matrix in world coordinates.
        """

        cur_frame = self.motion_data.iloc[frame_id]
        root_position_sr = cur_frame[[self.root_name + '_' + pos for pos in self.skeleton[self.root_name]['channels'][:3]]]
        root_position = torch.Tensor(np.expand_dims(root_position_sr.values, axis=0)).T

        root_global_rotation_sr = cur_frame[[self.root_name + '_' + rot for rot in ['Xrotation', 'Yrotation', 'Zrotation']]]
        root_rotation = euler_angles_to_matrix(torch.Tensor(root_global_rotation_sr.values), channel_order)

        return root_position, root_rotation

