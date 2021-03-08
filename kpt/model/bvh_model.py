from pymo.parsers import BVHParser
from kpt.model.kinematic_model import KinematicModel
import torch
import numpy as np


class BVHModel(KinematicModel):

    def __init__(self, bvh_path, device='cpu'):
        self.model_type = 'bvh'
        self.bvh_path = bvh_path
        self.device = device
        self.bvh_parsed = BVHParser().parse(bvh_path)
        self.motion_data = self.bvh_parsed.values
        self.joints = list(self.bvh_parsed.skeleton.keys())
        self.root_name = self.bvh_parsed.root_name

    def get_kinematic_chain(self):
        """Generate kinematic chain from parsed bvh"""
        kinematic_chain = {}

        for joint in self.joints:
            joint_info = self.bvh_parsed.skeleton[joint]
            joint_info['offsets'] = torch.Tensor(np.expand_dims(joint_info['offsets'], 1))
            if joint is self.root_name:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.bvh_parsed.skeleton[joint]['channels']])[-3:]
            else:
                joint_info['channel_order'] = ''.join([channel[0] for channel in self.bvh_parsed.skeleton[joint]['channels']])
            kinematic_chain[joint] = joint_info

        return kinematic_chain

