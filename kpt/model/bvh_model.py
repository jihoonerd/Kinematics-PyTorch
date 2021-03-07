from pymo.parsers import BVHParser
import torch
import os


class BVHModel:

    def __init__(self, bvh_path, device='cpu'):
        self.model_type = 'bvh'
        self.bvh_path = bvh_path
        self.device = device
        self.bvh_parsed = BVHParser().parse(bvh_path)
        self.joints = list(self.bvh_parsed.skeleton.keys())

    def get_kinematic_chain(self):
        """Generate kinematic chain from parsed bvh"""
        kinematic_chain = {}

        for joint in self.joints:
            joint_info = self.bvh_parsed.skeleton[joint]
            joint_info['offsets'] = torch.Tensor(joint_info['offsets'])
            joint_info['channel_order'] = ''.join([channel[0] for channel in self.bvh_parsed.skeleton[joint]['channels']])
            
            kinematic_chain[joint] = joint_info

        return kinematic_chain

