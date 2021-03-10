import copy

import numpy as np
import torch
from kpt.model.kinematic_model import KinematicModel
from pymo.parsers import BVHParser


class BVHModel(KinematicModel):
    """Class for parsing BVH and retrieving kinematic information."""    

    def __init__(self, bvh_path):
        self.model_type = 'bvh'
        self.bvh_path = bvh_path
        self.parsed = BVHParser().parse(bvh_path)
        self.motion_data = self.parsed.values
        self.framerate = self.parsed.framerate
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

