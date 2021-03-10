import numpy as np
import torch
from kpt.model.kinematic_model import KinematicModel


class RobotModel:

    def __init__(self, kinematic_model: KinematicModel):
        self.kinematic_model = kinematic_model
        self.kinematic_chain = self.kinematic_model.kinematic_chain
        self.frame_id = None

    def set_frame(self, frame_id: int):
        """Kinematic chain and motion sequence should be set by this method.

        Args:
            frame_id (int): [description]
        """
        self.frame_id = frame_id
        if self.kinematic_model.model_type == 'bvh':
           root_pos, root_rot = self.kinematic_model.get_root_pos_rot(frame_id)
        else:
            raise ValueError('Undefined model type')

        self.kinematic_chain[self.kinematic_model.root_name]['p'] = root_pos
        self.kinematic_chain[self.kinematic_model.root_name]['R'] = root_rot
        
    
    def forward_kinematics(self, joint_name):
        """Solve forward kinematics from given joint name."""
        if not self.kinematic_chain[joint_name]: # For end of kinematic chain.
            return None
        
        if (joint_name is not self.kinematic_model.root_name):
            parent = self.kinematic_chain[joint_name]['parent']
            self.kinematic_chain[joint_name]['p'] = torch.matmul(self.kinematic_chain[parent]['R'], self.kinematic_chain[joint_name]['offsets']) + self.kinematic_chain[parent]['p']
            self.kinematic_chain[joint_name]['R'] = torch.matmul(self.kinematic_chain[parent]['R'], self.kinematic_model.get_rotation_matrix(self.frame_id, joint_name))
        
        for child_name in self.kinematic_chain[joint_name]['children']:
            self.forward_kinematics(child_name)

    def export_positions(self):
        positions = []
        for joint in self.kinematic_chain:
            positions.append(self.kinematic_chain[joint]['p'].numpy())
        position_arr = np.array(positions)
        return position_arr.squeeze()
    
    def print_kinematic_chain(self, joint_name):
        if joint_name not in self.kinematic_chain.keys():
            raise KeyError("Does not have matching joint in kinematic chain.")

        querying_node = self.kinematic_chain[joint_name]
        print(f"NAME: {joint_name}")
        print(f"Parent: {querying_node['parent']}")
        print(f"Offsets: {querying_node['offsets']}")
        print(f"Children: {querying_node['children']}")
        print(f"Channels: {querying_node['channels']}")
