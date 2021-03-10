import numpy as np
import torch
from kpt.model.kinematic_model import KinematicModel
from pytorch3d.transforms import euler_angles_to_matrix


class RobotModel:

    def __init__(self, model: KinematicModel):
        self.model = model # Kinematic model.
        self._kinematic_chain = {} # Should be set from `set_frame`

    def set_frame(self, frame_id: int):
        """Kinematic chain and motion sequence should be set by this method.

        Args:
            frame_id (int): [description]
        """

        if self.model.type is 'bvh':
           self._kinematic_chain = self.model.get_kinematic_chain()
           root_pos, root_rot = self.model.get_root_pos_rot(frame_id)
        else:
            raise ValueError('Undefined model type')

        self._kinematic_chain[self.root_name]['p'] = root_pos
        self._kinematic_chain[self.root_name]['R'] = root_rot

    def get_R_offset(self, joint):
        # Set rotations
        if joint.endswith('Nub'):
            return torch.eye(3)
        rot_cols = [joint + '_' + channel for channel in ['Xrotation', 'Yrotation', 'Zrotation']]
        rot_val = euler_angles_to_matrix(torch.Tensor(self.cur_motion[rot_cols].values/180*np.pi), 'XYZ')
        return rot_val
    
    def forward_kinematics(self, joint_name):
        """Solve forward kinematics from given joint name."""
        if not self._kinematic_chain[joint_name]: # For end of kinematic chain.
            return None
        
        if (joint_name is not self.root_name):
            parent = self._kinematic_chain[joint_name]['parent']
            self._kinematic_chain[joint_name]['R'] = torch.matmul(self._kinematic_chain[parent]['R'], self.get_R_offset(joint_name))
            self._kinematic_chain[joint_name]['p'] = torch.matmul(self._kinematic_chain[parent]['R'], self._kinematic_chain[joint_name]['offsets']) + self._kinematic_chain[parent]['p']
        
        for child_name in self._kinematic_chain[joint_name]['children']:
            self.forward_kinematics(child_name)

    def export_positions(self):
        
        positions = []
        for joint in self._kinematic_chain:
            if not joint.endswith('Nub'):
                positions.append(self._kinematic_chain[joint]['p'].numpy())
        position_arr = np.array(positions)
        return position_arr.squeeze()
    
    def print__kinematic_chain(self, joint_name):
        if joint_name not in self._kinematic_chain.keys():
            raise KeyError("Does not have matching joint in kinematic chain.")

        querying_node = self._kinematic_chain[joint_name]
        print(f"NAME: {joint_name}")
        print(f"Parent: {querying_node['parent']}")
        print(f"Offsets: {querying_node['offsets']}")
        print(f"Children: {querying_node['children']}")
        print(f"Channels: {querying_node['channels']}")
