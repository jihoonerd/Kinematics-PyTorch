from kpt.model.kinematic_model import KinematicModel
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms import euler_angles_to_matrix
import numpy as np
import torch

class RobotModel:

    def __init__(self, model: KinematicModel):
        self.model = model
        self.motion_data = model.motion_data
        self._kinematic_chain = model.get_kinematic_chain()
        self.root_name = model.root_name
        self.cur_motion = None

    @property
    def kinematic_chain(self):
        return self._kinematic_chain

    def print_kinematic_chain(self, joint_name):
        if joint_name not in self._kinematic_chain.keys():
            raise KeyError("Does not have matching joint in kinematic chain.")

        querying_node = self._kinematic_chain[joint_name]
        print(f"NAME: {joint_name}")
        print(f"Parent: {querying_node['parent']}")
        print(f"Offsets: {querying_node['offsets']}")
        print(f"Children: {querying_node['children']}")
        print(f"Channels: {querying_node['channels']}")

    def set_frame(self, frame_id: int):
        # set current motion by given frame_id
        self.cur_motion = self.motion_data.iloc[frame_id]
        root_global_position = self.cur_motion[[self.root_name + '_' + pos for pos in self.kinematic_chain[self.root_name]['channels'][:3]]]
        root_global_rotation = self.cur_motion[[self.root_name + '_' + rot for rot in ['Xrotation', 'Yrotation', 'Zrotation']]]
        root_glboal_position_val = torch.Tensor(np.expand_dims(root_global_position.values, axis=0))
        root_rotation_val = euler_angles_to_matrix(torch.Tensor(root_global_rotation.values), 'ZYX')

        # Set root node's position and rotation
        self._kinematic_chain[self.root_name]['p'] = root_glboal_position_val.T
        self._kinematic_chain[self.root_name]['R'] = root_rotation_val


    def get_R_offset(self, joint):
        # Set rotations
        if not joint.endswith('Nub'):
            return torch.eye(3)
        rot_cols = [joint + '_' + channel for channel in ['Xrotation', 'Yrotation', 'Zrotation']]
        rot_val = euler_angles_to_matrix(torch.Tensor(self.cur_motion[rot_cols].values), 'ZYX')
        return rot_val
    
    def forward_kinematics(self, joint_name):
        """Solve forward kinematics from given joint name."""
        if not self._kinematic_chain[joint_name]: # For end of kinematic chain.
            return None
        
        if (joint_name is not self.root_name) and (not joint_name.endswith('Nub')):
            parent = self._kinematic_chain[joint_name]['parent']
            self._kinematic_chain[joint_name]['p'] = torch.matmul(self._kinematic_chain[parent]['R'], self._kinematic_chain[joint_name]['offsets']) + self._kinematic_chain[parent]['p']
            self._kinematic_chain[joint_name]['R'] = torch.matmul(self._kinematic_chain[parent]['R'], self.get_R_offset(joint_name))
        
        for child_name in self._kinematic_chain[joint_name]['children']:
            self.forward_kinematics(child_name)

    def export_positions(self):
        
        positions = []
        for joint in self.kinematic_chain:
            if not joint.endswith('Nub'):
                positions.append(self.kinematic_chain[joint]['p'].numpy())
        position_arr = np.array(positions)
        return position_arr.squeeze()