import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix


class TorchSkeleton:
    """Custom Model Class"""

    def __init__(self, skeleton: dict, root_name='Hips', device='cpu'):
        """
        Args:
            skeleton (dict): skeleton should provide 'offesets' 'children' and 'parent' at least.
            root_name (str, optional): name of root joint. Defaults to 'Hips'.
            device (str, optional): device to use. Defaults to 'cpu'.
        """        
        self.root_name = root_name
        self.device = device
        self._joints = None
        self._offset_arr = None
        self.skeleton = self._process(skeleton)

    @property
    def joints(self):
        return self._joints

    @property
    def offset_arr(self):
        return self._offset_arr

    def _process(self, skeleton: dict) -> dict:
        """This pre-process raw skeleton structure.
        Args:
            skeleton (dict): skeleton should provide 'offesets' 'children' and 'parent' at least.
        Returns:
            dict: kinematic chain (edge removed) with torch Tensor offsets.
        """        
        
        kinematic_chain = {}
        for joint_name in skeleton.keys():
            joint_info = skeleton[joint_name]

            if not joint_info['children']: # If joint is at edge, do not add to kinematic_chain
                continue

            joint_info['offsets'] = torch.Tensor(joint_info['offsets']).reshape(3, 1)

            if 'channels' in joint_info:
                if joint_name is self.root_name:
                    joint_info['channel_order'] = ''.join([channel[0] for channel in skeleton[joint_name]['channels']])[-3:]
                else:
                    joint_info['channel_order'] = ''.join([channel[0] for channel in skeleton[joint_name]['channels']])

            kinematic_chain[joint_name] = joint_info
        
        self._joints = list(kinematic_chain.keys()) # register joints list

        offsets = []
        for joint in kinematic_chain:
            offsets.append(kinematic_chain[joint]['offsets'].squeeze().numpy())
        self._offset_arr = torch.Tensor(np.array(offsets).squeeze()).to(self.device)
        return kinematic_chain

    def forward_kinematics(self, root_positions: torch.Tensor, rotations: torch.Tensor, rot_repr='quaternion'):
        """Conducts FK with given root position and rotations.

        Args:
            root_positions (torch.Tensor): (Batch Size, # Joints, 3)
            rotations (torch.Tensor): (Batch Size, # Joints, 4) # This should be changed to more versatile way. Temporarily limit only to quaternion case
            rot_repr (str, optional): One of a 'quaternion', '6d'. Defaults to 'quaternion'.

        Returns:
            torch.Tensor: Position array in world coordinates. (Batch Size, # Joints, 3)
        """    
        position_arr_w = []
        rotation_arr_w = []
        for joint in self.joints:
            joint_idx = self.joints.index(joint)
            if not self.skeleton[joint]['parent']: # If joint is root
                position_arr_w.append(root_positions)
                rotation_arr_w.append(rotations[:,0,:])
            else:
                parent_idx = self.joints.index(self.skeleton[joint]['parent'])
                position_arr_w.append(torch.matmul(quaternion_to_matrix(rotation_arr_w[parent_idx]), self.offset_arr[joint_idx, :]) + position_arr_w[parent_idx])
                rotation_arr_w.append(matrix_to_quaternion(torch.matmul(quaternion_to_matrix(rotation_arr_w[parent_idx]), quaternion_to_matrix(rotations[:, joint_idx, :]))))  
        position_tensor_w = torch.stack(position_arr_w, dim=2).permute(0,2,1)
        rotation_tensor_w = torch.stack(rotation_arr_w, dim=2).permute(0,2,1)
        return position_tensor_w, rotation_tensor_w

