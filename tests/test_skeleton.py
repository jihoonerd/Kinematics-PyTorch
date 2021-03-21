from kpt.model.skeleton import TorchSkeleton
import torch
import pickle

def test_skeleton():
    with open('sample_data/skeleton.pkl', 'rb') as f:
        skeleton = pickle.load(f)
    t_skeleton = TorchSkeleton(skeleton=skeleton)
    root_positions = torch.randn(256, 3)
    local_q_pred = torch.randn(256, len(t_skeleton.joints), 4)
    fk_position = t_skeleton.forward_kinematics(root_positions, local_q_pred, rot_repr='quaternion')
    assert fk_position.shape == (256, 31, 3)