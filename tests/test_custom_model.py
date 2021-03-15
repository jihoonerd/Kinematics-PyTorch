import numpy as np
import pytest
import torch
from kpt.model.custom_model import CustomModel
from pymo.parsers import BVHParser
from scipy.spatial.transform import Rotation


@pytest.fixture
def kinematic_model():
    path = 'sample_data/bvh/01_01.bvh'
    parsed = BVHParser().parse(path)
    # use same kinematic chain as bvh parser (PyMO)
    skeleton = parsed.skeleton
    motion_data = torch.Tensor(parsed.values.values)
    channel_name = parsed.values.iloc[0].index.tolist()
    return skeleton, motion_data, channel_name

def test_build_kinematic_chain(kinematic_model):
    custom_model = CustomModel(skeleton=kinematic_model[0], motion_data=kinematic_model[1][0], channel_name=kinematic_model[2])

    assert len(custom_model.skeleton) == 38
    assert len(custom_model.joints) == 38
    assert custom_model.model_type == 'custom'
    assert custom_model.root_name == 'Hips'
    assert custom_model.motion_data.shape[0] == 96
    assert len(custom_model.channel_name) == 96

def test_get_root_pos_rot(kinematic_model):
    custom_model_at_0 = CustomModel(skeleton=kinematic_model[0], motion_data=kinematic_model[1][0], channel_name=kinematic_model[2])
    pos0, rot0 = custom_model_at_0.get_root_pos_rot()
    np.testing.assert_array_equal(pos0, torch.Tensor([[9.3722, 17.8693, -17.3198]]).T)
    np.testing.assert_array_equal(rot0, torch.eye(3))
    
    custom_model_at_9 = CustomModel(skeleton=kinematic_model[0], motion_data=kinematic_model[1][9], channel_name=kinematic_model[2])
    pos9, rot9 = custom_model_at_9.get_root_pos_rot()
    np.testing.assert_array_equal(pos9, torch.Tensor([[9.3598, 17.8506, -17.3603]]).T)

    z_rot = Rotation.from_euler('z', -3.9143, degrees=True).as_matrix()
    y_rot = Rotation.from_euler('y', -7.0800, degrees=True).as_matrix()
    x_rot = Rotation.from_euler('x', -2.0917, degrees=True).as_matrix()
    rot_ref = z_rot @ y_rot @ x_rot
    np.testing.assert_almost_equal(rot9, rot_ref, decimal=4)
