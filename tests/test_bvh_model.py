import numpy as np
import pytest
from kpt.model.bvh_model import BVHModel
from scipy.spatial.transform import Rotation


@pytest.fixture
def kinematic_model():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    return bvh_model

def test_build_kinematic_chain(kinematic_model):
    bvh_model = kinematic_model
    kinematic_chain = bvh_model.kinematic_chain
    assert len(kinematic_chain) == 38
    assert kinematic_chain[bvh_model.root_name]['offsets'].shape == (3,1)

def test_get_root_pos_rot(kinematic_model):
    bvh_model = kinematic_model
    pos0, rot0 = bvh_model.get_root_pos_rot(frame_id=0)
    
    np.testing.assert_almost_equal(pos0.numpy(), np.array([[9.3722, 17.8693, -17.3198]]).T, decimal=6)
    np.testing.assert_array_equal(rot0.numpy(), np.eye(3))
    
    pos9, rot9 = bvh_model.get_root_pos_rot(frame_id=9)
    np.testing.assert_almost_equal(pos9.numpy(), np.array([[9.3598, 17.8506, -17.3603]]).T, decimal=6)

    z_rot = Rotation.from_euler('z', -3.9143, degrees=True).as_matrix()
    y_rot = Rotation.from_euler('y', -7.0800, degrees=True).as_matrix()
    x_rot = Rotation.from_euler('x', -2.0917, degrees=True).as_matrix()
    rot_ref = z_rot @ y_rot @ x_rot
    np.testing.assert_almost_equal(rot9, rot_ref, decimal=4)
