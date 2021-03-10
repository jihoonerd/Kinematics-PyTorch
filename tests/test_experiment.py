# This is for testing/debugging third party libraries

import numpy as np
import torch
from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer
from pytorch3d.transforms import euler_angles_to_matrix
from scipy.spatial.transform import Rotation


def test_pymo_parameterizer():
    path = 'sample_data/bvh/02_03.bvh' # Short motion ;)
    parsed = BVHParser().parse(path)
    mp = MocapParameterizer('position')
    positions = mp.fit_transform([parsed])
    position_df_cols = positions[0].values.columns.tolist()
    for col in position_df_cols:
        assert 'position' in col


def test_euler_rotmat_transform():

    rot_mat_x = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    rot_mat_y = Rotation.from_euler('y', 60, degrees=True).as_matrix()
    rot_mat_z = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    
    rot_mat1 = rot_mat_z @ rot_mat_y @ rot_mat_x

    rot_euler = torch.Tensor(np.array([30, 60, 90])/(180)*np.pi)
    
    rot_mat2 = euler_angles_to_matrix(rot_euler, 'YZX')

    print("done")
