from kpt.model.robot_model import RobotModel
from kpt.model.bvh_model import BVHModel
import copy
import numpy as np
import pandas as pd


def test_forward_kinematics():
    df = pd.read_csv('assets/cmu_01_01_pos.csv')

    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)

    testing_frame = [0, 1, 2, 50, 204]
    for i in testing_frame:
        robot_model.set_frame(i)
        robot_model.forward_kinematics(robot_model.kinematic_model.root_name)
        chain = copy.deepcopy(robot_model.kinematic_chain)

        for joint in robot_model.kinematic_model.joints:
            pos_cols = [joint + '_' + pos for pos in ['Xposition', 'Yposition', 'Zposition']]
            joint_pos = df.iloc[i][pos_cols].values
            np.testing.assert_almost_equal(chain[joint]['p'].numpy().squeeze(), joint_pos, decimal=4)

def test_export_position():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    robot_model.set_frame(0)
    robot_model.forward_kinematics(robot_model.kinematic_model.root_name)
    positions = robot_model.export_positions()
    assert positions.shape == (31,3)
