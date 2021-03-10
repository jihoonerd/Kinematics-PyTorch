from kpt.model.robot_model import RobotModel
from kpt.model.bvh_model import BVHModel
import copy
import numpy as np
import pandas as pd


def test_forward_kinematics():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    robot_model.set_frame(0)
    robot_model.forward_kinematics(robot_model.kinematic_model.root_name)
    time0_chain = copy.deepcopy(robot_model.kinematic_chain)

    robot_model.set_frame(50)
    robot_model.forward_kinematics(robot_model.kinematic_model.root_name)
    time50_chain = copy.deepcopy(robot_model.kinematic_chain)


    df = pd.read_csv('assets/cmu_01_01_pos.csv')
    for joint in robot_model.kinematic_model.joints:
        pos_cols = [joint + '_' + pos for pos in ['Xposition', 'Yposition', 'Zposition']]
        joint_pos = df.iloc[0][pos_cols].values
        np.testing.assert_almost_equal(time0_chain[joint]['p'].numpy().squeeze(), joint_pos, decimal=4)
    
    for joint in robot_model.kinematic_model.joints:
        pos_cols = [joint + '_' + pos for pos in ['Xposition', 'Yposition', 'Zposition']]
        joint_pos = df.iloc[1][pos_cols].values
        np.testing.assert_almost_equal(time0_chain[joint]['p'].numpy().squeeze(), joint_pos, decimal=4)

    for joint in robot_model.kinematic_model.joints:
        pos_cols = [joint + '_' + pos for pos in ['Xposition', 'Yposition', 'Zposition']]
        joint_pos = df.iloc[50][pos_cols].values
        np.testing.assert_almost_equal(time50_chain[joint]['p'].numpy().squeeze(), joint_pos, decimal=4)

    for joint in robot_model.kinematic_model.joints:
        pos_cols = [joint + '_' + pos for pos in ['Xposition', 'Yposition', 'Zposition']]
        joint_pos = df.iloc[204][pos_cols].values
        np.testing.assert_almost_equal(time50_chain[joint]['p'].numpy().squeeze(), joint_pos, decimal=4)


def test_export_position():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    robot_model.set_frame(0)
    robot_model.forward_kinematics(robot_model.root_name)
    positions = robot_model.export_positions()
    
    assert positions.shape == (38,3)
