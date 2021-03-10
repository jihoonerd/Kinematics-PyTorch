from kpt.model.robot_model import RobotModel
from kpt.model.bvh_model import BVHModel
import copy
import numpy as np
import pandas as pd


def test_bvh_robot_model():
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
    time0_rul_pos = df.iloc[0][['RightUpLeg_Xposition', 'RightUpLeg_Yposition', 'RightUpLeg_Zposition']].values
    np.testing.assert_almost_equal(time0_chain['RightUpLeg']['p'].numpy().squeeze(), time0_rul_pos, decimal=4)

    time50_root_pos = df.iloc[50][['Hips_Xposition', 'Hips_Yposition', 'Hips_Zposition']].values
    np.testing.assert_almost_equal(time50_chain['Hips']['p'].numpy().squeeze(), time50_root_pos, decimal=4)

    time50_rul_pos = df.iloc[50][['RightUpLeg_Xposition', 'RightUpLeg_Yposition', 'RightUpLeg_Zposition']].values
    np.testing.assert_almost_equal(time50_chain['RightUpLeg']['p'].numpy().squeeze(), time50_rul_pos, decimal=4)


def test_export_position():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    robot_model.set_frame(0)
    robot_model.forward_kinematics(robot_model.root_name)
    positions = robot_model.export_positions()
    
    assert positions.shape == (38,3)
