from kpt.model.robot_model import RobotModel
from kpt.model.bvh_model import BVHModel
import copy
import numpy as np


def test_bvh_robot_model():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    robot_model.set_frame(0)
    robot_model.forward_kinematics(robot_model.root_name)
    time0_chain = copy.deepcopy(robot_model.kinematic_chain)

    robot_model.set_frame(50)
    robot_model.forward_kinematics(robot_model.root_name)
    time50_chain = copy.deepcopy(robot_model.kinematic_chain)

    assert np.array_equal(time0_chain['RightUpLeg']['p'].numpy(), (time0_chain['RHipJoint']['p'] + time0_chain['RightUpLeg']['offsets']).numpy())
    assert np.array_equal(time50_chain['RightUpLeg']['p'].numpy(), (time50_chain['RHipJoint']['p'] + time50_chain['RightUpLeg']['offsets']).numpy())

def test_export_position():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    robot_model.set_frame(0)
    robot_model.forward_kinematics(robot_model.root_name)
    positions = robot_model.export_positions()
    
    assert positions.shape == (38,3)
