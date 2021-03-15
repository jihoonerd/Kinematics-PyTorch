import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from experiment.network import SimpleDenseNet
from kpt.model.bvh_model import BVHModel
from kpt.model.custom_model import CustomModel
from kpt.model.robot_model import RobotModel
from pymo.parsers import BVHParser


def test_forward_kinematics():
    df = pd.read_csv('assets/cmu_01_01_pos.csv')

    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)

    testing_frame = [0, 1, 2, 50, 204]
    for i in testing_frame:
        robot_model.set_frame(i)
        robot_model.forward_kinematics(robot_model.kinematic_model.root_name)
        chain = robot_model.kinematic_chain

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
    assert positions.shape == (38,3)

def test_learn_custom_model():
    df = pd.read_csv('assets/cmu_01_01_pos.csv')
    path = 'sample_data/bvh/01_01.bvh'
    parsed = BVHParser().parse(path)
    # use same kinematic chain as bvh parser (PyMO)
    kinematic_chain = parsed.skeleton
    motion_data = torch.randn(*parsed.values.iloc[0].shape)
    channel_name = parsed.values.iloc[0].index.tolist()
    simple_dnet = SimpleDenseNet(input_dims=96, out_dims=96)
    optimizer = optim.Adam(simple_dnet.parameters())
    
    hip_loss = []
    ll_loss = []
    rfa_loss = []

    epochs = 50
    for _ in range(epochs):
        optimizer.zero_grad()
        out_dnet = simple_dnet(motion_data)
        custom_model = CustomModel(kinematic_chain, out_dnet, channel_name=channel_name)
        robot_model = RobotModel(custom_model)
        robot_model.set_frame()
        robot_model.forward_kinematics(robot_model.kinematic_model.root_name)
        chain = robot_model.kinematic_chain
        # calculate positional loss
        total_loss = 0
        count = 0
        for joint_name in chain:
            count += 1
            target_pos = torch.Tensor(df.iloc[0][[joint_name + '_' + channel for channel in ['Xposition', 'Yposition', 'Zposition']]].values)
            joint_pos = chain[joint_name]['p'].squeeze()
            total_loss += torch.abs(joint_pos - target_pos).mean()
        total_loss = total_loss / count
        hip_loss.append(get_l1_loss(robot_model.kinematic_chain, df, 'Hips'))
        ll_loss.append(get_l1_loss(robot_model.kinematic_chain, df, 'LeftLeg'))
        rfa_loss.append(get_l1_loss(robot_model.kinematic_chain, df, 'RightForeArm'))
        total_loss.backward()
        optimizer.step()
    
    dense_out = simple_dnet(motion_data)
    custom_model = CustomModel(kinematic_chain, dense_out, channel_name=channel_name)
    robot_model = RobotModel(custom_model)
    robot_model.set_frame()
    robot_model.forward_kinematics(robot_model.kinematic_model.root_name)

    assert hip_loss[epochs-1] < hip_loss[epochs//2]
    assert ll_loss[epochs-1] < ll_loss[epochs//2]
    assert rfa_loss[epochs-1] < rfa_loss[epochs//2]
    
def get_l1_loss(kinematic_chain, df_pos, joint_name):
    l1_loss = torch.nn.L1Loss()
    pred = kinematic_chain[joint_name]['p'].squeeze().detach()
    position_cols = [joint_name + '_' + i for i in ['Xposition', 'Yposition', 'Zposition']]
    true = torch.Tensor(df_pos.iloc[0][position_cols].values)
    return l1_loss(pred, true).item()
