import torch
from kpt.model.robot_model import RobotModel
from kpt.model.custom_model import CustomModel
from kpt.model.bvh_model import BVHModel
from experiment.network import SimpleDenseNet
from pymo.parsers import BVHParser
import numpy as np
import pandas as pd
from torch.nn import L1Loss
import torch.optim as optim

def test_learnable_kinematics():
    df = pd.read_csv('assets/cmu_01_01_pos.csv')

    path = 'sample_data/bvh/01_01.bvh'
    parsed = BVHParser().parse(path)
    # use same kinematic chain as bvh parser (PyMO)
    kinematic_chain = parsed.skeleton
    motion_data = torch.randn(*parsed.values.iloc[0].shape)
    channel_name = parsed.values.iloc[0].index.tolist()
    simple_dnet = SimpleDenseNet(input_dims=96, out_dims=96)
    optimizer = optim.Adam(simple_dnet.parameters())

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()

        out_dnet = simple_dnet(motion_data)
        custom_model = CustomModel(kinematic_chain, out_dnet, channel_name=channel_name)
        robot_model = RobotModel(custom_model)
        robot_model.set_frame(0)
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
        total_loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch} / Total Loss:   {total_loss}")
