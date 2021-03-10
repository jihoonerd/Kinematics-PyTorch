from kpt.model.bvh_model import BVHModel
from kpt.model.robot_model import RobotModel
from kpt.vis.scatter_animation import scatter_animation, scatter_scene
import numpy as np

def test_scatter_plot1():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    for i in range(100):
        robot_model.set_frame(i)
        robot_model.forward_kinematics(robot_model.root_name)
        positions = robot_model.export_positions()
        scatter_scene(positions, f'img/test_png{i}.png')

def test_scatter_plot2():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)
    robot_model.set_frame(3)
    robot_model.forward_kinematics(robot_model.root_name)
    positions = robot_model.export_positions()
    scatter_scene(positions, 'test_png2.png')


def test_scatter_animation():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    robot_model = RobotModel(bvh_model)

    total_sequence = robot_model.model.bvh_parsed.values.shape[0]
    position_arrs = []
    for i in range(total_sequence):
        robot_model.set_frame(i)
        robot_model.forward_kinematics(robot_model.root_name)
        position_arr = robot_model.export_positions()
        position_arrs.append(position_arr)
    position_arrs = np.stack(position_arrs)
    scatter_animation(position_arrs, 'test_viz.mp4')

