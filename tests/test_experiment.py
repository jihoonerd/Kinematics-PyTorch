from kpt.model.bvh_model import BVHModel
from kpt.model.robot_model import RobotModel
from kpt.vis.scatter_animation import scatter_animation, scatter_scene
import numpy as np
from pymo.preprocessing import *
from pymo.parsers import BVHParser
def test_pymo():
    path = 'sample_data/bvh/01_01.bvh'
    parsed = BVHParser().parse(path)
    mp = MocapParameterizer('position')
    positions = mp.fit_transform([parsed])
    positions

def test_vis():
    path = 'pos.csv'
    import pandas as pd
    df = pd.read_csv(path)
    for i in range(1000):
        pos_arr = df.iloc[i].values.reshape(-1, 3)
        scatter_scene(pos_arr, f'img/test_png{i}.png')