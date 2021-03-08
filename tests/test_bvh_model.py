from kpt.model.bvh_model import BVHModel


def test_bvh_model():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    kinematic_chain = bvh_model.get_kinematic_chain()
    print(kinematic_chain)


def test_forward_kinematics():
    # b, r, q input
    pass