from kpt.model.bvh_model import BVHModel


def test_bvh_model():
    path = 'sample_data/bvh/01_01.bvh'
    bvh_model = BVHModel(path)
    kinematic_chain = bvh_model.get_kinematic_chain()
    assert len(kinematic_chain) == 38
    assert kinematic_chain[bvh_model.root_name]['offsets'].shape == (3,1)
