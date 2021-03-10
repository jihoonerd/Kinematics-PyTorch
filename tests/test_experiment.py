# This is for testing/debugging third party libraries

from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer


def test_pymo_parameterizer():
    path = 'sample_data/bvh/02_03.bvh' # Short motion ;)
    parsed = BVHParser().parse(path)
    mp = MocapParameterizer('position')
    positions = mp.fit_transform([parsed])
    position_df_cols = positions[0].values.columns.tolist()
    for col in position_df_cols:
        assert 'position' in col
