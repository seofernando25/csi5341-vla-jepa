import json

import numpy as np

from evaluation.control_trace import TraceEnvironment


def test_trace_preserves_control_and_records_distinct_states(tmp_path):
    class Environment:
        def __init__(self):
            self.raw = {'robot_state': {'eef': {'pos': np.array([0., 0., 0.])}},
                        'pixels': {'camera': np.zeros((2, 2, 3), dtype=np.uint8)}}
            self.reset_result = (self.raw, {})
            self.result = None
            self.closed = False

        def reset(self, **kwargs):
            assert kwargs == {'seed': 1000}
            return self.reset_result

        def step(self, action):
            self.received = action
            # Include an in-place observation update, as a simulator may reuse
            # its arrays between calls.
            self.raw['robot_state']['eef']['pos'][:] = [1., 2., 3.]
            self.result = (self.raw, 0., False, False, {'is_success': False})
            return self.result

        def close(self):
            self.closed = True

    env = Environment()
    trace = TraceEnvironment(env, tmp_path / 'trace', 14)
    assert trace.reset(seed=1000) is env.reset_result
    action = np.array([.1, .2, .3, 0., 0., 0., -1.])
    assert trace.step(action) is env.result
    assert env.received is action
    trace.close()
    assert env.closed
    row = json.loads((tmp_path / 'trace/commands.jsonl').read_text())
    assert row['state_before']['eef']['pos'] == [0., 0., 0.]
    assert row['state_after']['eef']['pos'] == [1., 2., 3.]
    assert row['action'] == action.tolist()
    with np.load(tmp_path / 'trace/frames-0000.npz') as image:
        np.testing.assert_array_equal(image['camera'], env.raw['pixels']['camera'])
