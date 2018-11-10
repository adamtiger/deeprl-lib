import time
import numpy as np
from deeprl_lib.core import ExpReplay

def test_reset_buffer():
    bf = ExpReplay(10000)
    shapes = (
        (2, 5),
        (2, 2),
        (1,), 
        (3, 9)
    )
    types = (
        np.uint8,
        np.float32,
        int,
        bool
    )
    bf.reset_buffer(shapes, types)

    assert bf.full == False
    assert bf.index == 0
    assert bf.buffer_state.shape == (10000, 2, 5)
    assert bf.buffer_action.shape == (10000, 2, 2)
    assert bf.buffer_reward.shape == (10000, 1)
    assert bf.buffer_done.shape == (10000, 3, 9)
    assert bf.buffer_state.dtype == np.uint8
    assert bf.buffer_action.dtype == np.float32
    assert bf.buffer_reward.dtype == int
    assert bf.buffer_done.dtype == bool


def test_append_one():
    bf = ExpReplay(2)
    shapes = ((2, 2), (1,), (1,), (1,))
    types = (np.float32, np.int32, np.uint8, bool)
    bf.reset_buffer(shapes, types)
    
    state = np.zeros((2, 2), dtype=np.float32) + 5
    action = np.array([3], dtype=np.int32)
    reward = np.array([2], dtype=np.uint8)
    done = np.array([True], dtype=bool)
    bf.append((state, action, reward, done))
    bf.append((state, action, reward, done))

    bf.buffer_state[0] += 1

    assert np.array_equal(bf.buffer_state[0], np.array([[6, 6], [6, 6]], dtype=np.float32))
    assert np.array_equal(bf.buffer_state[1], np.array([[5, 5], [5, 5]], dtype=np.float32))
    assert np.array_equal(bf.buffer_action[0], np.array([3], dtype=np.int32))
    assert np.array_equal(bf.buffer_reward[0], np.array([2], dtype=np.uint8))
    assert np.array_equal(bf.buffer_done[0], np.array([True], dtype=bool))
    assert bf.full == True


def test_append_second():
    bf = ExpReplay(2)
    shapes = ((2, 2), (1,), (1,), (1,))
    types = (np.float32, np.int32, np.uint8, bool)
    bf.reset_buffer(shapes, types)
    
    state = np.zeros((2, 2), dtype=np.float32) + 5
    action = np.array([3], dtype=np.int32)
    reward = np.array([2], dtype=np.uint8)
    done = np.array([True], dtype=bool)
    bf.append((state, action, reward, done))
    bf.append((state, action, reward, done))
    bf.buffer_state[1] += 1
    bf.append((state + 2, action + 3, reward + 1, done))

    assert np.array_equal(bf.buffer_state[1], np.array([[6, 6], [6, 6]], dtype=np.float32))
    assert np.array_equal(bf.buffer_state[0], np.array([[7, 7], [7, 7]], dtype=np.float32))
    assert np.array_equal(bf.buffer_action[0], np.array([6], dtype=np.int32))
    assert np.array_equal(bf.buffer_reward[0], np.array([3], dtype=np.uint8))
    assert np.array_equal(bf.buffer_done[0], np.array([True], dtype=bool))
    assert bf.full == True

