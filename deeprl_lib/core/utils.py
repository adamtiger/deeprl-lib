import numpy as np


class ExpReplay:
    """
    This is the buffer for storing the experiences.
    Experience: (state, action, reward, done)
    state - image (or observation), high dimensional
    action - vector with action values
    reward - generally one real value
    done - 0 (false), 1 (true)
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0  # shows the next empty place to put the item
        self.full = False  # important for sampling
    
    def reset_buffer(self, shapes, types):
        self.buffer_state = np.zeros((self.buffer_size,) + shapes[0], dtype=types[0])
        self.buffer_action = np.zeros((self.buffer_size,) + shapes[1], dtype=types[1])
        self.buffer_reward = np.zeros((self.buffer_size,) + shapes[2], dtype=types[2])
        self.buffer_done = np.zeros((self.buffer_size,) + shapes[3], dtype=types[3])
        self.index = 0
        self.full = False

    def append(self, item):
        state, action, reward, done = item
        self.buffer_state[self.index] = state
        self.buffer_action[self.index] = action
        self.buffer_reward[self.index] = reward
        self.buffer_done[self.index] = done
        self.index += 1
        if self.index >= self.buffer_size:
            self.full = True
            self.index = 0
    
    def sample(self, batch_size):
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.index, size=batch_size)
        states = self.buffer_state[indices]
        actions = self.buffer_action[indices]
        rewards = self.buffer_reward[indices]
        dones = self.buffer_done[indices]
        return (states, actions, rewards, dones)

