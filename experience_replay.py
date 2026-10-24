"""Experience replay buffer for off-policy DQN training."""

from collections import deque
import random

class ReplayMemory():
    
    # create FIFO queue -> experience replay
    def __init__(self, maxlen, seed=None):
        # deque automatically discards the oldest item when capacity is exceeded.
        # `seed` is accepted by the current signature but is not used in this implementation.
        self.memory = deque([], maxlen=maxlen)

    def append(self, new_exp):
        """Add one transition, typically (state, action, next_state, reward, done)."""
        self.memory.append(new_exp)

    def sample(self, sample_size):
        """Return a random subset of stored transitions without replacement."""
        return random.sample(self.memory, sample_size)

    # current buffer size
    def __len__(self):
        return len(self.memory)