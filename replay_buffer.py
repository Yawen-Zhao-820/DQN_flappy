from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
    Replay buffer object

    This object stores the gameplay experience.
    """

    def __init__(self, max_len=50000):
        self.gameplay_exps = deque(maxlen=max_len)

    def store_gameplay_exp(self, state, next_state, reward, action, terminate):
        """
        Store a step of gameplay experience.

        :param state: current state
        :param next_state: next state
        :param reward: the reward of taking the action at current state
        :param action: the action taken currently
        :param terminate: Boolean representing whether the game is over after
                          taking the action.
        """
        self.gameplay_exps.append((state, next_state, action, reward, terminate))

    def get_gameplay_exp_batch(self, batch_size=32):
        if len(self.gameplay_exps) > batch_size:
            exp_samples = random.sample(self.gameplay_exps, k=batch_size)
        else:
            exp_samples = self.gameplay_exps

        state_samples = []
        next_state_samples = []
        action_samples = []
        reward_samples = []
        terminate_samples = []

        for exp_sample in exp_samples:
            state_samples.append(exp_sample[0])
            next_state_samples.append(exp_sample[1])
            action_samples.append(exp_sample[2])
            reward_samples.append(exp_sample[3])
            terminate_samples.append(exp_sample[4])

        state_samples = np.array(state_samples)
        next_state_samples = np.array(next_state_samples)
        action_samples = np.array(action_samples)
        reward_samples = np.array(reward_samples)
        terminate_samples = np.array(terminate_samples)
        return state_samples, next_state_samples, action_samples, reward_samples, \
            terminate_samples