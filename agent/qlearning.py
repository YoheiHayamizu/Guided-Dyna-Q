from agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np
import random


class QLearningAgent(AgentBasisClass):
    def __init__(self,
                 name="QLearningAgent",
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 explore="uniform",
                 actions=None):
        super().__init__(name, actions, gamma)
        self.alpha = self.init_alpha = alpha
        self.epsilon = self.init_epsilon = epsilon
        self.explore = explore

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

    # Accessors

    def get_params(self):
        params = super().get_params()
        params["alpha"] = self.alpha
        params["epsilon"] = self.epsilon
        params["explore"] = self.explore
        return params

    def get_alpha(self):
        return self.alpha

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    # Setters

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    # Core

    def act(self, state):
        if self.explore == "uniform":
            action = self._epsilon_greedy_policy(state)
        elif self.explore == "softmax":
            action = self._soft_max_policy(state)
        elif self.explore == "random":
            action = random.choice(self.get_actions())
        else:
            action = self._epsilon_greedy_policy(state)  # default

        self._number_of_steps += 1

        return action

    def update(self, state, action, reward, next_state, done=False, **kwargs):
        next_action_value = 0
        if not done:
            next_action_value = self._get_max_q_val(next_state)
        diff = self.get_gamma() * next_action_value - self.get_q_val(state, action)
        self.Q[state][action] += self.alpha * (reward + diff)
        # print(state, action, self.Q[state][action])

    def reset(self):
        super().reset()
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        tmp = self.get_actions()
        best_action = random.choice(self.get_actions())
        actions = tmp[:]
        np.random.shuffle(actions)
        max_q_val = float("-inf")
        for key in actions:
            q_val = self.get_q_val(state, key)
            if q_val > max_q_val:
                best_action = key
                max_q_val = q_val
        return best_action, max_q_val

    def _soft_max_policy(self, state):
        pass

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > np.random.random():
            action = random.choice(self.get_actions())
        else:
            action = self._get_max_q_key(state)
        return action