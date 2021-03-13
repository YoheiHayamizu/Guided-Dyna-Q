from agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np
import random


class DynaQAgent(AgentBasisClass):
    def __init__(self,
                 name="DynaQAgent",
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 lookahead=10,
                 actions=None,
                 explore="uniform"):
        super().__init__(name, actions, gamma)
        self.alpha = self.init_alpha = alpha
        self.epsilon = self.init_epsilon = epsilon
        self.explore = explore
        self.lookahead = lookahead

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.V = defaultdict(lambda: 0.0)
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

    # Accessors

    def get_params(self):
        params = super().get_params()
        params["alpha"] = self.alpha
        params["epsilon"] = self.epsilon
        params["explore"] = self.explore
        params["lookahead"] = self.lookahead
        return params

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    def get_reward(self, state, action):
        if self.get_count(state, action) == 0:
            return 0.0
        return float(sum(self.rewards[state][action])) / self.get_count(state, action)

    def get_transition(self, state, action):
        next_state_probabilities = dict()
        for next_state in self.C_sas[state][action].keys():
            prob = self.get_count(state, action, next_state) / self.get_count(state, action)
            next_state_probabilities[next_state] = prob
        return next_state_probabilities

    def get_count(self, state, action, next_state=None):
        if next_state is None:
            return sum(self.C_sas[state][action].values())
        else:
            return self.C_sas[state][action][next_state]

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
        self.C_sas[state][action][next_state] += 1
        self.rewards[state][action] += [reward]
        next_action_value = 0
        if not done:
            next_action_value = self._get_max_q_val(next_state)

        # real experience
        diff = self.get_gamma() * next_action_value - self.get_q_val(state, action)
        self.Q[state][action] += self.alpha * (reward + diff)

        # simulated experience
        for n in range(self.lookahead):
            s = random.choice(list(self.C_sas.keys()))
            a = random.choice(list(self.C_sas[s].keys()))
            r = self.get_reward(s, a)
            next_state_probabilities = self.get_transition(s, a)
            if not next_state_probabilities == {}:
                ns = max(next_state_probabilities, key=next_state_probabilities.get)
            else:
                ns = None
            diff = self.get_gamma() * self._get_max_q_val(ns) - self.get_q_val(s, a)
            self.Q[s][a] += self.alpha * (r + diff)

    def reset(self):
        super().reset()
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        tmp = self.get_actions()
        best_action = random.choice(tmp)
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