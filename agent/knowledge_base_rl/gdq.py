from agent.AgentBasis import AgentBasisClass
from agent.knowledge_base_rl.planning import Planner
from collections import defaultdict
import numpy as np
import random
# import itertools
# import copy


class GDQAgent(AgentBasisClass):
    def __init__(self,
                 goal_state,
                 name="GDQAgent",
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 rmax=100,
                 u_count=5,
                 lookahead=50,
                 explore="uniform",
                 actions=None):
        super().__init__(name, actions, gamma)
        self.alpha, self.init_alpha = alpha, alpha
        self.epsilon, self.init_epsilon = epsilon, epsilon
        self.goal_state = goal_state
        self.planner = Planner()
        self.explore = explore
        self.is_initialize = self.init_is_initialize = True
        self.rmax = self.init_rmax = rmax
        self.u_count = self.init_u_count = u_count
        self.lookahead = lookahead

        self.all_trajectory = set()

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

    # Accessors

    def get_params(self):
        params = super().get_params()
        params["goal"] = self.goal_state
        params["urate"] = self.u_count
        params["alpha"] = self.alpha
        # params["Q"] = self.Q
        return params

    def get_urate(self):
        return self.u_count

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    def get_reward(self, state, action):
        if self.get_count(state, action) == 0:
            return self.rmax
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
        if self.is_initialize:
            self.is_initialize = False
            self.init_q_val(state)

        action = self._epsilon_greedy_policy(str(state))

        self._number_of_steps += 1

        return action

    def init_q_val(self, state):
        plans = self.planner.get_plans(state, self.goal_state)
        for plan in plans:
            s, a, ns = plan
            self.all_trajectory.add((s, a, ns))
            self.Q[s][a] = self.rmax

        lim = int(np.log(1 / (self.epsilon * (1 - self.get_gamma()))) / (1 - self.get_gamma()))
        for l in range(0, lim):
            for s in self.C_sas.keys():
                for a in self.C_sas[s].keys():
                    if self.get_count(s, a) >= self.u_count:
                        next_state_probabilities = self.get_transition(s, a)
                        self.Q[s][a] = self.get_reward(s, a) + self.get_gamma() * sum(
                            [p * self._get_max_q_val(ns) for ns, p in next_state_probabilities.items()])

    def _soft_max_policy(self, state):
        return NotImplemented

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > np.random.random():
            action = random.choice(self.get_actions())
        else:
            action = self._get_max_q_key(str(state))
        return action

    def update(self, state, action, reward, next_state, done=True, episode=None):
        if not done:
            self.C_sas[str(state)][action][str(next_state)] += 1
            self.rewards[str(state)][action] += [reward]
        next_action_value = 0
        if not done:
            next_action_value = self._get_max_q_val(str(next_state))

        diff = self.get_gamma() * next_action_value - self.get_q_val(str(state), action)
        self.Q[str(state)][action] += self.alpha * (reward + diff)

        self.simulated_update_with_guide(str(state))

    def simulated_update_with_guide(self, state):
        plans = self.planner.get_plans(str(state), str(self.goal_state))

        for plan in plans:
            s, a, ns = plan
            self.all_trajectory.add((s, a, ns))

        # print(self.all_trajectory)
        for n in range(self.lookahead):
            s = random.choice(list(self.C_sas.keys()))
            a = random.choice(list(self.C_sas[s].keys()))
            r = self.get_reward(s, a)
            next_state_probabilities = self.get_transition(s, a)
            if not next_state_probabilities == {}:
                ns = max(next_state_probabilities, key=next_state_probabilities.get)
            else:
                ns = None
            if (s, a, ns) in self.all_trajectory:
                diff = self.get_gamma() * self._get_max_q_val(ns) - self.get_q_val(s, a)
                self.Q[s][a] = self.get_q_val(s, a) + self.alpha * (r + diff)

    def reset(self):
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.is_initialize = self.init_is_initialize
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
