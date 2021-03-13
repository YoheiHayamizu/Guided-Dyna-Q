from agent.AgentBasis import AgentBasisClass
from agent.knowledge_base_rl.planning import Planner
from collections import defaultdict
import numpy as np
import random


# TODO: planningの長さ伸ばしたらoutputが出てこないとか，exitで埋まっちゃうとかを直す

class DARLINGAgent(AgentBasisClass):
    def __init__(self,
                 goal_state,
                 name="DARLINGAgent",
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 explore="uniform",
                 actions=None):
        super().__init__(name, actions, gamma)
        self.alpha, self.init_alpha = alpha, alpha
        self.epsilon, self.init_epsilon = epsilon, epsilon
        self.goal_state = goal_state
        self.planner = Planner()
        self.explore = explore
        self.is_initialize = self.init_is_initialize = True

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.rewards = defaultdict(lambda: defaultdict(list))

    # Accessors

    def get_params(self):
        params = super().get_params()
        params["goal"] = self.goal_state
        params["alpha"] = self.alpha
        # params["Q"] = self.Q
        return params

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    def get_executable_actions(self, state):
        new_actions = set()
        plans = self.planner.get_plans(state, self.goal_state)
        # print(plans)
        for plan in plans:
            # print(plan, str(state))
            if plan[0] == str(state):
                new_actions.add(plan[1])
        if len(new_actions) == 0:
            new_actions.add(("goto", 17))
            # print(state, plans)
        return list(new_actions)

    # Core

    def act(self, state):
        action_list = self.get_executable_actions(state)
        # print(len(self.get_actions()))

        if self.explore == "uniform":
            action = self._epsilon_greedy_policy(state)
        elif self.explore == "greedy":
            action = self._get_max_q_key(state)
        elif self.explore == "softmax":
            action = self._soft_max_policy(state)
        elif self.explore == "random":
            action = random.choice(list(self.get_actions()))
        else:
            action = self._epsilon_greedy_policy(state)  # default

        while action not in action_list:
            action = random.choice(list(self.get_actions()))
            # action = self._epsilon_greedy_policy(state)

        self._number_of_steps += 1

        return action

    def _soft_max_policy(self, state):
        return NotImplemented

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > np.random.random():
            action = random.choice(self.get_actions())
        else:
            action = self._get_max_q_key(state)
        return action

    def update(self, state, action, reward, next_state, done=True, episode=None):
        next_action_value = 0
        if not done:
            next_action_value = self._get_max_q_val(next_state)

        diff = self.get_gamma() * next_action_value - self.get_q_val(state, action)
        self.Q[state][action] += self.alpha * (reward + diff)

    def reset(self):
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.is_initialize = self.init_is_initialize
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.rewards = defaultdict(lambda: defaultdict(list))

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        # print(state, self.actions[state])
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
