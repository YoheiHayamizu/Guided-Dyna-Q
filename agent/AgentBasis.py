import pandas as pd
import dill


class AgentBasisClass:
    def __init__(self, name, actions=None, gamma=0.99):
        self.__name = name
        self.__actions = actions
        self.__gamma = gamma
        self._number_of_episodes = 0
        self._number_of_steps = 0

    def __str__(self):
        return str(self.__name)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        """
        Return parameters of this class
        :return: <dict>
        """
        params = dict()
        params["AgentName"] = self.__name
        params["Agent's actions"] = self.__actions
        params["gamma"] = self.__gamma
        return params

    def get_name(self):
        return self.__name

    def get_gamma(self):
        return self.__gamma

    def get_actions(self):
        return self.__actions

    # Setter

    def set_name(self, new_name):
        self.__name = new_name

    def set_gamma(self, new_gamma):
        self.__gamma = new_gamma

    def set_actions(self, new_actions):
        self.__actions = new_actions

    # Core

    def act(self, state): ...

    def reset(self):
        self._number_of_steps = 0
        self._number_of_episodes = 0

    def reset_of_episode(self):
        self._number_of_steps = 0
        self._number_of_episodes += 1

    def q_to_csv(self, filename):
        table = pd.DataFrame(self.Q, dtype=str)
        table.to_csv(filename)

    def to_pickle(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)
