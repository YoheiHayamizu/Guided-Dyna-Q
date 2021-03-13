import copy
import dill
from typing import Union, List, Any, Optional, Tuple
from collections import defaultdict


class MDPStateClass(object):
    def __init__(self, data, is_terminal=False):
        self.data = data
        self.__is_terminal = is_terminal

    # Accessors

    def get_data(self):
        return self.data

    def is_terminal(self):
        return self.__is_terminal

    # Setters

    def set_data(self, data):
        self.data = data

    def set_terminal(self, is_terminal=True):
        self.__is_terminal = is_terminal

    # Core

    def __hash__(self):
        if self.data.__hash__ is None:
            return hash(tuple(self.data))
        else:
            return hash(self.data)

    def __eq__(self, other):
        assert isinstance(other, MDPStateClass), "Arg object is not in " + type(self).__module__
        return self.data == other.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MDPBasisClass(object):
    """ abstract class for a MDP """

    def __init__(self,
                 init_state: MDPStateClass,
                 transition_func: Any,
                 reward_func: Any,
                 actions: Any = None):
        self.__init_state = copy.deepcopy(init_state)
        self.__current_state = copy.deepcopy(self.__init_state)
        self.__actions = actions
        self.__transition_func = transition_func
        self.__reward_func = reward_func
        self.state_counter = defaultdict(lambda: 0)

        # Accessors

    def get_params(self) -> dict:
        """ get parameters

        :return: dict
        """
        param_dict = dict()
        param_dict["init_state"] = self.__init_state
        param_dict["Env's actions"] = self.__actions
        return param_dict

    def get_init_state(self):
        return self.__init_state

    def get_current_state(self):
        return self.__current_state

    def get_actions(self):
        return self.__actions

    def get_transition_func(self):
        return self.__transition_func

    def get_reward_func(self):
        return self.__reward_func

    def get_state_count(self, state=None):
        if state is not None:
            return self.state_counter[state]
        return self.state_counter

    def get_executable_actions(self, state=None):
        raise NotImplementedError

    # Setters

    def set_init_state(self, new_init_state):
        self.__init_state = copy.deepcopy(new_init_state)

    def set_actions(self, new_actions):
        self.__actions = new_actions

    def set_transition_func(self, new_transition_func):
        self.__transition_func = new_transition_func

    # Core

    def step(self, action: Any) -> Tuple[MDPStateClass, float, bool, dict]:
        """ Proceed to next step

        :param action: <Any>
        :return: tuple[Any, float, bool, dict]
        """
        self.state_counter[self.__current_state] += 1
        next_state = self.__transition_func(self.__current_state, action)
        reward = self.__reward_func(self.__current_state, action, next_state)
        done = self.__current_state.is_terminal()
        self.__current_state = next_state

        return next_state, reward, done, self.get_params()

    def reset(self):
        self.__current_state = copy.deepcopy(self.__init_state)
        return self.__current_state

    def to_pickle(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)

    def render(self):
        raise NotImplementedError
