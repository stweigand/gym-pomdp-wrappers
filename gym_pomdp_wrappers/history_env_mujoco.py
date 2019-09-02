import numpy as np

import gym
from gym import Wrapper
from gym.spaces import Discrete, Box


class MuJoCoHistoryEnv(Wrapper):

    """
    takes observations from an environment and stacks to history given hist_len of history length
    """

    def __init__(self, env_id, hist_len=4, history_type='pomdp'):
        """
        Parameters
        ----------
        env_id - id of registered gym environment
        history_type - * pomdp: remove specified dimensions, meaning to return history of remaining dimensions
                       * mixed_full_pomdp: [flag=0] + [full observation] + history of [pomdp observation (see above)]
                       * history_full: history of [flag=1] + [full observation]
                       * history_pomdp: history of [flag=0] + [pomdp observation (replaced by zeros)]
                       * history_ac_full: history of [flag=1] + [full observation] + [action]
                       * history_ac_pomdp: history of [flag=0] + [pomdp observation (replaced by zeros)] + [action]
        hist_len - length of the history (hist_len==0 is without history, just current observation)
        """
        env = gym.make(env_id)
        super(MuJoCoHistoryEnv, self).__init__(env)
        self._wrapped_env = env
        self.hist_len = hist_len
        self.hist_type = history_type
        self.history = None
        self.full_obs_dim = self._wrapped_env.observation_space.shape[0]
        self.ac_dim = self._wrapped_env.action_space.shape[0]

        if self.hist_len <= 0:
            raise ValueError("History length must be greater or equals zero!")

        # specify ob dimensions which are inactive in POMDP versions
        if env_id == "Ant-v2":
            self.remove_dim_start = 57
            self.remove_dim_end = 110
        elif env_id == "Hopper-v2":
            # angle of thigh + leg + foot, velocity of x + z
            self.remove_dim_start = 2
            self.remove_dim_end = 6
        elif env_id == "HalfCheetah-v2":
            # angle of bthigh + bshin + bfoot + fthigh + fshin + ffoot, velocity of x + y
            self.remove_dim_start = 2
            self.remove_dim_end = 9
        elif env_id == "Humanoid-v2":
            # angle of left hip(x,y,z) + knee
            self.remove_dim_start = 6
            self.remove_dim_end = 15
        elif env_id == "InvertedDoublePendulum-v2":
            # angle of pole1 + pole2, y pos of pole1 + pole2
            self.remove_dim_start = 0
            self.remove_dim_end = 3
        elif env_id == "Reacher-v2":
            # target x and y pos, angular velocity both joints, dist fingertip <-> target
            self.remove_dim_start = 4
            self.remove_dim_end = 10
        elif env_id == "Swimmer-v2":
            # angle of mid + back, velocity of x + y, angular velocity of z
            self.remove_dim_start = 1
            self.remove_dim_end = 5
        elif env_id == "Walker2d-v2":
            # velocity of x + y, angular velocity of z + right thigh + leg + foot
            self.remove_dim_start = 8
            self.remove_dim_end = 13
        elif self.hist_type == "history_full" or self.hist_type == "history_ac_full":
            self.remove_dim_start = None
            self.remove_dim_end = None
        else:
            raise NotImplementedError("POMDP version not implemented for "+env_id)
        if self.hist_type != "history_full" and self.hist_type != "history_ac_full":
            if self.remove_dim_end < self.remove_dim_start:
                raise ValueError("revome_dim_end < remove_dim_start")
            self.remove_dim = self.remove_dim_end - self.remove_dim_start + 1

        # specify observation space and arrangement according to selected history type
        if self.hist_type == "pomdp":
            self.historyIgnoreIdx = 0
            self.genObservation = self.generateObservationPomdp
            self.total_obs_dim = self.full_obs_dim-self.remove_dim
            self.ob_space_dim = (self.full_obs_dim-self.remove_dim)*(self.hist_len+1)
        elif self.hist_type == "mixed_full_pomdp":
            self.historyIgnoreIdx = 1 + self.full_obs_dim
            self.genObservation = self.generateObservationMixed
            self.total_obs_dim = self.historyIgnoreIdx + (self.full_obs_dim-self.remove_dim)
            self.ob_space_dim = self.historyIgnoreIdx + (self.full_obs_dim-self.remove_dim)*(self.hist_len+1)
        elif self.hist_type == "history_full":
            self.historyIgnoreIdx = 0
            self.genObservation = self.generateObservationHistoryFull
            self.total_obs_dim = 1+self.full_obs_dim
            self.ob_space_dim = self.total_obs_dim*(self.hist_len+1)
        elif self.hist_type == "history_pomdp":
            self.historyIgnoreIdx = 0
            self.genObservation = self.generateObservationHistoryPomdp
            self.total_obs_dim = 1+self.full_obs_dim
            self.ob_space_dim = self.total_obs_dim*(self.hist_len+1)
        elif self.hist_type == "history_ac_full":
            self.historyIgnoreIdx = 0
            self.genObservation = self.generateObservationHistoryActionFull
            self.total_obs_dim = 1+self.full_obs_dim+self.ac_dim
            self.ob_space_dim = self.total_obs_dim*(self.hist_len+1)
        elif self.hist_type == "history_ac_pomdp":
            self.historyIgnoreIdx = 0
            self.genObservation = self.generateObservationHistoryActionPomdp
            self.total_obs_dim = 1+self.full_obs_dim+self.ac_dim
            self.ob_space_dim = self.total_obs_dim*(self.hist_len+1)
        else:
            raise NameError("error: wrong history type: " + self.hist_type)
        #self.total_obs_dim = self.historyIgnoreIdx + (self.full_obs_dim-self.remove_dim)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(self.ob_space_dim,))
     
        self.observation_dim_hist_part = self.total_obs_dim - self.historyIgnoreIdx
        print('-------- History Info: --------')
        print('total obs dim:', self.total_obs_dim)
        print('original obs dim:', self.full_obs_dim)
        print('history obs dim:', self.observation_dim_hist_part)
        print('-------------------------------')

    def reset_history(self, new_):
        self.history = np.zeros((self.observation_space.shape[0]-self.historyIgnoreIdx, ))
        self.history[0:self.observation_dim_hist_part] = new_[self.historyIgnoreIdx:]

    def add_to_history(self, new_):
        self.history[self.observation_dim_hist_part:] = self.history[:-self.observation_dim_hist_part]
        self.history[0:self.observation_dim_hist_part] = new_[self.historyIgnoreIdx:]

    def reset(self):
        obs = self._wrapped_env.reset()
        new_ob = self.genObservation(obs, np.zeros(self.ac_dim))
        self.reset_history(new_ob)
        # we return copy so that we can modify the history without changing already returned histories
        return np.concatenate([new_ob[0:self.historyIgnoreIdx],self.history])

    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        ob = self.genObservation(next_obs, action)
        self.add_to_history(ob)
        # we return copy so that we can modify the history without changing already returned histories
        return np.concatenate([ob[0:self.historyIgnoreIdx],self.history]), reward, done, info

    def generateObservationPomdp(self, ob, ac):
        return np.concatenate([ob[0:self.remove_dim_start], ob[self.remove_dim_end+1:]])

    def generateObservationMixed(self, ob, ac):
        # flag + full observation + history of pomdp observations
        flag = 1
        return np.concatenate([[flag], ob, self.generateObservationPomdp(ob, ac)])

    def generateObservationHistoryFull(self, ob, ac):
        # history of: flag + full obsservations
        flag = 1
        return np.concatenate([[flag], ob])

    def generateObservationHistoryPomdp(self, ob, ac):
        # history of: flag + pomdp obsservations
        flag = 0
        return np.concatenate([[flag], ob[0:self.remove_dim_start], np.zeros(self.remove_dim), ob[self.remove_dim_end+1:]])

    def generateObservationHistoryActionFull(self, ob, ac):
        # history of: flag + full obsservations + action
        flag = 1
        return np.concatenate([[flag], ob, ac])

    def generateObservationHistoryActionPomdp(self, ob, ac):
        # history of: flag + pomdp obsservations + action
        flag = 0
        return np.concatenate([[flag], ob[0:self.remove_dim_start], np.zeros(self.remove_dim), ob[self.remove_dim_end+1:], ac])


