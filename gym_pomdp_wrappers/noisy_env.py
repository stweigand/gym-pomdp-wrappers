import numpy as np

import gym
from gym import Wrapper
from gym.spaces import Discrete, Box


class NoisyEnv(Wrapper):

    """
    takes observations from an environment and adds guassian noise
    """

    def __init__(self, env, hist_len=0, history_type='history_noisy', mu=0.0, sigma=0.0):
        """
        Parameters
        ----------
        env - registered gym environment
        history_type - * history_noisy: history of [noisy observation]
                       * history_ac_full: history of [flag=1] + [full observation] + [action]
                       * history_ac_noisy: history of [flag=0] + [noisy observation] + [action]
        mu - mean of gaussian noise
        sigma - variance of gaussian noise
        """
        self._wrapped_env = env
        self.hist_len = hist_len
        self.hist_type = history_type
        self.mu = mu
        self.sigma = sigma
        super(NoisyEnv, self).__init__(env)
        self.history = None
        self.full_obs_dim = self._wrapped_env.observation_space.shape[0]
        self.ac_dim = self._wrapped_env.action_space.shape[0]
        self.historyIgnoreIdx = 0

        # specify observation space and arrangement according to selected history type
        if self.hist_type == "history_noisy":
            self.genObservation = self.generateObservationNoisy
            self.total_obs_dim = self.full_obs_dim
        elif self.hist_type == "history_ac_full":
            self.genObservation = self.generateObservationHistoryFull
            self.total_obs_dim = 1+self.full_obs_dim+self.ac_dim
        elif self.hist_type == "history_ac_noisy":
            self.genObservation = self.generateObservationHistoryNoisy
            self.total_obs_dim = 1+self.full_obs_dim+self.ac_dim  
        else:
            raise NameError("error: wrong history type: " + self.hist_type)

        self.ob_space_dim = self.total_obs_dim*(self.hist_len+1)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(self.ob_space_dim,))
        self.observation_dim_hist_part = self.total_obs_dim - self.historyIgnoreIdx

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

    def generateObservationNoisy(self, ob, ac):
        # history of: noisy observation
        flag = 0
        return ob+np.random.normal(self.mu,self.sigma,self.full_obs_dim)

    def generateObservationHistoryFull(self, ob, ac):
        # history of: flag + full obsservations + action
        flag = 1
        return np.concatenate([[flag], ob, ac])

    def generateObservationHistoryNoisy(self, ob, ac):
        # history of: flag + noisy obsservations + action
        flag = 0
        return np.concatenate([[flag], ob+np.random.normal(self.mu,self.sigma,self.full_obs_dim), ac])

