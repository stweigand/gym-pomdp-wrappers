import numpy as np

import gym
from gym import Wrapper
from gym.spaces import Discrete, Box


class NoisyEnv(Wrapper):

    """
    takes observations from an environment and adds guassian noise
    """

    def __init__(self, env, mu=0.0, sigma=0.05):
        """
        Parameters
        ----------
        env - registered gym environment
        mu - mean of gaussian noise
        sigma - variance of gaussian noise
        """
        self._wrapped_env = env
        self.mu = mu
        self.sigma = sigma
        super(NoisyEnv, self).__init__(env)
        self.full_obs_dim = self._wrapped_env.observation_space.shape[0]

    def reset(self):
        obs = self._wrapped_env.reset() + np.random.normal(self.mu,self.sigma,self.full_obs_dim)
        return obs

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        obs += np.random.normal(0,self.sigma,self.full_obs_dim)
        return obs, reward, done, info

