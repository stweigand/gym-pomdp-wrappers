import numpy as np

import gym
from gym import Wrapper
from gym.spaces import Discrete, Box
from gym_pomdp.envs.rock import RockEnv, Obs


class RockSampleHistoryEnv(Wrapper):

    """
    takes observations from an RockSample environment and stacks to history given hist_len of history length
    """

    def __init__(self, env_id, hist_len=4, history_type='standard', kwargs={}):
        """
        Parameters
        ----------
        env_id - id of registered gym environment (currently only implemented for Rock-v0)
        history_type - * one_hot: encodes the actions as one hot vector in the history
                       * one_hot_pos: one hot agent position and history of 'one_hot' observations
                       * standard: encodes the actions as action_index+1 (reason for this is that the initial history is
                         all zeros and we don't want to collide with action 0, which is move north)
                       * standard_pos: one hot agent position and history of 'standard' observations
                       * field_vision: encodes the actions as action_index+1 (reason: see 'standard')
                         and noisy observation for each rock 
                       * field_vision_pos: one hot agent position and history of noisy observations for each rock
                       * fully_observable: one hot agent position and history of true observations for each rock
                       * mixed_full_pomdp: flag to indicate if full information is avail + true observations for each rock + 
                         one hot agent position and history of 'one_hot' observations
                       * history_full: complete history of: flag to indicate if full information is avail (=1) + true observations for each rock + 
                         one hot agent position + 'one_hot' action + noisy rock observation
                       * history_pomdp: complete history of: flag to indicate if full information is avail (=0) + zeros(num rocks) + 
                         one hot agent position + 'one_hot' action + noisy rock observation
                       * history_rockpos_full: complete history of: flag to indicate if full information is avail (=1) + true observations for each rock + 
                         one hot agent position + 'one_hot' action + noisy rock observation + one hot position for all rocks
        hist_len - length of the history (hist_len==0 is without history, just current observation)
        kwargs - optional arguments for initializing the wrapped environment
        """
        if not env_id == "Rock-v0":
            raise NotImplementedError("history only implemented for Rock-v0")
        env = gym.make(env_id)
        env.__init__(**kwargs)
        super(RockSampleHistoryEnv, self).__init__(env)
        self._wrapped_env = env
        self.hist_len = hist_len
        self.hist_type = history_type
        self.history = None
        self.full_obs_dim = 1
        self.num_rocks = self._wrapped_env.num_rocks
        self.size_x, self.size_y = self._wrapped_env.grid.get_size

        # specify observation space and arrangement according to selected history type
        if self.hist_type == "standard":
            self.historyIgnoreIdx = 0
            self.total_obs_dim = (1+1) # standard obs
            self.observation_space = Box(low=0, high=(4+1)+self.num_rocks, shape=(self.total_obs_dim*(self.hist_len+1),)) # history of: ac + ob pairs
            self.genObservation = self.generateObservationStandard
        elif self.hist_type == "standard_pos":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.total_obs_dim = self.historyIgnoreIdx+(1+1) # agent pos + standard obs
            self.observation_space = Box(low=0, high=(4+1)+self.num_rocks, shape=(self.historyIgnoreIdx + (1+1)*(self.hist_len+1),)) # agent pos + history of: ac + ob pairs
            self.genObservation = self.generateObservationStandardPos
        elif self.hist_type == "one_hot":
            self.historyIgnoreIdx = 0
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = (self.nact+1) # one hot encoded actaion + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.total_obs_dim*(self.hist_len+1),)) # history of: one_hot_ac + ob pairs
            self.genObservation = self.generateObservationOneHot
        elif self.hist_type == "one_hot_pos":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = self.historyIgnoreIdx+(self.nact+1) # agent pos + one hot encoded actaion + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + (self.nact+1)*(self.hist_len+1),)) # agent pos + history of: one_hot_ac + ob pairs
            self.genObservation = self.generateObservationOneHotPos
        elif self.hist_type == "field_vision":
            self.historyIgnoreIdx = 0
            self.total_obs_dim = (1+self.num_rocks) # actaion + ob (for each rock)
            self.observation_space = Box(low=0, high=(4+1)+self.num_rocks, shape=(self.total_obs_dim*(self.hist_len+1),)) # history of: ac + ob (for each rock) pairs
            self.genObservation = self.generateObservationFieldVision
        elif self.hist_type == "field_vision_pos":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.total_obs_dim = (self.historyIgnoreIdx+self.num_rocks) # oneHot agent position + ob (for each rock)
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + self.num_rocks*(self.hist_len+1),)) # agent pos + history of: ac + ob (for each rock) pairs
            self.genObservation = self.generateObservationFieldVisionPos
        elif self.hist_type == "fully_observable":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.total_obs_dim = (self.historyIgnoreIdx+self.num_rocks) # oneHot agent position + ob (for each rock)
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + self.num_rocks*(self.hist_len+1),)) # agent pos + history of: ac + ob (for each rock) pairs
            self.genObservation = self.generateObservationFullState
        elif self.hist_type == "mixed_full_pomdp":
            self.historyIgnoreIdx = 1 + self.num_rocks + self.size_x + self.size_y
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = self.historyIgnoreIdx+(self.nact+1) # ignore index + agent pos + one hot encoded action + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + (self.nact+1)*(self.hist_len+1),)) # flag + full obs + agent pos + history of: one_hot_ac + ob pairs
            self.genObservation = self.generateObservationMixed
        elif self.hist_type == "history_full":
            self.historyIgnoreIdx = 0
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = 1 + self.size_x + self.size_y + self.num_rocks + self.nact + 1 # flag + one hot agent pos + rock obs + one hot action + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + self.total_obs_dim*(self.hist_len+1),))
            self.genObservation = self.generateObservationHistoryFull
        elif self.hist_type == "history_pomdp":
            self.historyIgnoreIdx = 0
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = 1 + self.size_x + self.size_y + self.num_rocks + self.nact + 1 # flag + one hot agent pos + rock obs (zeros) + one hot action + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + self.total_obs_dim*(self.hist_len+1),))
            self.genObservation = self.generateObservationHistoryPomdp
        elif self.hist_type == "history_rockpos_full":
            self.historyIgnoreIdx = (self.size_x + self.size_y) * self.num_rocks    # num of one_hot encoded rock positions
            self.nact = self._wrapped_env.action_space.n
            self.total_history_ob_dim = 1 + self.size_x + self.size_y + self.num_rocks + self.nact + 1 
            self.total_obs_dim = self.historyIgnoreIdx + self.total_history_ob_dim # ignoreIndex + flag + one hot agent pos + rock obs + one hot action + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + self.total_history_ob_dim*(self.hist_len+1),))
            self.genObservation = self.generateObservationHistoryRockPosFull
        else:
            raise NameError("error: wrong history type")

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
        xpos, ypos = self.generatePosOneHot(False)
        if self.hist_type == "standard":
            new_ob = np.array([np.zeros(1), obs])
        elif self.hist_type == "standard_pos":
            std_ob = np.array([np.zeros(1), obs])
            new_ob = np.concatenate([xpos, ypos, std_ob])
        elif self.hist_type == "one_hot":
            new_ob = np.concatenate([np.zeros(self.nact), [obs]])
        elif self.hist_type == "one_hot_pos":
            new_ob = np.concatenate([xpos, ypos,np.zeros(self.nact), [obs]])
        elif self.hist_type == "field_vision":
            observation_rocks = self.generateFieldVisionRockObservation(False)
            new_ob = np.concatenate([np.zeros(1), observation_rocks])
        elif self.hist_type == "field_vision_pos":
            observation_rocks = self.generateFieldVisionRockObservation(False)
            new_ob = np.concatenate([xpos, ypos, observation_rocks])
        elif self.hist_type == "fully_observable":
            observation_rocks = self.generateTrueRockOvservation(False)
            new_ob = np.concatenate([xpos, ypos, observation_rocks])
        elif self.hist_type == "mixed_full_pomdp" or self.hist_type == "history_full":
            observation_rocks = self.generateTrueRockOvservation(False)
            flag = 1
            new_ob = np.concatenate([[flag],observation_rocks,xpos,ypos,np.zeros(self.nact),[obs]])
        elif self.hist_type == "history_pomdp":
            observation_rocks = np.zeros(self.num_rocks)
            flag = 0
            new_ob = np.concatenate([[flag],observation_rocks,xpos,ypos,np.zeros(self.nact),[obs]])
        elif self.hist_type == "history_rockpos_full":
            observation_rocks = self.generateTrueRockOvservation(False)
            flag = 1
            rock_pos = self.generateRockPosOneHot(False)
            new_ob = np.concatenate([[flag],observation_rocks,xpos,ypos,np.zeros(self.nact),[obs],rock_pos])
        else:
            raise NameError("error: wrong history type")
        self.reset_history(new_ob)
        # we return copy so that we can modify the history without changing already returned histories
        return np.concatenate([new_ob[0:self.historyIgnoreIdx],self.history])

    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        ob = self.genObservation(next_obs, action, done)
        self.add_to_history(ob)
        # we return copy so that we can modify the history without changing already returned histories
        return np.concatenate([ob[0:self.historyIgnoreIdx],self.history]), reward, done, info

    def generateObservationStandard(self, ob, a, done):
        return np.array([a+1, ob])

    def generateObservationStandardPos(self, ob, a, done):
        xpos, ypos = self.generatePosOneHot(done)
        std_ob = np.array([a+1, ob])
        return np.concatenate([xpos,ypos,std_ob])

    def generateObservationOneHot(self, ob, a, done):
        one_hot_a = np.zeros(self.nact, dtype=np.int)
        one_hot_a[int(a)] = 1
        return np.concatenate([one_hot_a, [ob]])

    def generateObservationOneHotPos(self, ob, a, done):
        xpos, ypos = self.generatePosOneHot(done)
        one_hot_a = np.zeros(self.nact, dtype=np.int)
        one_hot_a[int(a)] = 1
        return np.concatenate([xpos,ypos,one_hot_a,[ob]])

    def generateObservationFieldVision(self, ob, a, done):
        # action + noisy value of all rocks
        observation_rocks = self.generateFieldVisionRockObservation(done)
        return np.concatenate([[a+1], observation_rocks])

    def generateObservationFieldVisionPos(self, ob, a, done):
        # agent pos + noisy value of all rocks
        observation_rocks = self.generateFieldVisionRockObservation(done)
        xpos, ypos = self.generatePosOneHot(done)
        return np.concatenate([xpos,ypos,observation_rocks])

    def generateObservationFullState(self, ob, a, done):
        # agent pos + true value of all rocks
        observation_rocks = self.generateTrueRockOvservation(done)
        xpos, ypos = self.generatePosOneHot(done)
        return np.concatenate([xpos,ypos,observation_rocks])

    def generateObservationMixed(self, ob, a, done):
        # flag + true value of all rocks + agent pos + history of: one_hot_ac + noisy ob pairs
        flag = 1
        observation_rocks = self.generateTrueRockOvservation(done)
        xpos, ypos = self.generatePosOneHot(done)
        one_hot_a = np.zeros(self.nact, dtype=np.int)
        one_hot_a[int(a)] = 1
        return np.concatenate([[flag],observation_rocks,xpos,ypos,one_hot_a,[ob]])

    def generateObservationHistoryFull(self, ob, a, done):
        # flag + one hot agent pos + rock obs + one hot action + single ob
        return self.generateObservationMixed(ob, a, done)

    def generateObservationHistoryPomdp(self, ob, a, done):
        # flag + one hot agent pos + rock obs (zeros) + one hot action + single ob
        flag = 0
        observation_rocks = np.zeros(self.num_rocks)
        xpos, ypos = self.generatePosOneHot(done)
        one_hot_a = np.zeros(self.nact, dtype=np.int)
        one_hot_a[int(a)] = 1
        return np.concatenate([[flag],observation_rocks,xpos,ypos,one_hot_a,[ob]])

    def generateObservationHistoryRockPosFull(self, ob, a, done):
        # num of one_hot encoded rock positions 
        # flag + one hot agent pos + rock obs + one hot action + single ob + one hot rock positions
        rock_pos = self.generateRockPosOneHot(done)
        full_ob = self.generateObservationMixed(ob, a, done)
        return np.concatenate([full_ob, rock_pos])

    def generateFieldVisionRockObservation(self, done):
        # noisy value of all rocks
        observation_rocks = np.zeros((self.num_rocks,))
        if not done:
            for rock in range(0, self.num_rocks):
                if self._wrapped_env.state.rocks[rock].status == 0:  # collected
                    ob = Obs.NULL.value
                else:
                    ob = self._wrapped_env._sample_ob(self._wrapped_env.state.agent_pos, self._wrapped_env.state.rocks[rock])
                observation_rocks[rock] = ob
        return observation_rocks

    def generateTrueRockOvservation(self, done):
        # true value of all rocks
        observation_rocks = np.zeros((self.num_rocks,))
        if not done:
            for rock in range(0, self.num_rocks):
                rock_status = self._wrapped_env.state.rocks[rock].status
                if rock_status == 1:    #good
                    observation_rocks[rock] = Obs.GOOD.value
                elif rock_status == -1: #bad
                    observation_rocks[rock] = Obs.BAD.value
                else:   # collected
                    observation_rocks[rock] = Obs.NULL.value
        return observation_rocks

    def generatePosOneHot(self, done):
        xpos=np.zeros(self.size_x)
        ypos=np.zeros(self.size_y)
        if not done:
            # one hot encoded x and y position of the agent
            xpos = np.zeros(self.size_x, dtype=np.int)
            xpos[int(self._wrapped_env.state.agent_pos.x)] = 1
            ypos = np.zeros(self.size_y, dtype=np.int)
            ypos[int(self._wrapped_env.state.agent_pos.y)] = 1
        return xpos, ypos

    def generateRockPosOneHot(self, done):
        rocks = []
        if not done:
            for rock in self._wrapped_env._rock_pos:
                # one hot encoded x and y position of the rocks
                xpos = np.zeros(self.size_x, dtype=np.int)
                xpos[int(rock.x)] = 1
                ypos = np.zeros(self.size_y, dtype=np.int)
                ypos[int(rock.y)] = 1
                rocks.append(xpos)
                rocks.append(ypos)
        if len(rocks) > 0:
            return np.hstack(rocks)
        else:
            return np.zeros((self.size_x+self.size_y)*self.num_rocks)

