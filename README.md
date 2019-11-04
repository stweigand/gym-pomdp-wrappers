# gym-pomdp-wrappers

**gym-pomdp-wrappers** is a set of wrappers that can be used to turn some specific OpenAI Gym environments into Partially Observable Markov Decision Processes (POMDPs).


## Dependencies

This pacakge requires python3 (>=3.5). Additionally, you need the following packages:

* [OpenAI Gym](https://github.com/openai/gym)
* [gym_pomdp](https://github.com/d3sm0/gym_pomdp)


## Installation

After installaling the dependencies, you can perform a install of **gym-pomdp-wrappers** with:


```bash
git clone https://github.com/stweigand/gym-pomdp-wrappers.git
cd gym-pomdp-wrappers
pip install -e .
```

## Implemented Wrappers

* **MuJoCoHistoryEnv:** Stacks observations from a given MuJoCo environment to a history of given length. There are multiple history types implemented. For the POMDP versions of some tasks are dimensions of the observation vecotor specified which will be overwritten with zeros.

* **RockSampleHistoryEnv:** Stacks observations (and additional information) from a RockSample instance to a history of given length. The history types specifiy which information are added to the history.

* **NoisyEnv:** Takes observations from an environment, adds guassian noise and stacks them to a history of given length. There are multiple history types implemented.

Detailed information on the different history types and usage will be added later! See the comments in the code for further information.


## Example

Environments could be initialized as follows:
```python3
import gym
from gym_pomdp_wrappers import MuJoCoHistoryEnv
env = MuJoCoHistoryEnv("Walker2d-v2", hist_len=4, history_type="history_ac_pomdp")

```

```python3
import gym
import gym_pomdp
from gym_pomdp_wrappers import RockSampleHistoryEnv
env = RockSampleHistoryEnv("Rock-v0", hist_len=4, history_type="history_full", kwargs={'board_size':5, 'num_rocks':5})

```

## MuJoCo

Some of the wrappers use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)


## Citing the Project

This repository is part of my Master's thesis which is about "Guided Reinforcement Learning Under Partial Observability" (Link will be added later). Here is a BibTeX entry that you can use to cite it in a publication::

```
@mastersthesis{guidedreinforcementlearning,
    Author = {Stephan Weigand},
    School = {Technische Universit{\"a}t Darmstadt},
    Title = {Guided Reinforcement Learning Under Partial Observability},
    Year = {2019}
}
```
