# gym-pomdp-wrappers

**gym-pomdp-wrappers** is a set of wrappers that can be used to turn some specific OpenAI gym environments into Partially Observable Markov Decision Processes (POMDPs).


## Dependencies

This pacakge requires python3 (>=3.5). Additionally, you need the following packages:

* [OpenAI Gym](https://github.com/openai/gym)
* [gym_pomdp](https://github.com/d3sm0/gym_pomdp)


## Installation

After installaling the dependencies, you can perform a install of **gym-pomdp-wrappers** with:

.. code:: shell

    git clone https://github.com/stweigand/gym-pomdp-wrappers.git
    cd gym-pomdp-wrappers
    pip install -e .


## Implemented Wrappers

[WIP]


## Example

[WIP]


## MuJoCo

Some of the wrappers use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)


## Citing the Project

This repository is part of my Master's thesis which is about "Guided Reinforcement Learning Under Partial Observability". (I will later add a link here)
Here is a BibTeX entry that you can use to cite it in a publication::

```
    @mastersthesis{guidedreinforcementlearning,
    Author = {Stephan Weigand},
    School = {Technische Universit{\"a}t Darmstadt},
    Title = {Guided Reinforcement Learning Under Partial Observability},
    Year = {2019}}
```
