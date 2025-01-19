# RL Playground

This repository contains a unified python implementation of Reinforcement Learning (RL) algorithms presented in
the [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/RLbook2020.pdf) book. 
It also contains several example RL experiments, primarily based on the University of Alberta's [RL Coursera specialization](https://www.coursera.org/specializations/reinforcement-learning).
The example experiments can be run via the [RL Playground notebook](RL%20Playground.ipynb). An [html version](RL%20Playground.html) 
of the notebook with example outputs is also provided. The example experiments are initiated and configured in the notebook, 
but the experiments themselves are defined in the [rl_experiments.py](src/rl_experiments.py) file.

In terms of the core RL code, the primary file is the [rl_agent.py](src/rl_agent.py) file. The file contains a unified 
python implementation of RL algorithms, from tabular methods, to function approximation, to eligibility traces. 
All algorithms can be run using just a pandas/numpy implementation. For convenience, basic Keras and PyTorch implementations
are also provided. Several simplified environments are also provided (to remove any dependencies on gym, etc.). 
The [rl_main.py](src/rl_main.py) file runs the main RL loop for both the agent and the chosen environment. Finally, the [state_representation.py](src/state_representation.py)
and [tiles.py](src/tiles.py) provide ways to represent the state, such as one-hot encoding, state aggregation, and tile coding.