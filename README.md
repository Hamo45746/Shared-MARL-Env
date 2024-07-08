# 8/07/2024
## Potential Environment Improvements 
Make more training maps \
Cut observations at buildings \
Observation map at the end of simulation to show full observed / discovered area \
Ensure Task Allocation and Continuous agents can intergrate together \
Create interface doc \
Gazebo integration 


## Current banches:
Main, env2, pygame-rendering, Modular, broken-observations: Stages of development for the final environment. \
Temporary: Current working environment, training with stable baselines3 \
marllib - integration: Branch of the current working environment to integrate with MARLlib \

## MARL/Rl open source libraries for use to look into integrating for benchmarking:
### Stable Baselines3: https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
  - https://github.com/DLR-RM/stable-baselines3
### RLLib: https://docs.ray.io/en/latest/rllib/index.html
  - https://docs.ray.io/en/latest/rllib/key-concepts.html
  - By anyscale, uses Ray https://www.anyscale.com/ray-open-source
  - https://github.com/ray-project/ray/tree/master/rllib
### MARLLib: https://marllib.readthedocs.io/en/latest/index.html
  - Extension of RLlib but not by the Rllib team
  - Github: https://github.com/Replicable-MARL/MARLlib/tree/master
  - Great documentation 
  - Says that it is only working with Linux 
### AgileRL: https://docs.agilerl.com/en/latest/
  - Compatible with gymnasium 
  - Compatible with PettingZoo: https://pettingzoo.farama.org/tutorials/agilerl/
  - Only has MADDPG or MATD3 for multi-agent 
### EPyMarl: https://agents.inf.ed.ac.uk/blog/epymarl/
  - Compatible with OpenAI gym
  - Github: https://github.com/uoe-agents/epymarl

