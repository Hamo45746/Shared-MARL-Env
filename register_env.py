from MARLlib.marllib.envs.base_env import ENV_REGISTRY
from env import Environment

def env_creator(env_config):
    return Environment(**env_config)

ENV_REGISTRY["Thesis_Env"] = env_creator