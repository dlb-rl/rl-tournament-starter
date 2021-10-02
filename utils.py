import gym
from ray.rllib import MultiAgentEnv
import soccer_twos

# a RLLib wrapper so our env can inherit from MultiAgentEnv
class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def create_rllib_env(env_config={}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    return RLLibWrapper(env)
