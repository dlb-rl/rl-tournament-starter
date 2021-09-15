from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from wrappers import MultiAgentUnityWrapper, RLLibWrapper


def create_env(env_config={}):
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        env_config.get("env_path", "./envs/soccer-ones/soccer-ones.x86_64"),
        no_graphics=not env_config.get("render", False),
        base_port=env_config.get("base_port", 40039),
        worker_id=env_config.get("worker_id", 0),
        side_channels=[channel],
    )
    channel.set_configuration_parameters(
        time_scale=env_config.get("time_scale", 20),
        quality_level=env_config.get("quality_level", 0),
        target_frame_rate=-1,
        capture_frame_rate=60,
    )
    return MultiAgentUnityWrapper(unity_env)


def create_rllib_env(env_config={}):
    env_config["worker_id"] = env_config.get("worker_index", 0) * env_config.get(
        "num_envs_per_worker", 1
    ) + env_config.get("vector_index", 0)
    env = create_env(env_config)
    return RLLibWrapper(env)
