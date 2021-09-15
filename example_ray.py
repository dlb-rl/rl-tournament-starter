import ray
from ray import tune
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from wrappers import RLLibWrapper


ENVS_PER_WORKER = 1


if __name__ == "__main__":
    ray.init()

    def create_env(env_config=None):
        worker_id = (
            env_config.worker_index * ENVS_PER_WORKER + env_config.vector_index
            if env_config is not None
            else 0
        )
        # TODO update this with the correct env path on your system (the one you downloaded separately)
        build_path = "/home/bryan/Documents/rl/env_baselines/unity/soccer/envs/soccer-ones/soccer-ones.x86_64"
        channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(
            build_path,
            no_graphics=True,
            worker_id=worker_id,
            side_channels=[channel],
        )
        channel.set_configuration_parameters(
            time_scale=20, quality_level=0, target_frame_rate=-1, capture_frame_rate=60
        )
        return RLLibWrapper(unity_env, allow_multiple_obs=True)

    tune.registry.register_env("Soccer", create_env)
    temp_env = create_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_1",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 6,
            "num_envs_per_worker": ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "multiagent": {
                "policies": {
                    "learning_agent": (None, obs_space, act_space, {}),
                    # "opponent_1": (None, obs_space, act_space, {}),
                    # "opponent_2": (None, obs_space, act_space, {}),
                    # "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda x: "learning_agent",  # tune.function(policy_mapping_fn),
                "policies_to_train": ["learning_agent"],
            },
            "env": "Soccer",
        },
        stop={
            "training_iteration": 10000,
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/...",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
