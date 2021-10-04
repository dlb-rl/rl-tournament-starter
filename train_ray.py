import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"  # Choose 01 policy for agent_01
    else:
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            size=1,
            p=[0.50, 0.25, 0.125, 0.125],
        )[0]


class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when reward is high enough
        """
        if info["result"]["episode_reward_mean"] > 0.5:
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_twos_2",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                # "env_path": "/home/bryan/Documents/rl/env_baselines/unity/soccer/envs/soccer-ones/soccer-ones.x86_64",
                "env_path": "/home/bryan/Documents/rl/env_baselines/unity/soccer/envs/soccer-twos/soccer-twos.x86_64",
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
            # "exploration_config": {
            #     "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            #     "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            #     "lr": 0.0003,  # Learning rate of the curiosity (ICM) module.
            #     "feature_dim": 64,  # Dimensionality of the generated feature vectors.
            #     # Setup of the feature net (used to encode observations into feature (latent) vectors).
            #     "feature_net_config": {
            #         "fcnet_hiddens": [256],
            #         "fcnet_activation": "relu",
            #     },
            #     "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            #     "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            #     "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            #     "forward_net_activation": "relu",  # Activation of the "forward" model.
            #     "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            #     # Specify, which exploration sub-type to use (usually, the algo's "default"
            #     # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            #     "sub_exploration": {
            #         "type": "StochasticSampling",
            #     },
            # },
        },
        stop={
            "timesteps_total": 15000000,
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="./ray_results/PPO_selfplay_twos_2/PPO_Soccer_a8b44_00000_0_2021-09-18_11-13-55/checkpoint_000600/checkpoint-600",
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
