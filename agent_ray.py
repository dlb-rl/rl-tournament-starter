import pickle
import os

import ray
from ray.rllib.agents import ppo
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from agent import Agent


class RayAgent(Agent):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, algorithm: str, checkpoint_path: str):
        """Initialize the RayAgent.
        Args:
            algorithm: The Ray RLlib algorithm to use.
            checkpoint_path: The path to the checkpoint to load.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if checkpoint_path:
            config_dir = os.path.dirname(checkpoint_path)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        # If no pkl file found, require command line `--config`.
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        # create a dummy env since we only care about the policy
        config["env"] = "DummyEnv"
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        # create the Trainer from config
        cls = get_trainable_cls(algorithm)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(checkpoint_path)
        # get policy for evaluation
        self.policy = agent.get_policy("learning_agent")

    def act(self, observation):
        """The act method is called when the agent is asked to act.
        Args:
            observation: the observation of the environment.
        Returns:
            action: np.array representing the action to be taken.
        """
        action = self.policy.compute_single_action(observation)
        # `actions` is a tuple of (action, action_info)
        return action[0]
