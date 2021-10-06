from enum import Enum
from typing import Callable

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class MultiagentTeamEnv(gym.core.Wrapper, MultiAgentEnv):
    """
    A wrapper for multiagent team-controlled environment.
    Uses a 2x2 (4 players) environment to expose a 1x1 (2 teams) environment.
    """

    def __init__(self, env):
        super(MultiagentTeamEnv, self).__init__(env)
        self.env = env
        # duplicate obs & action spaces (concatenate players)
        self.observation_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(env.observation_space.shape[0] * 2,)
        )
        self.action_space = gym.spaces.MultiDiscrete(
            np.repeat(env.action_space.nvec, 2)
        )
        self.action_space_n = len(env.action_space.nvec)

    def step(self, action):
        action = {
            # slice actions for team 1
            0: action[0][: self.action_space_n],
            1: action[0][self.action_space_n :],
            # slice actions for team 2
            2: action[1][: self.action_space_n],
            3: action[1][self.action_space_n :],
        }
        obs, reward, done, info = self.env.step(action)
        return self._preprocess_obs(obs), self._preprocess_reward(reward), done, info

    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        return {
            0: np.concatenate((obs[0], obs[1])),
            1: np.concatenate((obs[2], obs[3])),
        }

    def _preprocess_reward(self, reward):
        return {
            0: reward[0] + reward[1],
            1: reward[2] + reward[3],
        }


class TeamVsPolicyEnv(gym.core.Wrapper):
    """
    A wrapper for team vs given policy environment.
    Uses random policy as opponent by default.
    Uses a 2x2 (4 players) environment to expose a 1x1 (2 teams) environment.
    """

    def __init__(self, env, opponent: Callable = None):
        super(TeamVsPolicyEnv, self).__init__(env)
        self.env = env

        # duplicate obs & action spaces
        self.observation_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(env.observation_space.shape[0] * 2,)
        )
        self.action_space = gym.spaces.MultiDiscrete(
            np.repeat(env.action_space.nvec, 2)
        )
        self.action_space_n = len(env.action_space.nvec)

        if opponent is None:
            # a function that returns random actions no matter the input
            self.opponent = lambda *_: self.env.action_space.sample()
        else:
            self.opponent = opponent

        self.last_obs = None

    def step(self, action):
        action = {
            # slice actions for team 1
            0: action[: self.action_space_n],
            1: action[self.action_space_n :],
            # slice actions for team 2
            2: self.opponent(self.last_obs[2]),
            3: self.opponent(self.last_obs[3]),
        }
        obs, reward, done, info = self.env.step(action)
        return self._preprocess_obs(obs), self._preprocess_reward(reward), done, info

    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        self.last_obs = obs
        return np.concatenate((obs[0], obs[1]))

    def _preprocess_reward(self, reward):
        return reward[0] + reward[1]


class EnvType(Enum):
    multiagent_player = "multiagent_player"
    multiagent_team = "multiagent_team"
    team_vs_policy = "team_vs_policy"


def create_rllib_env(
    env_config: dict = {},
):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - type: one of EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = RLLibWrapper(env)

    if "type" in env_config:
        if env_config["type"] == EnvType.multiagent_player:
            return env
        elif env_config["type"] == EnvType.multiagent_team:
            return MultiagentTeamEnv(env)
        elif env_config["type"] == EnvType.team_vs_policy:
            return TeamVsPolicyEnv(
                env,
                env_config["opponent_policy"]
                if "opponent_policy" in env_config
                else None,
            )
        else:
            raise ValueError(
                "EnvType invalid. Must be one of ", [e.value for e in EnvType]
            )
