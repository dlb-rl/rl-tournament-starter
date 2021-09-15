from typing import Any, Dict, List, Optional, Tuple, Union

from ray.rllib import MultiAgentEnv
import numpy as np
import gym
from gym import spaces
from gym_unity.envs import UnityToGymWrapper, UnityGymException, ActionFlattener
from mlagents_envs import logging_util
from mlagents_envs.base_env import ActionTuple, BaseEnv, DecisionSteps, TerminalSteps


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)
GymStepResult = Tuple[np.ndarray, float, bool, Dict]


class TerminationMode:
    ALL = "ALL"
    ANY = "ANY"
    MAJORITY = "MAJORITY"


class MultiAgentUnityWrapper(UnityToGymWrapper):
    """An implementation of the UnityToGymWrapper that supports multi-agent environments.
    Based on `UnityToGymWrapper` from the Unity's ML-Toolkits [1] and on ai-traineree modifications [2]:
    Updated to work with ml-agents v0.27.0.

    At the time of writting the official package doesn't support multi agents.
    Until it's clear why it doesn't support [3] and whether they plan on adding
    anything, we're keeping this version. When the fog of unknown has been
    blown away, we might consider doing a Pull Request to `ml-agents`.

    [1]: https://github.com/Unity-Technologies/ml-agents/blob/56e6d333a52863785e20c34d89faadf0a115d320/gym-unity/gym_unity/envs/__init__.py
    [2]: https://github.com/laszukdawid/ai-traineree/blob/a9d89b458e40724211d4a0cc8331886dead3eb57/ai_traineree/tasks.py#L260
    [3]: https://github.com/Unity-Technologies/ml-agents/issues/4120
    """

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
        action_space_seed: Optional[int] = None,
        termination_mode: str = TerminationMode.ANY,
    ):
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        :param action_space_seed: If non-None, will be used to set the random seed on created gym.Space instances.
        :termination_mode: A string (enum) suggesting when to end an episode. Supports "ANY", "MAJORITY" and "ALL"
            which are atributes on `TerminationMode`.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        # < multiagent mod > (removed len check)
        # When to stop the game, considering all agents
        assert termination_mode in TerminationMode.__dict__
        self.termination_mode = termination_mode

        self.name = list(self._env.behavior_specs.keys())[0]
        # < multiagent mod > (added agent_prefix)
        self.agent_prefix = self.name[: self.name.index("=") + 1]
        self.group_spec = self._env.behavior_specs[self.name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise UnityGymException(
                "There are no observations provided by the environment."
            )

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual
        if (
            self._get_n_vis_obs() + self._get_vec_obs_size() >= 2
            and not self._allow_multiple_obs
        ):
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.name)
        # < multiagent mod > (removed len check)
        self.num_agents = len(self._env.behavior_specs)
        self._previous_decision_step = decision_steps

        # Set action spaces
        if self.group_spec.action_spec.is_discrete():
            self.action_size = self.group_spec.action_spec.discrete_size
            branches = self.group_spec.action_spec.discrete_branches
            if self.group_spec.action_spec.discrete_size == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        elif self.group_spec.action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )

            self.action_size = self.group_spec.action_spec.continuous_size
            high = np.array([1] * self.group_spec.action_spec.continuous_size)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            raise UnityGymException(
                "The gym wrapper does not provide explicit support for both discrete "
                "and continuous actions."
            )

        if action_space_seed is not None:
            self._action_space.seed(action_space_seed)

        # Set observations space
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if uint8_visual:
                list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if self._allow_multiple_obs:
            self._observation_space = spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # only return the first one

    def reset(self) -> Union[Dict[int, np.ndarray], np.ndarray]:
        """Resets the state of the environment and returns an initial observation.
        If the number of agents is greater than one, the observations will be a dict.
        Returns:
            observation (object/list): the initial observation of the space.
        """
        self._env.reset()
        # < multiagent mod >
        if self.num_agents > 1:
            states = {}
            for agent_id in range(self.num_agents):
                decision_step, _ = self._env.get_steps(
                    self.agent_prefix + str(agent_id)
                )
                self.game_over = False
                res: GymStepResult = self._single_step(decision_step)
                states[agent_id] = res[0]
            return states
        else:
            decision_step, _ = self._env.get_steps(self.name)
            self.game_over = False
            res: GymStepResult = self._single_step(decision_step)
            return res[0]  # res contains tuple with `state` on first pos

    def step(self, action: Union[Dict[int, List[Any]], List[Any]]) -> GymStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object/list): a single action or a dict with actions for each agent (multiagent setting)
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        if self.game_over:
            raise UnityGymException(
                "You are calling 'step()' even though this environment has already "
                "returned done = True. You must always call 'reset()' once you "
                "receive 'done = True'."
            )

        # < multiagent mod >
        if self.num_agents > 1:
            assert (
                type(action) is dict
            ), "The environment requires a dictionary for multi-agent setting."
            for agent_id in action:
                self.set_action(action[agent_id], self.agent_prefix + str(agent_id))
        else:
            self.set_action(action, self.name)

        self._env.step()

        # < multiagent mod >
        if type(action) is dict:
            obs_dict = {}
            rew_dict = {}
            done_dict = {}
            info_dict = {}
            for agent_id in action:
                o, r, d, i = self.get_step_results(self.agent_prefix + str(agent_id))
                obs_dict[agent_id] = o
                rew_dict[agent_id] = r
                done_dict[agent_id] = d
                info_dict[agent_id] = i
            done_dict["__all__"] = max(done_dict.values())
            return obs_dict, rew_dict, done_dict, info_dict
        else:
            return self.get_step_results(self.name)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        obs, rew, done, _info = super()._single_step(info)
        rew += info.group_reward[0]
        return obs, rew, done, _info

    # < multiagent mod >
    def detect_game_over(self, terminal_steps: List[TerminalSteps]) -> bool:
        """Determine whether the episode has finished.

        Expects the `terminal_steps` to contain only steps that terminated. Note that other steps
        are possible in the same iteration.
        This is to keep consistent with Unity's framework but likely will go through refactoring.

        Args:
            terminal_steps (list): list of all the steps that terminated.
        """
        if self.termination_mode == TerminationMode.ANY and len(terminal_steps) > 0:
            return True
        elif (
            self.termination_mode == TerminationMode.MAJORITY
            and len(terminal_steps) > 0.5 * self.num_agents
        ):
            return True
        elif (
            self.termination_mode == TerminationMode.ALL
            and len(terminal_steps) == self.num_agents
        ):
            return True
        else:
            return False

    def set_action(self, action: List[Any], agent_name: str) -> None:
        """Sets the action for an agent within the environment.
        Args:
            action (list): the action to take
            agent_name (str): the name of the agent to set the action for
        """

        if self._flattener is not None:
            # Translate action into list
            action = self._flattener.lookup_action(action)

        action = np.array(action).reshape((-1, self.action_size))

        action_tuple = ActionTuple()
        if self.group_spec.action_spec.is_continuous():
            action_tuple.add_continuous(action)
        else:
            action_tuple.add_discrete(action)

        self._env.set_actions(agent_name, action_tuple)

    def get_step_results(self, agent_name: str) -> GymStepResult:
        """Returns the observation, reward and whether the episode is over after taking the action.
        Args:
            agent_name (str): the name of the agent to get the step results for
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        decision_step, terminal_step = self._env.get_steps(agent_name)

        if self.detect_game_over(terminal_step):
            self.game_over = True
            out = self._single_step(terminal_step)
            self.reset()  # TODO: This is a hack to allow remaining agents to "do something". Remove!
            return out
        else:
            return self._single_step(decision_step)


class RLLibWrapper(MultiAgentUnityWrapper, MultiAgentEnv):
    pass
