from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from wrappers import MultiAgentUnityWrapper


RUN_1V1 = True

if RUN_1V1:
    print("Running 1v1")
    build_path = "./envs/soccer-ones/soccer-ones.x86_64"
else:
    print("Running 2v2")
    build_path = "./envs/soccer-twos/soccer-twos.x86_64"

channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(build_path, no_graphics=False, side_channels=[channel])
channel.set_configuration_parameters(
    time_scale=20, quality_level=0, target_frame_rate=-1, capture_frame_rate=60
)
env = MultiAgentUnityWrapper(unity_env=unity_env, allow_multiple_obs=True)

print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)

team0_reward = 0
while True:
    if RUN_1V1:
        # soccer ones
        obs, reward, done, info = env.step(
            {
                # random actions for both teams
                0: env.action_space.sample(),
                1: env.action_space.sample(),
            }
        )
    else:
        # soccer twos
        obs, reward, done, info = env.step(
            {
                0: [[1, 0, 0], [1, 0, 0]],  # team 0 goes forward
                1: [[0, 0, 1], [0, 0, 1]],  # team 1 spins
            }
        )

    team0_reward += reward[0]
    if max(done.values()):  # if any agent is done
        print("Total team 0 reward", team0_reward)
        team0_reward = 0
        env.reset()
