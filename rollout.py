from ray import tune
from agent_ray import RayAgent
from utils import create_env, create_rllib_env


agent1 = RayAgent(
    "PPO",
    "./ray_results/PPO_selfplay_2/PPO_Soccer_de343_00000_0_2021-09-15_12-30-34/checkpoint_000900/checkpoint-900",
)
agent2 = RayAgent(
    "PPO",
    "./ray_results/PPO_selfplay_1/PPO_Soccer_c5067_00000_0_2021-09-15_01-16-59/checkpoint_000600/checkpoint-600",
)

env = create_env(
    {
        "env_path": "./envs/soccer-twos/soccer-twos.x86_64",
        "render": True,
        "base_port": 50100,
        "time_scale": 1,
        "quality_level": 5,
    }
)
obs = env.reset()
team0_reward = 0
while True:
    # soccer ones
    obs, reward, done, info = env.step(
        {
            0: [agent1.act(obs[0]), agent1.act(obs[0])],
            1: [agent2.act(obs[1]), agent2.act(obs[1])],
        }
    )

    team0_reward += reward[0]
    if max(done.values()):  # if any agent is done
        print("Total team 0 reward", team0_reward)
        team0_reward = 0
        env.reset()
