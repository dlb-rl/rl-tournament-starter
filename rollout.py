from agent_ray import RayAgent
from utils import create_env


agent1 = RayAgent(
    "PPO",
    "./ray_results/PPO_selfplay_2/PPO_Soccer_de343_00000_0_2021-09-15_12-30-34/checkpoint_001400/checkpoint-1400",
    policy_name="learning_agent",
)
agent2 = RayAgent(
    "PPO",
    "./ray_results/PPO_selfplay_twos_1/PPO_Soccer_b5b40_00000_0_2021-09-18_00-22-53/checkpoint_000800/checkpoint-800",
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
team1_reward = 0
while True:
    obs, reward, done, info = env.step(
        {
            0: agent1.act(obs[0]),
            1: agent1.act(obs[1]),
            2: agent2.act(obs[2]),
            3: agent2.act(obs[3]),
        }
    )

    team0_reward += reward[0] + reward[1]
    # team1_reward += reward[1]
    team1_reward += reward[2] + reward[3]
    if max(done.values()):  # if any agent is done
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()
