import argparse

from agent_ray import RayAgent
import soccer_twos


parser = argparse.ArgumentParser(description="Rollout soccer-twos.")
parser.add_argument("--agent1-checkpoint", help="Team 1 Ray Checkpoint")
parser.add_argument("--agent2-checkpoint", help="Team 2 Ray Checkpoint")
parser.add_argument("--agent1-algorithm", default="PPO", help="Team 1 Algorithm")
parser.add_argument("--agent2-algorithm", default="PPO", help="Team 2 Algorithm")
args = parser.parse_args()

agent1 = RayAgent(
    args.agent1_algorithm,
    args.agent1_checkpoint,
)
agent2 = RayAgent(
    args.agent2_algorithm,
    args.agent2_checkpoint,
)
env = soccer_twos.make(watch=True)
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
    team1_reward += reward[2] + reward[3]
    if max(done.values()):  # if any agent is done
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()
