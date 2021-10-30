import soccer_twos

env = soccer_twos.make(render=True)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)

team0_reward = 0
team1_reward = 0
env.reset()
while True:
    obs, reward, done, info = env.step(
        {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
            3: env.action_space.sample(),
        }
    )

    team0_reward += reward[0] + reward[1]
    team1_reward += reward[2] + reward[3]
    if max(done.values()):  # if any agent is done
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_reward = 0
        team1_reward = 0
        env.reset()
