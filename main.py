import gym


def main():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        print("--> Starting new episode")
        observation = env.reset()
        for t in range(100):
            env.render()
            print("\t Observation {}: {}".format(t, observation))
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("<-- Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    main()
