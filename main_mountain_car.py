import gym
import neat
import numpy as np
from gym import envs

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mountain-car')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

env = gym.make('MountainCarContinuous-v0')

n = 100
t_steps = 5000


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 1.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for i_episode in range(1):
            # print("--> Starting new episode")
            observation = env.reset()

            action = eval_network(net, observation)

            for t in range(t_steps):

                # env.render()

                observation, reward, done, info = env.step(action)

                # print("\t Reward {}: {}".format(t, reward))
                # print("\t Action {}: {}".format(t, action))

                action = eval_network(net, observation)

                genome.fitness += reward

                if done:
                    # print("<-- Episode finished after {} timesteps with reward {}".format(t + 1, genome.fitness))
                    break


def eval_network(net, net_input):
    assert (len(net_input == 2))

    result = net.activate(net_input)

    assert (result[0] >= -1.0 or result[0] <= 1.0)

    return result


def run_neat():
    # Run until a solution is found.
    winner = p.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    n = 100

    avg_reward = 0

    for i_episode in range(n):
        print("--> Starting test episode trial {}".format(i_episode + 1))
        observation = env.reset()

        action = eval_network(net, observation)

        reward_episode = 0

        for t in range(t_steps):

            env.render()

            observation, reward, done, info = env.step(action)

            # print("\t Observation {}: {}".format(t, observation))
            # print("\t Info {}: {}".format(t, info))

            action = eval_network(net, observation)

            reward_episode += reward

            # print("\t Reward {}: {}".format(t, reward))

            if done:
                print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                break

        avg_reward += reward_episode / n

    print("Average reward was: {}".format(avg_reward))


def main():
    print("Available episodes are: ")
    print(envs.registry.all())

    run_neat()


if __name__ == '__main__':
    main()
