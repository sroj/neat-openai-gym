import gym
import neat
from gym import envs

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-bipedal-walker')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

env = gym.make('BipedalWalker-v2')

n = 100


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 1.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # print("--> Starting new episode")
        observation = env.reset()

        action = eval_network(net, observation)

        done = False

        t = 0

        while not done:

            # env.render()

            observation, reward, done, info = env.step(action)

            # print("\t Reward {}: {}".format(t, reward))
            # print("\t Action {}: {}".format(t, action))

            action = eval_network(net, observation)

            genome.fitness += reward

            t += 1

            if done:
                # print("<-- Episode finished after {} timesteps with reward {}".format(t + 1, genome.fitness))
                pass


def eval_network(net, net_input):
    # assert (len(net_input) == 24)

    return net.activate(net_input)

    # assert (len(result) == 4)

    # assert (result[0] >= -1.0 or result[0] <= 1.0)


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

        done = False

        t = 0

        while not done:

            # env.render()

            observation, reward, done, info = env.step(action)

            # print("\t Observation {}: {}".format(t, observation))
            # print("\t Info {}: {}".format(t, info))

            action = eval_network(net, observation)

            reward_episode += reward

            # print("\t Reward {}: {}".format(t, reward))

            t += 1

            if done:
                # print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                pass

        avg_reward += reward_episode / n

    print("Average reward was: {}".format(avg_reward))


def main():
    print("Available episodes are: ")
    print(envs.registry.all())

    run_neat()


if __name__ == '__main__':
    main()
