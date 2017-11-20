import argparse

import gym
import neat
from neat.parallel import ParallelEvaluator

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-bipedal-walker')

env = gym.make('BipedalWalker-v2')

n = 100
t_steps = 10000

num_workers = 1
checkpoint_generation_interval = 20
checkpoint_prefix = 'neat-checkpoint-'


def eval_network(net, net_input):
    return net.activate(net_input)


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)

    total_reward = 0.0

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

        total_reward += reward

        t += 1

        if done:
            # print("<-- Episode finished after {} timesteps with reward {}".format(t + 1, genome.fitness))
            break

    return total_reward


def eval_genomes(genomes, neat_config):
    print("Evaluating genomes")
    
    parallel_evaluator = ParallelEvaluator(num_workers, eval_function=eval_single_genome)

    parallel_evaluator.evaluate(genomes, neat_config)


def run_neat(checkpoint):
    # Create the population, which is the top-level object for a NEAT run.

    print("Running with {} workers".format(num_workers))
    print("Running with checkpoint prefix: {}".format(checkpoint_prefix))

    if checkpoint is not None:
        print("Resuming from checkpoint: {}".format(checkpoint))
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Starting run from scratch")
        p = neat.Population(config)

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.add_reporter(neat.Checkpointer(checkpoint_generation_interval, filename_prefix=checkpoint_prefix))

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', nargs='?', default=None,
                        help='The filename for a checkpoint file to restart from')

    parser.add_argument('--workers', nargs='?', default=1, help='How process workers to spawn')

    parser.add_argument('--gi', nargs='?', default=20, help='Maximum number of generations between save intervals')

    parser.add_argument('--checkpoint-prefix', nargs='?', default='neat-checkpoint-',
                        help='Prefix for the filename (the end will be the generation number)')

    command_line_args = parser.parse_args()

    return command_line_args


def main():
    command_line_args = parse_args()

    global num_workers
    num_workers = command_line_args.workers

    global checkpoint_generation_interval
    checkpoint_generation_interval = command_line_args.gi

    global checkpoint_prefix
    checkpoint_prefix = command_line_args.checkpoint_prefix

    checkpoint = command_line_args.checkpoint

    run_neat(checkpoint)


if __name__ == '__main__':
    main()
