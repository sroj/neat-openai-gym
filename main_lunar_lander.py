import argparse

import gym
import neat
import numpy as np
from neat.parallel import ParallelEvaluator

env = gym.make('LunarLander-v2')

n = 100
t_steps = 10000

CONFIG_FILE_NAME = 'config-lunar-lander'
NUM_WORKERS = 1
CHECKPOINT_GENERATION_INTERVAL = 20
CHECKPOINT_PREFIX = 'checkpoint-lunar-lander-'

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     CONFIG_FILE_NAME)


def eval_network(net, net_input):
    return np.argmax(net.activate(net_input))


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

    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)

    parallel_evaluator.evaluate(genomes, neat_config)


def run_neat(checkpoint):
    # Create the population, which is the top-level object for a NEAT run.

    print("Running with {} workers".format(NUM_WORKERS))
    print("Running with checkpoint prefix: {}".format(CHECKPOINT_PREFIX))

    if checkpoint is not None:
        print("Resuming from checkpoint: {}".format(checkpoint))
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Starting run from scratch")
        p = neat.Population(config)

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.add_reporter(neat.Checkpointer(CHECKPOINT_GENERATION_INTERVAL, filename_prefix=CHECKPOINT_PREFIX))

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
    global NUM_WORKERS
    global CHECKPOINT_GENERATION_INTERVAL
    global CHECKPOINT_PREFIX

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', nargs='?', default=None,
                        help='The filename for a checkpoint file to restart from')

    parser.add_argument('--workers', nargs='?', default=NUM_WORKERS, help='How many process workers to spawn')

    parser.add_argument('--gi', nargs='?', default=CHECKPOINT_GENERATION_INTERVAL,
                        help='Maximum number of generations between save intervals')

    parser.add_argument('--checkpoint-prefix', nargs='?', default=CHECKPOINT_PREFIX,
                        help='Prefix for the filename (the end will be the generation number)')

    command_line_args = parser.parse_args()

    NUM_WORKERS = command_line_args.workers

    CHECKPOINT_GENERATION_INTERVAL = command_line_args.gi

    CHECKPOINT_PREFIX = command_line_args.checkpoint_prefix

    return command_line_args


def main():
    command_line_args = parse_args()

    checkpoint = command_line_args.checkpoint

    run_neat(checkpoint)


if __name__ == '__main__':
    main()
