import argparse
from functools import partial

import gym
import neat
from neat.parallel import ParallelEvaluator

import datetime

n = 100
T_STEPS = 10000

ENVIRONMENT_NAME = None
CONFIG_FILENAME = None

NUM_WORKERS = 1
CHECKPOINT_GENERATION_INTERVAL = 20
CHECKPOINT_PREFIX = None

env = None

config = None


def _eval_genomes(eval_single_genome, genomes, neat_config):
    print("Evaluating genomes")

    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)

    parallel_evaluator.evaluate(genomes, neat_config)


def _run_neat(checkpoint, eval_network, eval_single_genome):
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
    winner = p.run(partial(_eval_genomes, eval_single_genome))

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    net = neat.nn.FeedForwardNetwork.create(winner, config)

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
                print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                pass

        avg_reward += reward_episode / n

    print("Average reward was: {}".format(avg_reward))


def _parse_args():
    global NUM_WORKERS
    global CHECKPOINT_GENERATION_INTERVAL
    global CHECKPOINT_PREFIX

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', nargs='?', default=None,
                        help='The filename for a checkpoint file to restart from')

    parser.add_argument('--workers', nargs='?', type=int, default=NUM_WORKERS, help='How many process workers to spawn')

    parser.add_argument('--gi', nargs='?', type=int, default=CHECKPOINT_GENERATION_INTERVAL,
                        help='Maximum number of generations between save intervals')

    parser.add_argument('--checkpoint-prefix', nargs='?', default=CHECKPOINT_PREFIX,
                        help='Prefix for the filename (the end will be the generation number)')

    command_line_args = parser.parse_args()

    NUM_WORKERS = command_line_args.workers

    CHECKPOINT_GENERATION_INTERVAL = command_line_args.gi

    CHECKPOINT_PREFIX = command_line_args.checkpoint_prefix

    return command_line_args


def run(eval_network, eval_single_genome, environment_name, config_filename):
    global ENVIRONMENT_NAME
    global CONFIG_FILENAME
    global env
    global config
    global CHECKPOINT_PREFIX

    ENVIRONMENT_NAME = environment_name
    CONFIG_FILENAME = config_filename

    env = gym.make(ENVIRONMENT_NAME)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_FILENAME)

    command_line_args = _parse_args()

    checkpoint = command_line_args.checkpoint

    if CHECKPOINT_PREFIX is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        CHECKPOINT_PREFIX = "cp_" + environment_name.lower() + "_" + timestamp

    _run_neat(checkpoint, eval_network, eval_single_genome)
