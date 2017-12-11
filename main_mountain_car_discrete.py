import neat

import run_neat_base

import numpy as np


def eval_network(net, net_input):
    assert (len(net_input) == 2)

    result = np.argmax(net.activate(net_input))

    assert (result in range(0, 3))

    return result


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    for i in range(run_neat_base.n):
        # print("--> Starting new episode")
        observation = run_neat_base.env.reset()

        action = eval_network(net, observation)

        done = False

        while not done:

            # run_neat_base.env.render()

            observation, reward, done, info = run_neat_base.env.step(action)

            # print("\t Reward {}: {}".format(t, reward))
            # print("\t Action {}: {}".format(t, action))

            action = eval_network(net, observation)

            total_reward += reward

            if done:
                # print("<-- Episode finished after {} time-steps with reward {}".format(t + 1, genome.fitness))
                break

    return total_reward / run_neat_base.n


def main():
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment_name="MountainCar-v0",
                      config_filename="config-mountain-car-discrete")


if __name__ == '__main__':
    main()
