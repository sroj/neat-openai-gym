import neat
import numpy as np

import run_neat_base


def eval_single_genome(genome, genome_config):
    fitness = 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)

    for i_episode in range(100):
        # print("--> Starting new episode")
        observation = run_neat_base.env.reset()

        action = eval_network(net, observation)

        for t in range(4000):

            # env.render()

            observation, reward, done, info = run_neat_base.env.step(action)

            # print("\t Reward {}: {}".format(t, reward))

            action = eval_network(net, observation)

            fitness += reward

            if done:
                # print("<-- Episode finished after {} timesteps".format(t + 1))
                break

    return fitness / 100


def eval_network(net, net_input):
    assert (len(net_input == 4))

    result = np.argmax(net.activate(net_input))

    assert (result == 0 or result == 1)

    return result


def main():
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment_name="CartPole-v1",
                      config_filename="config-cart-pole")


if __name__ == '__main__':
    main()
