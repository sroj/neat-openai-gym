import neat
import numpy as np

import run_neat_base


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    for i in range(run_neat_base.n):
        # print("--> Starting new episode")
        observation = run_neat_base.env.reset()

        action = eval_network(net, observation)

        done = False

        while not done:

            # env.render()

            observation, reward, done, info = run_neat_base.env.step(action)

            # print("\t Reward {}: {}".format(t, reward))

            action = eval_network(net, observation)

            total_reward += reward

            if done:
                # print("<-- Episode finished after {} timesteps".format(t + 1))
                break

    return total_reward / run_neat_base.n


def eval_network(net, net_input):
    assert (len(net_input == 4))

    result = np.argmax(net.activate(net_input))

    assert (result == 0 or result == 1)

    return result


def main():
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment_name="CartPole-v0")


if __name__ == '__main__':
    main()
