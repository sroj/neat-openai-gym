import neat
import numpy as np

import run_neat_base


def eval_network(net, net_input):
    activation = net.activate(net_input)
    return np.argmax(activation)


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)

    total_reward = 0.0

    # print("--> Starting new episode")
    observation = run_neat_base.env.reset()

    action = eval_network(net, observation)

    done = False

    t = 0

    for i in range(10):
        while not done:

            # run_neat_base.env.render()

            observation, reward, done, info = run_neat_base.env.step(action)

            # print("\t Reward {}: {}".format(t, reward))
            # print("\t Action {}: {}".format(t, action))

            action = eval_network(net, observation)

            total_reward += reward

            t += 1

            if done:
                # print("<-- Episode finished after {} timesteps with reward {}".format(t + 1, genome.fitness))
                break

    return total_reward / 10


def main():
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment_name="LunarLander-v2",
                      config_filename="config-lunar-lander")


if __name__ == '__main__':
    main()
