import neat
import numpy as np

import run_neat_base


def eval_network(net, net_input):
    net_input_array = np.zeros(6)
    net_input_array[net_input] = 1
    activation = net.activate(net_input_array)

    # slice1 = activation[0:2]
    # slice2 = activation[2:4]
    # slice3 = activation[4:9]

    maxarg = np.argmax(activation)

    d2 = maxarg // 10

    d1 = (maxarg % 10) // 5

    d0 = (maxarg % 10) % 5

    return [d2, d1, d0]


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    for i in range(run_neat_base.n):
        # print("--> Starting new episode")
        observation = run_neat_base.env.reset()

        done = False

        t = 0

        while not done:

            # run_neat_base.env.render()
            action = eval_network(net, observation)

            # print("\t Observation {}: {}".format(t, observation))

            observation, reward, done, info = run_neat_base.env.step(action)

            # print("\t Action {}: {}".format(t, action))
            # print("\t Reward {}: {}".format(t, reward))

            total_reward += reward

            t += 1

            if done:
                # print("<-- Episode finished after {} timesteps with reward {}".format(t + 1, total_reward))
                break

    return total_reward / run_neat_base.n


def main():
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment_name="RepeatCopy-v0")


if __name__ == '__main__':
    main()
