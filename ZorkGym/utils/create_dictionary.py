import argparse
import itertools
import textworld.agents
from gym.zork_gym import ZorkEnv
from text_utils.text_parser import tokenizer, BasicParser
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='human', metavar='MODE',
                        choices=['test', 'human'],
                        help='Select an agent to play the game: %(choices)s.'
                             ' Default: %(default)s.')
    parser.add_argument('--max-steps', type=int, default=50, metavar='STEPS',
                        help='Limit maximum number of steps.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose mode.')
    parser.add_argument('-t', '--task', default='egg',
                        help='Select game inside Zork1 to play, options are: open (entire quest), egg, troll')
    return parser.parse_args()


def main():
    args = parse_args()
    word_tokenizer = tokenizer
    parser = BasicParser(lambda x: x)

    env = ZorkEnv(task='troll', verbose=args.verbose, word_tokenizer=word_tokenizer, output_parser=parser)

    vocabulary = set()

    walkthrough_file = './solutions/zork_troll_optimal.txt'
    with open(walkthrough_file) as f:
        commands = f.readlines()
    agent = textworld.agents.WalkthroughAgent(commands)

    for _ in range(10):
        seed = np.random.randint(1000)

        done, reward = False, 0
        game_state = env.reset(seed)
        agent.reset(env.env)

        for t in range(args.max_steps) if args.max_steps > 0 else itertools.count():
            try:
                command = agent.act(game_state, reward, done)
            except:
                command = 'kill troll with sword'

            try:
                game_state, reward, done, has_won = env.step(command)
            except:
                game_state, reward, done, has_won = [''], 0, True, False

            for state in game_state:
                for item in state:
                    vocabulary.add(item)

            env.render()

            if done:
                break

    env = ZorkEnv(task='egg', verbose=args.verbose, word_tokenizer=word_tokenizer, output_parser=parser)

    walkthrough_file = './solutions/zork_egg_optimal.txt'
    with open(walkthrough_file) as f:
        commands = f.readlines()
    agent = textworld.agents.WalkthroughAgent(commands)

    for _ in range(10):
        seed = np.random.randint(1000)

        done, reward = False, 0
        game_state = env.reset(seed)
        agent.reset(env.env)

        for t in range(args.max_steps) if args.max_steps > 0 else itertools.count():
            command = agent.act(game_state, reward, done)

            try:
                game_state, reward, done, has_won = env.step(command)
            except:
                game_state, reward, done, has_won = [''], 0, True, False

            for state in game_state:
                for item in state:
                    vocabulary.add(item)

            env.render()

            if done:
                break

    f = open('vocabulary.txt', 'w')
    f.write(','.join(map(str, vocabulary)))


if __name__ == '__main__':
    main()
