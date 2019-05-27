import argparse
import itertools
import textworld.agents
from gym.zork_gym import ZorkEnv
from text_utils.text_parser import tokenizer, BasicParser
import pickle
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='human', metavar='MODE',
                        choices=['test', 'human'],
                        help='Select an agent to play the game: %(choices)s.'
                             ' Default: %(default)s.')
    parser.add_argument('--max-steps', type=int, default=0, metavar='STEPS',
                        help='Limit maximum number of steps.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose mode.')
    parser.add_argument('-t', '--task', default='egg',
                        choices=['egg', 'troll', 'egg_troll', 'full'],
                        help='Select game inside Zork1 to play, options are: open (entire quest), egg, troll')
    return parser.parse_args()


def main():
    args = parse_args()
    word_tokenizer = tokenizer
    parser = BasicParser(lambda x: x)

    env = ZorkEnv(task=args.task, verbose=args.verbose, word_tokenizer=word_tokenizer, output_parser=parser,
                  max_steps=420, pomdp_mode=False)

    data_dict = {'states': [], 'actions': []}

    if args.mode == 'human':
        agent = textworld.agents.HumanAgent()
    elif args.mode == 'test':
        if args.task == 'egg':
            walkthrough_file = './solutions/zork_egg_optimal.txt'
        elif args.task == 'troll':
            walkthrough_file = './solutions/zork_troll_optimal.txt'
        elif args.task =='egg_troll':
            walkthrough_file = './solutions/zork_egg_troll_optimal.txt'
        else:
            walkthrough_file = './solutions/zork_full_optimal.txt'
        with open(walkthrough_file) as f:
            commands = f.readlines()
        agent = textworld.agents.WalkthroughAgent(commands)
        agent.reset(env.env)

    seed = 12

    done, reward = False, 0
    game_state = env.reset(seed)

    data_dict['states'].append(game_state)

    agent.reset(env.env)

    if args.mode == 'human' or args.verbose:
        env.render()

    for t in range(args.max_steps) if args.max_steps > 0 else itertools.count():
        command = agent.act(game_state, reward, done)
        data_dict['actions'].append(command)

        game_state, reward, done, has_won = env.step(command)
        data_dict['states'].append(game_state)

        if args.mode == 'human' or args.verbose:
            print(command)
            env.render()

        if done:
            break

    with open(os.getcwd() + '/../zork_walkthrough.txt', 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == '__main__':
    main()
