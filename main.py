import random
import torch
import numpy as np
import gensim
import gensim.downloader as glove_api
import argparse
import os

from ZorkGym.text_utils.text_parser import BagOfWords, Word2Vec, TextParser, tokenizer
from agents.DRRN import DRRN
from agents.DQN import DQN
from agents.Discrete_DDPG import DDDPG
from agents.OMP_DDPG import OMPDDPG


def load_list_from_file(file_path):
    with open(file_path) as file:
        content = file.readlines()
    ret = []
    for elem in content:
        clean_elem = elem.strip()
        if len(clean_elem) > 0:
            ret.append(clean_elem)
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=0, type=int)
    parser.add_argument('--simulations', nargs='+', default=['dqn_mlp'])
    parser.add_argument('--pomdp', action='store_true', default=False)
    parser.add_argument('--action_w2v', action='store_true', default=False)
    parser.add_argument('--linear', action='store_true', default=False)
    parser.add_argument('--grad_loss', action='store_true', default=False)
    parser.add_argument('--improved_omp', action='store_true', default=False)
    parser.add_argument('--nn', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--test_interval', default=5000, type=int)
    return parser.parse_args()


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def main():
    args = parse_args()

    if args.grad_loss:
        loss_weighting = 0.0
    else:
        loss_weighting = 1.0

    verbs = ['go', 'take', 'open', 'grab', 'run', 'walk', 'climb']
    vocabulary = load_list_from_file('./data/vocabulary.txt')

    basic_actions = ['open egg', 'go east', 'go west', 'go north', 'go south', 'go up', 'go down', 'look', 'take egg']

    dictionary = ['pray', 'yellow', 'trapdoor', 'open', 'bell', 'touch', 'pile', 'trunk', 'sack', 'inflate',
                  'southeast', 'of', 'move', 'match', 'figurine', 'railing', 'with', 'map', 'mirror', 'wind', 'examine',
                  'north', 'out', 'trident', 'turn', 'skull', 'throw', 'northwest', 'case', 'bag', 'red', 'press',
                  'jewels', 'east', 'pump', 'bolt', 'rusty', 'window', 'douse', 'boat', 'bracelet', 'matchbook',
                  'basket', 'book', 'coffin', 'bar', 'rug', 'lid', 'drop', 'nasty', 'wrench', 'light', 'sand', 'bauble',
                  'kill', 'tie', 'painting', 'sword', 'wave', 'in', 'south', 'northeast', 'ring', 'canary', 'lower',
                  'egg', 'all', 'to', 'candles', 'page', 'and', 'echo', 'emerald', 'tree', 'from', 'rope', 'troll',
                  'screwdriver', 'torch', 'enter', 'coal', 'go', 'look', 'shovel', 'knife', 'down', 'take', 'switch',
                  'prayer', 'launch', 'diamond', 'read', 'up', 'get', 'scarab', 'west', 'land', 'southwest', 'climb',
                  'thief', 'raise', 'wait', 'odysseus', 'button', 'sceptre', 'lamp', 'chalice', 'garlic', 'buoy', 'pot',
                  'label', 'put', 'dig', 'machine', 'close']

    actions = basic_actions

    optimize_memory = False
    sparse_reward = True
    actor_train_start = 0
    eps_start = 1.0

    test_params = {
        'nn=-1': {'number_of_neighbors': -1},
        'nn=1': {'number_of_neighbors': 1},
        'nn=3': {'number_of_neighbors': 3},
        'nn=11': {'number_of_neighbors': 11},
    }

    game_seed = 52
    if args.task == 0:
        buffer_size = 20000
        time_steps = 100000
        project_name = 'egg_quest_minimal_actions'
        task = 'egg'
    elif args.task == 1:
        buffer_size = 20000
        time_steps = 2000000
        project_name = 'egg_quest_extended_actions'
        actions = dictionary
        task = 'egg'
    elif args.task == -1:
        buffer_size = 20000
        time_steps = 100000
        project_name = 'egg_quest_baby_actions'
        actions = ['open', 'egg', 'north', 'climb', 'tree', 'take']
        task = 'egg'
    elif args.task == 2:
        buffer_size = 40000
        time_steps = 1000000
        project_name = 'troll_imitation'
        actions = dictionary
        task = 'troll'
        sparse_reward = False

        test_params = {
            'nn': {'number_of_neighbors': args.nn},
        }
        game_seed = 12
    elif args.task == 3:
        buffer_size = 40000
        time_steps = 1000000
        project_name = 'troll'
        actions = ['north', 'south', 'east', 'west', 'open window', 'take sword', 'take lamp', 'move rug',
                   'open trapdoor', 'go down', 'light lamp', 'kill troll with sword']
        task = 'troll'
        sparse_reward = False
    else:
        raise NotImplementedError

    words = list()
    words.append('')
    for action in actions:
        tokens = tokenizer(action)
        for token in tokens:
            if token not in words:
                words.append(token)

    sentences = list()
    for i, word1 in enumerate(words):
        for word2 in words[i + 1:]:
            if word1 in verbs:
                sentences.append(word1 + ' ' + word2)
            else:
                sentences.append(word2 + ' ' + word1)

    if args.pomdp:
        project_name = project_name + '_pomdp'

    seed = args.seed
    disable_cuda = False

    #random.seed(seed)
    #torch.manual_seed(seed)
    if torch.cuda.is_available() and not disable_cuda:
        # free_gpu = get_free_gpu()
        device = torch.device('cuda')  # + str(free_gpu))
        #torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = False
    else:
        device = torch.device('cpu')

    vocab_size = len(vocabulary)
    bow_parser = BagOfWords(vocabulary=vocabulary,
                            type_func=lambda x: torch.FloatTensor(x).to(device).unsqueeze(1))

    # word2vec_model_path = os.getcwd() + '/../ZorkGym/text_utils/GoogleNews-vectors-negative300.bin'
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    word2vec_model = glove_api.load('glove-wiki-gigaword-50')
    embedding_size = word2vec_model.vector_size
    word2vec_parser = Word2Vec(type_func=lambda x: torch.FloatTensor(x).to(device).unsqueeze(0),
                               word2vec_model=word2vec_model,
                               return_func=lambda x: word2vec_padding(x, 65, embedding_size))
    onehot_parser = OneHotParser(type_func=lambda x: torch.FloatTensor(x).to(device).unsqueeze(1),
                                 vocabulary=actions)

    """
        Experiments from here and below
    """
    for simulation in args.simulations:
        tau = 0.0
        train_params = {'seed': seed}
        if simulation == 'dqn_mlp':
            test_params = None
            agent = DQN(actions=sentences,
                        model_type='MLP',
                        parser=bow_parser,
                        input_length=vocab_size + 1,
                        input_width=1,
                        history_size=1,
                        device=device,
                        pomdp_mode=args.pomdp,
                        task=task,
                        sparse_reward=sparse_reward)

        elif simulation == 'dqn_cnn':
            test_params = None
            word2vec_parser.return_func = lambda x: word2vec_padding(x, 65, embedding_size)

            agent = DQN(actions=sentences,
                        model_type='CNN',
                        parser=word2vec_parser,
                        input_length=embedding_size,
                        input_width=65,
                        history_size=1,
                        device=device,
                        pomdp_mode=args.pomdp,
                        task=task,
                        sparse_reward=sparse_reward)

        elif simulation == 'drrn_mlp':
            test_params = None
            agent = DRRN(actions=sentences,
                         model_type='MLP',
                         parser=bow_parser,
                         input_length=vocab_size + 1,
                         input_width=1,
                         history_size=1,
                         device=device,
                         pomdp_mode=args.pomdp,
                         task=task,
                         sparse_reward=sparse_reward)
            tau = 0.2
        elif simulation == 'drrn_cnn':
            test_params = None
            agent = DRRN(actions=sentences,
                         model_type='CNN',
                         parser=word2vec_parser,
                         input_length=embedding_size,
                         input_width=65,
                         history_size=1,
                         device=device,
                         pomdp_mode=args.pomdp,
                         task=task,
                         sparse_reward=sparse_reward)
            tau = 0.2
        elif simulation == 'dddpg_mlp':
            word2vec_parser.return_func = lambda x: word2vec_sum(x, embedding_size)

            action_vocab_list = []
            for action in sentences:
                tokens = tokenizer(action)
                for token in tokens:
                    if token not in action_vocab_list:
                        action_vocab_list.append(token)
            action_vocabulary = {}
            embedding_size = len(action_vocab_list)
            for idx, action in enumerate(action_vocab_list):
                action_vocabulary[action] = np.zeros(embedding_size)
                action_vocabulary[action][idx] = 1.0
            word2vec_parser.word2vec_model = action_vocabulary

            train_params['number_of_neighbors'] = args.nn

            agent = DDDPG(actions=sentences,
                          state_parser=bow_parser,
                          action_parser=word2vec_parser,
                          embedding_size=embedding_size,
                          input_length=vocab_size + 1,
                          input_width=1,
                          history_size=1,
                          loss_weighting=loss_weighting,
                          model_type='MLP',
                          device=device,
                          pomdp_mode=args.pomdp,
                          task=task,
                          sparse_reward=sparse_reward)

        elif simulation == 'dddpg_cnn':
            action_word2vec_parser = Word2Vec(type_func=lambda x: torch.FloatTensor(x).to(device).unsqueeze(0),
                                              word2vec_model=word2vec_model,
                                              return_func=lambda x: word2vec_padding(x, 65, embedding_size))
            action_word2vec_parser.return_func = lambda x: word2vec_sum(x, embedding_size)

            train_params['number_of_neighbors'] = args.nn

            agent = DDDPG(actions=sentences,
                          state_parser=word2vec_parser,
                          action_parser=action_word2vec_parser,
                          embedding_size=embedding_size,
                          input_length=embedding_size,
                          input_width=65,
                          history_size=1,
                          loss_weighting=loss_weighting,
                          model_type='CNN',
                          device=device,
                          pomdp_mode=args.pomdp,
                          task=task,
                          sparse_reward=sparse_reward)

        elif simulation == 'ompddpg_mlp':
            words = set()
            for action in sentences:
                for word in tokenizer(action):
                    words.add(word)
            action_vocabulary = {}
            if args.action_w2v:
                for word in words:
                    action_vocabulary[word] = word2vec_model[word]
                action_vocabulary[''] = [0 for _ in range(len(action_vocabulary['open']))]
            else:
                words.add('')
                for idx, word in enumerate(words):
                    action_vocabulary[word] = np.zeros(len(words))
                    action_vocabulary[word][idx] = 1.0

            embedding_size = len(action_vocabulary['open'])

            train_params['number_of_neighbors'] = args.nn

            agent = OMPDDPG(actions=action_vocabulary,
                            state_parser=bow_parser,
                            embedding_size=embedding_size,
                            input_length=vocab_size + 1,
                            input_width=1,
                            history_size=1,
                            model_type='MLP',
                            device=device,
                            pomdp_mode=args.pomdp,
                            loss_weighting=loss_weighting,
                            linear=args.linear,
                            improved_omp=args.improved_omp,
                            model_path=args.model_path,
                            task=task,
                            sparse_reward=sparse_reward)

        elif simulation == 'ompddpg_cnn':
            words = set()
            for action in sentences:
                for word in tokenizer(action):
                    words.add(word)
            action_vocabulary = {}
            if args.action_w2v:
                for word in words:
                    action_vocabulary[word] = word2vec_model[word]
                action_vocabulary[''] = [0 for _ in range(len(action_vocabulary['open']))]
            else:
                words.add('')
                for idx, word in enumerate(words):
                    action_vocabulary[word] = np.zeros(len(words))
                    action_vocabulary[word][idx] = 1.0

            embedding_size = len(action_vocabulary['open'])

            train_params['number_of_neighbors'] = args.nn

            agent = OMPDDPG(actions=action_vocabulary,
                            state_parser=word2vec_parser,
                            embedding_size=embedding_size,
                            input_length=embedding_size,
                            input_width=65,
                            history_size=1,
                            model_type='CNN',
                            device=device,
                            pomdp_mode=args.pomdp,
                            loss_weighting=loss_weighting,
                            linear=args.linear,
                            improved_omp=args.improved_omp,
                            model_path=args.model_path,
                            task=task,
                            sparse_reward=sparse_reward)

        else:
            raise NotImplementedError

        model_name = simulation

        if not ('dqn' in simulation or 'drrn' in simulation):
            model_name += '/neighbors=' + str(args.nn)
        if args.action_w2v:
            model_name += '/w2v'
        if args.linear:
            model_name += '/linear'

        agent.learn(total_timesteps=time_steps,
                    buffer_size=buffer_size,
                    visualize=True,
                    vis_name=project_name + '/' + model_name + '/' + str(seed),
                    optimize_memory=optimize_memory,
                    tau=tau,
                    learn_start_steps=0, #20000,
                    train_params=train_params,
                    test_params=test_params,
                    eps_start=eps_start,
                    test_interval=args.test_interval,
                    game_seed=game_seed)


def word2vec_padding(list_of_embeddings, length, embedding_length):
    zero_vec = np.zeros(embedding_length)
    for _ in range(length - len(list_of_embeddings)):
        list_of_embeddings.append(zero_vec)
    return list_of_embeddings[:length]


def word2vec_sum(list_of_embeddings, embedding_length):
    ret_value = np.zeros(embedding_length)
    for embedding in list_of_embeddings:
        ret_value += embedding
    return ret_value


class OneHotParser(TextParser):
    def __init__(self, vocabulary, type_func):
        """

        :param vocabulary: List of strings representing the vocabulary.
        :param type_func: Function which converts the output to the desired type, e.g. np.array.
        """
        self.vocab = vocabulary
        self.vocab_size = len(self.vocab)
        TextParser.__init__(self, type_func)

    def __call__(self, x):
        one_hot = np.zeros((len(x), self.vocab_size))  # +1 for out of vocabulary tokens.
        for idx, token_list in enumerate(x):
            sentence = ' '.join(token_list)
            vocab_idx = self.vocab.index(sentence)
            one_hot[idx, vocab_idx] = 1

        return self.convert_type(one_hot)


if __name__ == '__main__':
    main()
