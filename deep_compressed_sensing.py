import pickle
import random
import torch
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torch import optim
import torch.nn as nn
import numpy as np
import gensim.downloader as glove_api
import os
import argparse

from networks.cnn import TextCNN
from ZorkGym.text_utils.text_parser import Word2Vec, tokenizer
# from agents.OMP_DDPG import OMPDDPG


class MlpBow(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_layers):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(embedding_size, hidden_layers[0]))
        for idx in range(len(hidden_layers) - 1):
            self.linears.append(nn.Linear(hidden_layers[idx], hidden_layers[idx + 1]))

        self.linears.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x, sigmoid=False):
        x_relu = x.view(x.size(0), -1)
        for idx in range(len(self.linears)):
            x = self.linears[idx](x_relu)
            x_relu = F.relu(x)
        if sigmoid:
            return F.sigmoid(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='troll')
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--training_steps', default=20000, type=int)
    parser.add_argument('--prob', default=0, type=float)
    parser.add_argument('--amb', default=0, type=float)
    parser.add_argument('--model', default='cs', choices=['cs', 'full'])
    return parser.parse_args()

ambiguities = {'go': ['move', 'walk', 'run'], 'get': ['take'], 'kill': ['hit', 'attack'], 'press': ['push'], 'put': ['place'], 'drop': ['toss']} 

args = parse_args()
model = args.model  # cs = embedding to BoW. full = state to BoW
task = args.task
amb = args.amb
if task == 'troll':
    history_size = 1
else:
    history_size = 12

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = False
else:
    device = torch.device('cpu')


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


word2vec_model = glove_api.load('glove-wiki-gigaword-50')
embedding_size = word2vec_model.vector_size
word2vec_parser = Word2Vec(type_func=lambda x: torch.FloatTensor(x).to(device).unsqueeze(0),
                           word2vec_model=word2vec_model,
                           return_func=lambda x: word2vec_padding(x, 65, embedding_size))

with open(os.getcwd() + '/data/zork_walkthrough_' + task + '.txt', 'rb') as f:
    data = pickle.load(f)

states = [word2vec_parser(state) for state in data['states']]
raw_actions = data['actions']
actions = []
bows = []

noise = MultivariateNormal(torch.zeros(50), torch.eye(50))

#dictionary = ['pray', 'yellow', 'trapdoor', 'open', 'bell', 'touch', 'pile', 'trunk', 'sack', 'inflate', 'southeast',
#              'of', 'move', 'match', 'figurine', 'railing', 'with', 'map', 'mirror', 'wind', 'examine', 'north', 'out',
#              'trident', 'turn', 'skull', 'throw', 'northwest', 'case', 'bag', 'red', 'press', 'jewels', 'east', 'pump',
#              'bolt', 'rusty', 'window', 'douse', 'boat', 'bracelet', 'matchbook', 'basket', 'book', 'coffin', 'bar',
#              'rug', 'lid', 'drop', 'nasty', 'wrench', 'light', 'sand', 'bauble', 'kill', 'tie', 'painting', 'sword',
#              'wave', 'in', 'south', 'northeast', 'ring', 'canary', 'lower', 'egg', 'all', 'to', 'candles', 'page',
#              'and', 'echo', 'emerald', 'tree', 'from', 'rope', 'troll', 'screwdriver', 'torch', 'enter', 'coal', 'go',
#              'look', 'shovel', 'knife', 'down', 'take', 'switch', 'prayer', 'launch', 'diamond', 'read', 'up', 'get',
#              'scarab', 'west', 'land', 'southwest', 'climb', 'thief', 'raise', 'wait', 'odysseus', 'button', 'sceptre',
#              'lamp', 'chalice', 'garlic', 'buoy', 'pot', 'label', 'put', 'dig', 'machine', 'close', 'walk', 'run', 'hit', 'attack']

dictionary = set()
for action in raw_actions:
    for token in tokenizer(action):
        dictionary.add(token)
for ambiguity in ambiguities:
    for token in ambiguities[ambiguity]:
        dictionary.add(token)
dictionary = list(dictionary)
dictionary.sort()
print(dictionary)

for action in raw_actions:
    vect = 0
    bow = torch.zeros(len(dictionary), device=device)
    for token in tokenizer(action):
        vect += word2vec_model[token]
        bow[dictionary.index(token)] += 1

    actions.append(torch.Tensor(vect).to(device))
    bows.append(bow)

embedding_size = word2vec_model.vector_size

if model == 'cs':
    network = MlpBow(embedding_size=embedding_size, output_size=len(dictionary), hidden_layers=[100, 100])
else:
    network = TextCNN(embedding_size=embedding_size, history_size=history_size, input_width=65,
                          output_size=len(dictionary))

network = network.to(device)
optimizer = optim.Adam(network.parameters(), lr=0.0001)
bce_criterion = nn.BCEWithLogitsLoss()

batch_size = 128
num_iters = 20000
save_interval = 1000

if model == 'cs':
    path = os.getcwd() + '/deep_cs_' + task + '_' + model + '/' + str(args.iter) + '/'
else:
    path = os.getcwd() + '/deep_cs_' + task + '_' + model + '_' + str(args.prob) + '_' + str(args.amb) + '/' + str(args.iter) + '/'
os.makedirs(path)

for iteration in range(num_iters + 1):
    indices = np.random.randint(0, len(actions), batch_size)
    obs_batch = []
    action_batch = []
    bow_batch = []

    for idx in indices:
        full_state = torch.zeros((history_size,
                                  2,
                                  65,
                                  embedding_size), dtype=torch.float32).to(device)
        for i in reversed(range(history_size)):
            if idx - i >= 0:
                full_state[history_size-1-i] = states[idx - i]
        obs_batch.append(full_state)

        if np.random.rand() < amb:
            bow = torch.zeros(len(dictionary), device=device)
            for token in tokenizer(raw_actions[idx]):
                if token in ambiguities:
                    token = random.choice(ambiguities[token])
                token_idx = dictionary.index(token)
                bow[token_idx] = 1

            action_batch.append(actions[idx])
            bow_batch.append(bow)
        elif np.random.rand() < args.prob:
            idx_1 = np.random.randint(len(dictionary)-1)
            idx_2 = np.random.randint(len(dictionary) - 1)
            while idx_2 == idx_1:
                idx_2 = np.random.randint(len(dictionary)-1)
            vec = word2vec_model[dictionary[idx_1]] + word2vec_model[dictionary[idx_2]]

            bow = torch.zeros(len(dictionary), device=device)
            bow[idx_1] = 1
            bow[idx_2] = 1

            action_batch.append(torch.Tensor(vec).to(device))
            bow_batch.append(bow)
        else:
            action_batch.append(actions[idx])
            bow_batch.append(bows[idx])

    obs_batch = torch.stack(obs_batch).detach()
    action_batch = torch.stack(action_batch).detach()
    bow_batch = torch.stack(bow_batch).detach()

    optimizer.zero_grad()
    if model == 'cs':
        predicted_bow = network(action_batch)
    else:
        predicted_bow = network(obs_batch)

    loss = bce_criterion(predicted_bow, bow_batch)
    loss.backward()

    optimizer.step()

    print('Iteration ' + str(iteration) + ': ' + str(loss.item()))

    if iteration % save_interval == 0:
        sub_path = path + str(iteration) + '/'
        os.makedirs(sub_path)
        torch.save(network.state_dict(), sub_path + 'network')
