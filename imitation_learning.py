import pickle
import random
import torch
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
import numpy as np
import gensim.downloader as glove_api
import os

import argparse

from ZorkGym.text_utils.text_parser import Word2Vec, tokenizer
from agents.OMP_DDPG import OMPDDPG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='troll')
    parser.add_argument('--iter', default=0, type=int)
    parser.add_argument('--snr', default=0, type=float)
    parser.add_argument('--training_steps', default=200000, type=int)
    parser.add_argument('--history_size', default=1, type=int)
    return parser.parse_args()

args = parse_args()

history_size = args.history_size

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

with open(os.getcwd() + '/data/zork_walkthrough_' + args.task + '.txt', 'rb') as f:
    data = pickle.load(f)

single_states = [word2vec_parser(state) for state in data['states']]
raw_actions = data['actions']
states = []
actions = []

noise = MultivariateNormal(torch.zeros(50), torch.eye(50))

dictionary = set()
for action in raw_actions:
    vect = 0
    for token in tokenizer(action):
        dictionary.add(token)
        vect += word2vec_model[token]

    sampled_noise = noise.sample().numpy()
    normalized_noise = args.snr * np.linalg.norm(vect) * sampled_noise / np.linalg.norm(sampled_noise)
    actions.append(torch.Tensor(vect + normalized_noise).to(device))

for idx in range(len(single_states)):
    full_state = torch.zeros((history_size, 2, 65, embedding_size), dtype=torch.float32).to(device)
    start_idx = 0
    for i in range(history_size):
        if idx - i >= 0:
            full_state[start_idx:i+1] = single_states[idx - i]
            start_idx = i+1
    states.append(full_state)

action_vocabulary = {}
for word in dictionary:
    action_vocabulary[word] = word2vec_model[word]
action_vocabulary[''] = [0 for _ in range(len(action_vocabulary['open']))]

embedding_size = len(action_vocabulary['open'])

agent = OMPDDPG(actions=action_vocabulary,
                state_parser=word2vec_parser,
                embedding_size=embedding_size,
                input_length=embedding_size,
                input_width=65,
                history_size=history_size,
                model_type='CNN',
                device=device,
                pomdp_mode=True,
                loss_weighting=1.0,
                linear=False,
                improved_omp=False)

optimizer = agent._create_optimizer(lr=0.000001)
batch_size = 128
save_interval = 1000

path = os.getcwd() + '/imitation_agent_' + args.task + '_' + str(args.snr) + '/' + str(args.iter) + '/'
os.makedirs(path)

for iteration in range(args.training_steps + 1):
    indices = np.random.randint(0, len(actions), batch_size)
    obs_batch = []
    action_batch = []

    for idx in indices:
        obs_batch.append(states[idx])
        action_batch.append(actions[idx])

    obs_batch = torch.stack(obs_batch).detach()
    action_batch = torch.stack(action_batch).detach()

    optimizer[0].zero_grad()
    predicted_actions = agent.network[0](obs_batch)
    loss = F.mse_loss(predicted_actions, action_batch.view(batch_size, -1))
    #loss = F.l1_loss(predicted_actions, action_batch.view(batch_size, -1))
    loss.backward()
    print('Iteration ' + str(iteration) + ': ' + str(loss.item()))
    optimizer[0].step()

    if iteration % save_interval == 0:
        sub_path = path + str(iteration) + '/'
        os.makedirs(sub_path)
        torch.save(agent.network[0].state_dict(), sub_path + 'actor')
        torch.save(agent.network[1].state_dict(), sub_path + 'critic')
