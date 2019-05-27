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
    parser.add_argument('--prob', default=0, type=float)
    parser.add_argument('--snr', default=0, type=float)
    parser.add_argument('--training_steps', default=20000, type=int)
    parser.add_argument('--amb', default=0, type=float)
    return parser.parse_args()

args = parse_args()

ambiguities = {'go': ['move', 'walk', 'run'], 'get': ['take'], 'kill': ['hit', 'attack'], 'press': ['push'], 'put': ['place'], 'drop': ['toss']}

snr = args.snr
prob = args.prob
amb = args.amb
if args.task == 'troll':
    history_size = 1
else:
    history_size = 12
task = args.task

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

noise = MultivariateNormal(torch.zeros(50), torch.eye(50))

dictionary = set()
for action in raw_actions:
    vect = 0
    for token in tokenizer(action):
        dictionary.add(token)
        vect += word2vec_model[token]

    sampled_noise = noise.sample().numpy()
    normalized_noise = snr * np.linalg.norm(vect) * sampled_noise / np.linalg.norm(sampled_noise)
    actions.append(torch.Tensor(vect + normalized_noise).to(device))

action_vocabulary = {}
for word in dictionary:
    action_vocabulary[word] = word2vec_model[word]
action_vocabulary[''] = [0 for _ in range(len(action_vocabulary['open']))]
dictionary = list(dictionary)

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
num_iters = 20000
save_interval = 1000

path = os.getcwd() + '/imitation_agent_action_noise_' + task + '_' + str(prob) + '_' + str(snr) + '_' + str(amb) + '/' + str(args.iter) + '/'
os.makedirs(path)

for iteration in range(num_iters + 1):
    indices = np.random.randint(0, len(actions), batch_size)
    obs_batch = []
    action_batch = []

    for idx in indices:
        full_state = torch.zeros((history_size,
                                  2,
                                  65,
                                  embedding_size), dtype=torch.float32).to(device)
        for i in reversed(range(history_size)):
            if idx - i >= 0:
                full_state[history_size-1-i] = states[idx - i]
        obs_batch.append(full_state)

        has_amb = False
        if np.random.rand() < amb:
            for token in tokenizer(action):
                if token in ambiguities:
                    has_amb = True
            if has_amb:
                vec = 0
                for token in tokenizer(action):
                    if token in ambiguities:
                        token = random.choice(ambiguities[token])
                    vec += word2vec_model[token]
                action_batch.append(torch.Tensor(vec).to(device))
        if not has_amb:
            if np.random.rand() < prob:
                vec = 0
                vec += word2vec_model[dictionary[np.random.randint(len(dictionary)-1)]]
                vec += word2vec_model[dictionary[np.random.randint(len(dictionary)-1)]]
                action_batch.append(torch.Tensor(vec).to(device))
            else:
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
