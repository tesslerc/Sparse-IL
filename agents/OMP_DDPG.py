import torch
from torch import optim
from torch.nn import functional as F
import os
import numpy as np
import pickle
import random
import torch.multiprocessing as mp
from itertools import repeat
import time

from agents.BaseAgent import BaseAgent
from networks.mlp import MlpBow, MlpDRRN
from networks.cnn import TextCNN, TextCNN_DRRN
from networks.init_weights import init_weights
from ZorkGym.text_utils.text_parser import tokenizer


def batch_beam_search(omp_steps, number_of_neighbors, sentence_length, word_embeddings, hashmap, residual):
    if hashmap is not None and str(residual) in hashmap:
        return hashmap[str(residual)]

    beam = [[list(), 0, 0]]
    for omp_step in range(omp_steps):
        all_candidates = list()

        # expand each current candidate
        for i in range(len(beam)):
            seq, seq_embedding, remainder = beam[i]

            for j in range(sentence_length):
                embedding = seq_embedding + word_embeddings[j]
                candidate = [sorted(seq + [j]), embedding, np.linalg.norm(embedding - residual)]
                all_candidates.append(tuple(candidate))

        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[2])
        idx = 1

        # only keep unique
        while idx < len(ordered):
            if ordered[idx][0] == ordered[idx - 1][0]:
                ordered.pop(idx)
            else:
                idx += 1

        # select k best
        beam = ordered[:min(number_of_neighbors, len(ordered))]
    return beam


class OMPDDPG(BaseAgent):
    def __init__(self, actions, task='egg', name=None, state_parser=lambda x: x, input_length=None, input_width=None,
                 history_size=1, model_type='MLP', embedding_size=None, device=torch.device('cpu'), pomdp_mode=True,
                 loss_weighting=0, linear=False, improved_omp=False, model_path=None, sparse_reward=True):
        super(OMPDDPG, self).__init__(actions=actions, task=task, name=name, parser=state_parser,
                                      input_length=input_length, input_width=input_width, history_size=history_size,
                                      model_type=model_type, device=device, pomdp_mode=pomdp_mode,
                                      sparse_reward=sparse_reward)

        assert embedding_size is not None

        """
        actions should be like this:
        {   
            'verbs': {'go': 0.11423, 'take': 342342},
            'nouns': {'egg': 1111, 'tree': 6544}
        }
        """

        self.linear = linear
        self.embedding_size = embedding_size
        self.improved_omp = improved_omp

        self.sentence_length = 4

        self.network = self._create_network()

        if model_path is not None:
            self.network[0].load_state_dict(torch.load(model_path))
            self.sparse_drrn = True
            self.hash = {}
        else:
            self.sparse_drrn = False
            self.hash = None

        self.verbs = ['raise', 'turn', 'bring', 'grab', 'run', 'close', 'go', 'heave', 'drop', 'ring', 'kill', 'climb',
                      'push', 'light', 'lower', 'wave', 'enter', 'read', 'hit', 'put', 'walk', 'press', 'get', 'throw',
                      'examine', 'take', 'touch', 'douse', 'move', 'dig', 'open', 'inflate']

        # all_actions and synonyms are merely a remark. not in use in the code.
        all_actions = ['button', 'tie', 'turn', 'land', 'ring', 'coffin', 'wait', 'match', 'bolt', 'douse', 'boat',
                       'light', 'pile', 'launch', 'and', 'basket', 'window', 'switch', 'wrench', 'map', 'knife', 'wave',
                       'trapdoor', 'dig', 'lamp', 'wind', 'shovel', 'lid', 'emerald', 'scarab', 'case', 'figurine',
                       'from', 'railing', 'odysseus', 'bag', 'enter', 'drop', 'troll', 'label', 'east', 'coal', 'red',
                       'pot', 'get', 'trident', 'buoy', 'bracelet', 'move', 'mirror', 'rope', 'trunk', 'with',
                       'sceptre', 'all', 'torch', 'canary', 'north', 'rusty', 'read', 'yellow', 'examine', 'take',
                       'candles', 'rug', 'pray', 'echo', 'prayer', 'south', 'press', 'screwdriver', 'egg', 'chalice',
                       'page', 'bauble', 'inflate', 'painting', 'matchbook', 'to', 'put', 'out', 'look', 'in', 'touch',
                       'southeast', 'southwest', 'sand', 'up', 'kill', 'bell', 'sword', 'raise', 'throw', 'close',
                       'down', 'thief', 'skull', 'bar', 'open', 'diamond', 'west', 'garlic', 'northeast', 'nasty', 'of',
                       'northwest', 'machine', 'jewels', 'book', 'sack', 'pump', 'lower']
        synonyms = ['walk', 'run', 'take', 'place']

        long_sentences = [
            'kill troll with sword',
            'tie rope to railing',
            'put coffin in case',
            'light candles with match',
            'put skull in case',
            'put bar in case',
            'get knife and bag',
            'put bag in case',
            'drop rusty knife',
            'press red button',
            'press yellow button',
            'turn bolt with wrench',
            'put trunk in case',
            'put trident in case',
            'throw sceptre in boat',
            'get out of boat',
            'put sceptre in case',
            'put pot in case',
            'put emerald in case',
            'put scarab in case',
            'get rusty knife',
            'get nasty knife',
            'kill thief with knife',
            'drop rusty knife',
            'drop nasty knife',
            'put painting in case',
            'put chalice in case',
            'put egg in case',
            'put canary in case',
            'put bauble in case',
            'put jewels in case',
            'put torch in basket',
            'put screwdriver in basket',
            'put coal in basket',
            'get all from basket',
            'put coal in machine',
            'turn switch with screwdriver',
            'put diamond in basket',
            'put torch in basket',
            'put screwdriver in basket',
            'put diamond in case',
            'put torch in case',
            'put bracelet in case',
            'put figurine in case',
            'put trunk in case',
            'take torch and lamp',
            'get rope and knife'
        ]

        self.long_sentence_sets = []
        for sentence in long_sentences:
            set_of_tokens = set(tokenizer(sentence))

            exists = False
            for item in self.long_sentence_sets:
                if set_of_tokens == item[0]:
                    exists = True
                    break
            if not exists:
                self.long_sentence_sets.append((set_of_tokens, sentence))

        self.words = []
        self.word_embeddings_np = []
        for word in self.actions:
            self.words.append(word)
            self.word_embeddings_np.append(self.actions[word])

        self.word_embeddings = torch.Tensor(self.word_embeddings_np)
        self.word_embeddings_np = np.array(self.word_embeddings_np)
        self.word_embeddings_device = torch.Tensor(self.word_embeddings).to(self.device)

        # self.sentences = []
        # self.sentence_embeddings = []
        # for idx_1, word in enumerate(self.words):
        #     for idx_2, word_2 in enumerate(self.words[idx_1+1:]):
        #         self.sentences.append([self.words.index(word), self.words.index(word_2)])
        #         self.sentence_embeddings.append(np.array(self.actions[word]) + np.array(self.actions[word_2]))
        #
        # self.sentence_embeddings = torch.Tensor(self.sentence_embeddings)
        # self.sentence_embeddings_device = torch.Tensor(self.sentence_embeddings).to(self.device)

        self.number_of_words = len(self.words)
        self.loss_weighting = loss_weighting

        self.training_steps = 0

    def _select_eps_greedy_action(self, eps, action, candidates=None):
        # noise = torch.FloatTensor(self.ounoise.noise(eps * 10))
        # action += noise

        if candidates is None:
            if random.random() < eps:
                idx_1 = random.randint(0, len(self.actions) - 1)
                idx_2 = idx_1
                while idx_2 == idx_1:
                    idx_2 = random.randint(0, len(self.actions) - 1)

                action = [idx_1, idx_2]
        elif random.random() < eps:
            action = candidates[random.randint(0, len(candidates) - 1)]

        words = []
        embedding = torch.zeros(self.sentence_length, self.embedding_size)
        for idx, word_idx in enumerate(action):
            embedding[idx] += self.word_embeddings[word_idx]
            if len(self.words[word_idx]) > 0:
                words.append(self.words[word_idx])

        text_action = ''

        is_long_sentence = False
        set_of_words = set(words)
        for sentence in self.long_sentence_sets:
            if set_of_words == sentence[0]:
                is_long_sentence = True
                text_action = sentence[1]

        if not is_long_sentence:
            for word in words:
                if word in self.verbs:
                    text_action = word + ' ' + text_action
                else:
                    text_action += ' ' + word

        return embedding.unsqueeze(0), text_action

    def _get_action_consider_all(self, state, eps, additional_prints):
        raise NotImplementedError
        # Doesn't consider actions of more than 2 words

        batch_size = state.shape[0]

        state_features = self.network[1].get_state_stream(state)
        tiled_state_features = self._tile(state_features, 0, len(self.sentences))

        action_features = self.network[1].get_action_stream(self.sentence_embeddings_device)
        tiled_action_features = action_features.repeat(batch_size, 1)

        q_vals = self.network[1].get_q(tiled_state_features, tiled_action_features).view(batch_size, -1)

        sentence_indices = q_vals.max(1, keepdim=True)[1]

        if additional_prints:
            print('Actions')
            for i in range(len(self.sentences)):
                act, text = self._select_eps_greedy_action(0, self.sentences[i])
                print(act)
                print(q_vals[0, i])
                print(text)

        if state.shape[0] == 1:
            sentence_idx = sentence_indices[0]
            return self._select_eps_greedy_action(eps, self.sentences[sentence_idx], None)

        batch_action = []
        for batch_idx in range(batch_size):
            batch_action.append(self.sentence_embeddings_device[sentence_indices[batch_idx]])
        return torch.cat(batch_action), ''

    def _standard_omp(self, action, return_candidates=False):
        batch_size = action.shape[0]

        assert batch_size == 1

        residual = action.view(1, self.embedding_size).cpu()

        number_of_words = self.word_embeddings.shape[0]

        words = list()
        for omp_step in range(self.sentence_length):
            correlations = (residual * self.word_embeddings.view(number_of_words, self.embedding_size)).sum(
                -1).view(number_of_words)

            best_word_idx = correlations.argmax().item()
            if self.word_embeddings[best_word_idx].sum() == 0:
                break
            else:
                support = (residual.view(self.embedding_size) * self.word_embeddings[best_word_idx].view(
                    self.embedding_size)).sum() / (self.word_embeddings[best_word_idx].view(self.embedding_size) *
                                                   self.word_embeddings[best_word_idx].view(self.embedding_size)).sum()

            residual -= self.word_embeddings[best_word_idx] * support
            words.append((best_word_idx, support))

        words_indices = []
        for word in words:
            for _ in range(int(round(word[1].item()))):
                words_indices.append(word[0])
        candidate = words_indices
        if return_candidates:
            return [candidate], [candidate]

        return self._select_eps_greedy_action(0, candidate, None)

    def _fista(self, action):
        raise NotImplementedError
        batch_size = action.shape[0]

        assert batch_size == 1

        residual = action.view(1, self.embedding_size).cpu()

    def _get_action_beam_search(self, action, state, eps, additional_prints, number_of_neighbors, use_critic, return_candidates=False):
        batch_size = action.shape[0]
        action = action.cpu()

        residual = [action[i].numpy() for i in range(batch_size)]
        # if self.improved_omp:
        #     C = self.network[1].get_state_stream(state).view(batch_size, self.embedding_size).cpu()
        #     residual = (residual + (self.word_embeddings.inverse().t().unsqueeze(0) * C.unsqueeze(-1)).sum(-1) / 2)

        omp_combined = torch.zeros((batch_size, number_of_neighbors, self.sentence_length, self.embedding_size))
        remaining_residual = []
        indices = []
        for batch_idx in range(batch_size):
            if self.hash is not None and str(residual[batch_idx]) in self.hash:
                for neighbor_idx in range(number_of_neighbors):
                    tup = []
                    for idx in range(self.sentence_length):
                        tup.append(self.hash(str(residual[batch_idx]))[neighbor_idx][0][idx])
                    omp_combined[batch_idx, neighbor_idx] = self._select_eps_greedy_action(0, tup)[0]
            else:
                remaining_residual.append(residual[batch_idx])
                indices.append(batch_idx)

        with mp.Pool() as pool:
            k_candidates = pool.starmap(batch_beam_search,
                                        zip(repeat(self.sentence_length), repeat(number_of_neighbors),
                                            repeat(self.number_of_words), repeat(self.word_embeddings_np),
                                            repeat(self.hash), remaining_residual)
                                        )
        if self.hash is not None:
            for idx, val in enumerate(remaining_residual):
                self.hash[str(remaining_residual)] = k_candidates[idx]

        for candidate_idx, original_idx in enumerate(indices):
            for neighbor_idx in range(number_of_neighbors):
                tup = []
                for idx in range(self.sentence_length):
                    tup.append(k_candidates[candidate_idx][neighbor_idx][0][idx])
                omp_combined[original_idx, neighbor_idx] = self._select_eps_greedy_action(0, tup)[0]

        if use_critic:
            obs_features = self.network[1].get_state_stream(state)
            obs_features = self._tile(obs_features, 0, number_of_neighbors)
            act_features = self.network[1].get_action_stream(omp_combined.view(-1, self.sentence_length, self.embedding_size).to(self.device))

            q_vals = self._get_q_from_features(self.network, obs_features, act_features).view(batch_size,
                                                                                              number_of_neighbors)

            index = q_vals.max(1, keepdim=True)[1].cpu()

            if additional_prints:
                print('OMP nearest neighbors')
                print(action)

                for i in range(number_of_neighbors):
                    tup = []
                    for idx in range(self.sentence_length):
                        tup.append(k_candidates[0][i][0][idx])
                    act, text = self._select_eps_greedy_action(0, tup)
                    print(act)
                    print(q_vals[0, i])
                    print(text)

        else:
            index = [torch.tensor([0], dtype=torch.int8) for _ in range(batch_size)]

        if batch_size == 1:
            tup = []
            for idx in range(self.sentence_length):
                tup.append(k_candidates[0][index[0].item()][0][idx])
            candidates = []
            for candidate in range(number_of_neighbors):
                candidates.append([])
                for idx in range(self.sentence_length):
                    candidates[candidate].append(k_candidates[0][candidate][0][idx])
            if return_candidates:
                return tup, candidates
            return self._select_eps_greedy_action(eps, tup, candidates)

        ret = []
        for idx in range(batch_size):
            ret.append(omp_combined[idx, index[idx], :, :])
        return torch.cat(ret).to(self.device), ''

    def _get_action(self, state, **kwargs):
        eps = kwargs['eps']
        number_of_neighbors = kwargs['number_of_neighbors']
        additional_prints = False if 'additional_prints' not in kwargs else kwargs['additional_prints']
        use_critic = True if 'use_critic' not in kwargs else kwargs['use_critic']
        alg_type = 'integer_omp' if 'alg_type' not in kwargs else kwargs['alg_type']

        action = self.network[0](state)
        if alg_type == 'integer_omp':
            if number_of_neighbors == -1:
                return self._get_action_consider_all(state, eps, additional_prints)
            return self._get_action_beam_search(action, state, eps, additional_prints, number_of_neighbors, use_critic)
        elif alg_type == 'omp':
            return self._standard_omp(action)
        elif alg_type == 'bp':
            return self._fista(action)
        else:
            raise NotImplementedError

    def _get_q_value(self, network, state, action):
        batch_size = state.shape[0]
        action = action.view(batch_size, self.sentence_length, self.embedding_size).to(self.device)
        if self.linear:
            state_features = network[1](state)
            q_vals = self.network[1].get_q(state_features, action)
        else:
            q_vals = network[1](state, action)
        return q_vals.view(batch_size, 1)

    def _get_q_from_features(self, network, state_features, action_features):
        q_vals = network[1].get_q(state_features, action_features)
        return q_vals.view(state_features.shape[0], 1)

    def _train(self, target_network, gamma, replay_buffer, batch_size, optimizer, **kwargs):
        if len(replay_buffer) < batch_size:
            return None, None

        self.training_steps += 1

        number_of_neighbors = kwargs['number_of_neighbors']

        critic_loss_item = 0
        td_error = 0
        for _ in range(10):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            obs_batch = obs_batch.to(self.device).detach()
            act_batch = torch.stack(act_batch).to(self.device).detach()
            rew_batch = torch.tensor(rew_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
            next_obs_batch = next_obs_batch.to(self.device).detach()
            not_done_mask = 1 - torch.tensor(done_mask, dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                next_state_actions = self._get_action(next_obs_batch,
                                                      tau=0.0,
                                                      eps=0.0,
                                                      number_of_neighbors=number_of_neighbors)[0]
                next_Q = self._get_q_value(target_network, next_obs_batch, next_state_actions) * not_done_mask
                target_Q = rew_batch + (gamma * next_Q)

            obs_features = self.network[1].get_state_stream(obs_batch)
            act_features = self.network[1].get_action_stream(act_batch)

            current_Q = self._get_q_from_features(self.network, obs_features, act_features)

            optimizer[1].zero_grad()
            critic_loss = F.smooth_l1_loss(current_Q, target_Q.detach())
            critic_loss.backward()
            optimizer[1].step()
            critic_loss_item += critic_loss.item() / 10
            td_error += (current_Q - target_Q).mean().item() / 10

        if not self.sparse_drrn:
            optimizer[0].zero_grad()

            predicted_actions = self.network[0](obs_batch)
            best_actions = self._get_action(obs_batch, tau=0.0, eps=0.0, number_of_neighbors=number_of_neighbors)[0]

            if self.loss_weighting == 0:
                actor_loss = -torch.mean(self._get_q_from_features(self.network, obs_features.detach(),
                                                                   self.network[1].get_action_stream(
                                                                       self.network[0](obs_batch))))
            elif self.loss_weighting == 1:
                actor_loss = self.loss_weighting * F.mse_loss(predicted_actions, best_actions.view(batch_size, -1).detach())
            else:
                actor_loss = self.loss_weighting * F.mse_loss(predicted_actions, best_actions.view(batch_size, -1).detach()) \
                             - (1 - self.loss_weighting) * \
                             torch.mean(self._get_q_from_features(self.network,
                                                                  obs_features.detach(),
                                                                  self.network[1].get_action_stream(
                                                                      self.network[0](obs_batch))))
            actor_loss.backward()
            optimizer[0].step()
            actor_loss_item = actor_loss.item()
        else:
            actor_loss_item = 0

        return actor_loss_item + critic_loss_item, td_error

    def _create_network(self):
        network = []
        if self.model_type == 'MLP':
            network.append(MlpBow(self.input_length,
                                  self.embedding_size,
                                  [100, 100],
                                  self.history_size).to(self.device))
            if self.linear:
                network.append(MlpBow(self.input_length,
                                      self.embedding_size,
                                      [100, 100],
                                      self.history_size).to(self.device))
            else:
                network.append(MlpDRRN(self.input_length,
                                       self.embedding_size,
                                       [100, 100],
                                       self.history_size).to(self.device))
        elif self.model_type == 'CNN':
            network.append(TextCNN(self.input_length,
                                   self.embedding_size,
                                   self.history_size,
                                   self.input_width).to(self.device))
            if self.linear:
                network.append(TextCNN(embedding_size=self.input_length,
                                       output_size=self.embedding_size,
                                       history_size=self.history_size,
                                       input_width=self.input_width).to(self.device))
            else:
                network.append(TextCNN_DRRN(embedding_size=self.embedding_size,
                                            hidden_size=100,
                                            history_size=self.history_size,
                                            input_width=self.input_width,
                                            action_length=self.sentence_length).to(self.device))
        else:
            raise NotImplementedError

        for net in network:
            net.apply(init_weights)
        return network

    @staticmethod
    def _copy_network(network, target_network):
        for idx in range(len(network)):
            target_network[idx].load_state_dict(network[idx].state_dict())
        return target_network

    def _create_optimizer(self, lr):
        return [optim.RMSprop(self.network[0].parameters(),
                              lr=lr,
                              eps=0.00001,
                              momentum=0.0,
                              alpha=0.95,
                              centered=True),
                optim.RMSprop(self.network[1].parameters(),
                              lr=lr,
                              eps=0.00001,
                              momentum=0.0,
                              alpha=0.95,
                              centered=True)]

    def save_model(self, project_name, results, **kwargs):
        save_name = 'omp_ddpg_' + str(self.embedding_size) +\
                    ('_linear' if self.linear else '') + '_' +\
                    str(kwargs['number_of_neighbors'])

        if self.loss_weighting > 0:
            save_name += '_mse_loss'

        time_step = results[list(results.keys())[0]][-1][0]
        base_path = os.getcwd() + '/' + project_name + '/' + save_name + '/' + str(kwargs['seed']) + '/' + str(time_step)
        os.makedirs(base_path)
        actor_path = base_path + '/actor'
        critic_path = base_path + '/critic'

        torch.save(self.network[0].state_dict(), actor_path)
        torch.save(self.network[1].state_dict(), critic_path)

        with open(base_path + '/results', 'wb') as f:
            pickle.dump(results, f)
