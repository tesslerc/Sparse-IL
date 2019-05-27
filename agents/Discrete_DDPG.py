import os
import pickle
import torch
from torch import optim
from torch.nn import functional as F
from torch import nn
from agents.BaseAgent import BaseAgent
from networks.mlp import MlpBow, MlpDRRN
from networks.cnn import TextCNN, TextCNN_DRRN
from networks.init_weights import init_weights


class DDDPG(BaseAgent):
    def __init__(self, actions, task='egg', name=None, state_parser=lambda x: x,
                 action_parser=lambda x: x, input_length=None, input_width=None, history_size=1, model_type='MLP',
                 embedding_size=None, device=torch.device('cpu'), pomdp_mode=True, loss_weighting=0.0,
                 sparse_reward=True):
        super(DDDPG, self).__init__(actions=actions, task=task, name=name, parser=state_parser,
                                    input_length=input_length, input_width=input_width, history_size=history_size,
                                    model_type=model_type, device=device, pomdp_mode=pomdp_mode,
                                    sparse_reward=sparse_reward)

        assert embedding_size is not None

        self.loss_weighting = loss_weighting
        self.embedding_size = embedding_size
        self.action_parser = action_parser
        self.action_embeddings = self._create_action_embeddings(self.action_parser)

        self.network = self._create_network()

    def _get_action(self, state, **kwargs):
        eps = kwargs['eps']
        tau = kwargs['tau']
        additional_prints = False if 'additional_prints' not in kwargs else kwargs['additional_prints']
        number_of_neighbors = kwargs['number_of_neighbors']

        if number_of_neighbors == -1:
            number_of_neighbors = len(self.actions)

        action_embedding = self.network[0](state)
        k_indices = self._find_k_nearest_neighbors(action_embedding, number_of_neighbors)

        tiled_state = self._tile(state, 0, number_of_neighbors)

        flattened_k_indices = k_indices.view(-1)
        tiled_actions = self.action_embeddings[flattened_k_indices, :]

        q_vals = self.network[1](tiled_state, tiled_actions).view(state.shape[0], number_of_neighbors)

        if tau == 0:
            index = q_vals.max(1, keepdim=True)[1]
        else:
            dist = torch.distributions.Categorical(logits=(q_vals / tau))
            index = dist.sample()

        if additional_prints:
            print(action_embedding)
            print(self.action_embeddings[k_indices.gather(1, index)])

        return self._select_eps_greedy_action(eps, k_indices.gather(1, index))

    def _get_q_value(self, network, state, action):
        actions_tensor = self.action_embeddings[action]
        return network[1](state, actions_tensor).view(actions_tensor.shape[0], 1)

    def _find_k_nearest_neighbors(self, embedding, number_of_neighbors):
        distances = ((embedding.unsqueeze(1) - self.action_embeddings.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
        _, indices = torch.sort(distances)

        return indices[:, :number_of_neighbors].contiguous()

    def _train(self, target_network, gamma, replay_buffer, batch_size, optimizer, **kwargs):
        if len(replay_buffer) < batch_size:
            return None, None

        number_of_neighbors = kwargs['number_of_neighbors']

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
        obs_batch = obs_batch.to(self.device).detach()
        act_batch = torch.tensor(act_batch, dtype=torch.int64, device=self.device).unsqueeze(-1)
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
        current_Q = self._get_q_value(self.network, obs_batch, act_batch)

        optimizer[0].zero_grad()
        if self.loss_weighting == 0:
            actor_loss = -torch.mean(self.network[1](obs_batch, self.network[0](obs_batch)))
        elif self.loss_weighting == 1:
            best_action_idx = self._get_action(obs_batch, tau=0.0, eps=0.0, number_of_neighbors=number_of_neighbors)[0]
            best_action = self.action_embeddings[best_action_idx]
            actor_loss = F.mse_loss(self.network[0](obs_batch), best_action.view(batch_size, -1).detach())
        else:
            actor_loss = -torch.mean(self.network[1](obs_batch, self.network[0](obs_batch))) * (1 - self.loss_weighting)
            best_action_idx = self._get_action(obs_batch, tau=0.0, eps=0.0, number_of_neighbors=number_of_neighbors)[0]
            best_action = self.action_embeddings[best_action_idx]
            actor_loss += self.loss_weighting * F.mse_loss(self.network[0](obs_batch),
                                                           best_action.view(batch_size, -1).detach())
        actor_loss.backward()
        optimizer[0].step()

        optimizer[1].zero_grad()
        critic_loss = F.smooth_l1_loss(current_Q, target_Q.detach())
        critic_loss.backward()
        optimizer[1].step()

        td_error = (current_Q - target_Q).mean().item()

        return actor_loss.item() + critic_loss.item(), td_error

    def _create_network(self):
        if self.model_type == 'MLP':
            network = [MlpBow(self.input_length, self.embedding_size, [100, 100], self.history_size),
                       MlpDRRN(self.input_length, self.embedding_size, [100, 100], self.history_size)]
        elif self.model_type == 'CNN':
            network = [TextCNN(self.input_length,
                               self.embedding_size,
                               self.history_size,
                               self.input_width),
                       TextCNN_DRRN(embedding_size=self.embedding_size,
                                    hidden_size=100,
                                    history_size=self.history_size,
                                    input_width=self.input_width).to(self.device)]
        else:
            raise NotImplementedError

        # if torch.cuda.device_count() > 1:
        #     network = [nn.DataParallel(network[0]),
        #                nn.DataParallel(network[1])]

        network[0].to(self.device)
        network[1].to(self.device)

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
        save_name = 'dddpg_' + str(self.embedding_size) + '_' + str(kwargs['number_of_neighbors'])
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
