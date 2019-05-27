"""
He et. al (2016), http://www.aclweb.org/anthology/P16-1153
"""
import os
import pickle
import torch
from agents.BaseAgent import BaseAgent
from networks.mlp import MlpDRRN
from networks.cnn import TextCNN_DRRN
from networks.init_weights import init_weights


class DRRN(BaseAgent):
    def __init__(self, actions, task='egg', name=None, parser=lambda x: x, input_length=None,
                 input_width=1, history_size=1, model_type='MLP', device=torch.device('cpu'), pomdp_mode=True,
                 sparse_reward=True):
        super(DRRN, self).__init__(actions=actions, task=task, name=name, parser=parser,
                                   input_length=input_length, input_width=input_width, history_size=history_size,
                                   model_type=model_type, device=device, pomdp_mode=pomdp_mode,
                                   sparse_reward=sparse_reward)

        if self.model_type == 'CNN':
            self.network = TextCNN_DRRN(embedding_size=self.input_length,
                                        hidden_size=100,
                                        history_size=self.history_size,
                                        input_width=self.input_width,
                                        action_length=self.input_width).to(self.device)
        elif self.model_type == 'MLP':
            self.network = MlpDRRN(dict_size=self.input_length, action_size=self.input_length, hidden_layers=[100, 100],
                                   history_size=history_size).to(self.device)
        else:
            raise NotImplementedError

        self.network.apply(init_weights)

        self.action_embeddings = self._create_action_embeddings(self.parser)

    def _get_action(self, state, **kwargs):
        if 'additional_prints' in kwargs:
            additional_prints = kwargs['additional_prints']
        else:
            additional_prints = False
        tau = kwargs['tau']
        eps = kwargs['eps']

        tiled_state = self._tile(state, 0, self.action_size)

        tiled_actions = self.action_embeddings.repeat(state.shape[0], 1, 1)
        state_batches = torch.split(tiled_state, 32)
        action_batches = torch.split(tiled_actions, 32)

        q_vals = []
        if additional_prints:
            print('DRRN')
        for idx in range(len(state_batches)):
            q_vals.append(self.network(state_batches[idx], action_batches[idx]))
            if additional_prints:
                print(self.actions[idx])
                print(q_vals[-1])
        q_vals = torch.cat(q_vals).view(state.shape[0], self.action_size)
        if tau == 0:
            argmax = q_vals.max(1, keepdim=True)[1]
            if eps > 0:
                actions = []
                for batch_idx in range(state.shape[0]):
                    actions.append(self._select_eps_greedy_action(eps, argmax[batch_idx])[0])
                return torch.Tensor(actions, device=self.device), ''
            return argmax, ''
        dist = torch.distributions.Categorical(logits=(q_vals / tau))
        action = dist.sample()
        return action, self.actions[action]

    def _get_q_value(self, network, state, action):
        actions_tensor = []
        for act in action:
            actions_tensor.append(self.action_embeddings[act])
        actions_tensor = torch.stack(actions_tensor)
        return network(state, actions_tensor).view(actions_tensor.shape[0], 1)

    def save_model(self, project_name, results, **kwargs):
        pass
        # save_name = 'drrn_' + str(self.input_length)
        #
        # time_step = results[list(results.keys())[0]][-1][0]
        # base_path = os.getcwd() + '/' + project_name + '/' + save_name + '/' + str(kwargs['seed']) + '/' + str(time_step)
        # os.makedirs(base_path)
        # actor_path = base_path + '/actor'
        # critic_path = base_path + '/critic'
        #
        # torch.save(self.network[0].state_dict(), actor_path)
        # torch.save(self.network[1].state_dict(), critic_path)
        #
        # with open(base_path + '/results', 'wb') as f:
        #     pickle.dump(results, f)
