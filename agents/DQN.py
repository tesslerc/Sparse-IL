import pickle
import os
import torch
from agents.BaseAgent import BaseAgent
from networks.mlp import MlpBow
from networks.cnn import TextCNN
from networks.init_weights import init_weights


class DQN(BaseAgent):
    def __init__(self, actions, task='egg', name=None, parser=lambda x: x, input_length=None,
                 input_width=None, history_size=1, model_type='MLP', device=torch.device('cpu'), pomdp_mode=True,
                 sparse_reward=True):
        super(DQN, self).__init__(actions=actions, task=task, name=name, parser=parser,
                                  input_length=input_length, input_width=input_width, history_size=history_size,
                                  model_type=model_type, device=device, pomdp_mode=pomdp_mode,
                                  sparse_reward=sparse_reward)

        if self.model_type == 'CNN':
            self.network = TextCNN(embedding_size=self.input_length, output_size=self.action_size,
                                   history_size=self.history_size, input_width=self.input_width).to(self.device)
        elif self.model_type == 'MLP':
            self.network = MlpBow(dict_size=self.input_length, output_size=self.action_size, hidden_layers=[100, 100],
                                  history_size=self.history_size).to(self.device)
        else:
            raise NotImplementedError

        self.network.apply(init_weights)

    def _get_action(self, state, **kwargs):
        additional_prints = False if 'additional_prints' not in kwargs else kwargs['additional_prints']
        tau = kwargs['tau']
        eps = kwargs['eps']

        q_vals = self.network(state)
        if additional_prints:
            print('DQN')
            print(q_vals)
        if tau == 0:
            return self._select_eps_greedy_action(eps, q_vals.max(1, keepdim=True)[1])
        else:
            raise NotImplementedError
            # dist = torch.distributions.Categorical(logits=(q_vals / tau))
            # return dist.sample()

    def _get_q_value(self, network, state, action):
        return network(state).gather(1, action)

    def save_model(self, project_name, results, **kwargs):
        save_name = 'dqn'

        time_step = results['rewards']['base'][-1][0]
        base_path = os.getcwd() + '/' + project_name + '/' + save_name + '/' + str(kwargs['seed']) + '/' + str(time_step)
        os.makedirs(base_path)
        model_path = base_path + '/model'

        torch.save(self.network.state_dict(), model_path)

        with open(base_path + '/results', 'wb') as f:
            pickle.dump(results, f)
