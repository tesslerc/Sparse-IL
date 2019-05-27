import torch
from torch import optim
import torch.nn.functional as F

import random
import copy
import numpy as np

from baselines.common.schedules import LinearSchedule
from ZorkGym.gym.zork_gym import ZorkEnv
from ZorkGym.text_utils.text_parser import tokenizer

from utils.replay_buffer import ReplayBuffer, SuccessReplayBuffer
from utils.plots import viz, vis_plot

from abc import ABC, abstractmethod

import time


class BaseAgent(ABC):
    def __init__(self, actions, task='egg', name=None, parser=lambda x: x, input_length=None,
                 input_width=1, history_size=1, model_type='MLP', device=torch.device('cpu'), pomdp_mode=True,
                 sparse_reward=True):
        assert input_length is not None

        self.device = device

        self.name = name if name is not None else ''
        self.actions = actions
        self.action_size = len(actions)

        self.parser = parser
        self.input_length = input_length
        self.input_width = input_width
        self.history_size = history_size

        self.success_reward = 0
        self.network = None
        self.pomdp_mode = pomdp_mode
        self.sparse_reward = sparse_reward
        self.env = self._create_env(task, lambda x: x, lambda x: tokenizer(x), 'ZorkGym/gym/zork1.z5',
                                    self.success_reward, self.pomdp_mode, self.sparse_reward)
        self.model_type = model_type

    def _select_eps_greedy_action(self, eps, action):
        sample = random.random()

        if sample < eps:
            action = torch.LongTensor([random.randrange(self.action_size)])
        return action, self.actions[action[0]]

    def learn(self,
              eps_decay_steps=100000,
              eps_min=0.1,
              eps_start=1.0,
              total_timesteps=2500000,
              learn_start_steps=20000,
              buffer_size=1000,
              target_update_interval=2000,
              test_interval=5000,
              train_interval=16,
              batch_size=128,
              lr=0.001,
              gamma=0.99,
              tau=0.0,
              visualize=True,
              vis_name=None,
              optimize_memory=False,
              train_params=None,
              test_params=None,
              game_seed=52):

        if test_params is None:
            test_params = {'base': {}}
        if train_params is None:
            train_params = {}

        target_network = self._create_network()
        self._copy_network(self.network, target_network)

        optimizer = self._create_optimizer(lr=lr)

        exploration = LinearSchedule(schedule_timesteps=int(eps_decay_steps * total_timesteps - learn_start_steps),
                                     initial_p=eps_start, final_p=eps_min)

        if optimize_memory:
            def state_parser(x): return torch.cat([self._parse_state(elem) for elem in x])

            def state_to_memory(raw, _): return raw
        else:
            def state_parser(x): return torch.cat(x)

            def state_to_memory(_, parsed): return parsed
        # replay_buffer = ReplayBuffer(capacity=buffer_size, hist_len=self.history_size, state_parser=state_parser)
        replay_buffer = SuccessReplayBuffer(capacity=buffer_size, hist_len=self.history_size, state_parser=state_parser)

        vis = viz(visualize, vis_name)
        losses = []
        td_errs = []
        rewards = {}
        for key in test_params.keys():
            rewards[key] = []
        successes = []
        tps = []

        t = 0
        start_time = time.clock()
        previous_step = 0
        eval_required = False
        while t <= total_timesteps:
            obs = self.env.reset(seed=game_seed)
            done = False
            self._new_game_started()

            full_state = torch.zeros((self.history_size, 2, self.input_width, self.input_length),
                                     dtype=torch.float32).to(self.device)
            try:
                while not done:
                    raw_obs = obs
                    obs = self._parse_state(obs).view(2, self.input_width, self.input_length)
                    full_state[:self.history_size - 1] = full_state[1:]
                    full_state[-1] = obs

                    with torch.no_grad():
                        eps = 1.0 if t < learn_start_steps else exploration.value(t - learn_start_steps)
                        action, text_command = self._get_action(full_state.unsqueeze(0),
                                                                tau=tau,
                                                                eps=eps,
                                                                additional_prints=False,
                                                                **train_params)

                    new_obs, reward, done, has_won, timeout = self.env.step(text_command)

                    replay_buffer.add(state_to_memory(raw_obs, obs), action.squeeze(0).cpu(), reward, float(done), has_won, timeout)
                    obs = new_obs

                    if done:
                        self._append_and_plot(successes, int(has_won), t, 'Successes', vis)

                    if t % train_interval == 0 and t >= learn_start_steps:
                        loss, td_error = self._train(target_network=target_network, gamma=gamma,
                                                     replay_buffer=replay_buffer, batch_size=batch_size,
                                                     optimizer=optimizer,
                                                     **train_params)
                        if loss is not None:
                            self._append_and_plot(losses, loss, t, 'Loss', vis)
                            self._append_and_plot(td_errs, td_error, t, 'TD Error', vis)

                    if t % target_update_interval == 0 and t >= learn_start_steps:
                        self._copy_network(self.network, target_network)

                    if t % test_interval == 0 and t >= learn_start_steps:
                        eval_required = True

                    t += 1
            except EnvironmentError:
                print('There was some issue with the Zork env.')

            if eval_required:
                average_tps = t * 1.0 / (time.clock() - start_time)
                for key in test_params.keys():
                    avg_reward = self.test(test_params[key], game_seed)

                    self._append_and_plot(rewards[key], avg_reward, t, 'Rewards_' + key, vis)

                self._append_and_plot(tps, average_tps, t, 'Steps per second', vis)

                eval_required = False

                self.save_model(vis_name,
                                {'loss': losses,
                                 'td_errs': td_errs,
                                 'rewards': rewards,
                                 'successes': successes,
                                 'tps': tps},
                                **train_params
                                )

        return {'loss': losses, 'td_errs': td_errs, 'rewards': rewards, 'successes': successes, 'tps': tps}

    def test(self, test_params, game_seed):
        total_reward = 0
        iteration = 0
        test_iterations = 1
        with torch.no_grad():
            while iteration < test_iterations:
                reward = 0
                additional_prints = True
                # if 'number_of_neighbors' in test_params:
                #     additional_prints = (iteration == 0) and test_params['number_of_neighbors'] == 1
                # else:
                #     additional_prints = (iteration == 0)

                try:
                    obs = self.env.reset(seed=game_seed)
                    done = False

                    full_state = torch.zeros((self.history_size,
                                              2,
                                              self.input_width,
                                              self.input_length), dtype=torch.float32).to(self.device)

                    episode_reward = 0
                    while not done:
                        obs = self._parse_state(obs).view(2, self.input_width, self.input_length)
                        full_state[:self.history_size - 1] = full_state[1:]
                        full_state[-1] = obs

                        action, text_command = self._get_action(full_state.unsqueeze(0),
                                                                tau=0,
                                                                eps=0,
                                                                test=True,
                                                                additional_prints=False, #additional_prints,
                                                                **test_params)
                        if additional_prints:
                            self.env.render()
                            print(text_command)
                            print(action)
                            print(reward)
                            print(self._get_q_value(self.network,
                                                    full_state.unsqueeze(0),
                                                    action))

                        obs, reward, done, has_won, timeout = self.env.step(text_command)

                        episode_reward += reward

                    if additional_prints:
                        self.env.render()

                    total_reward += episode_reward
                    iteration += 1
                except EnvironmentError:
                    print('There was some issue with the Zork test env.')

        return total_reward * 1.0 / test_iterations

    def _train(self, target_network, gamma, replay_buffer, batch_size, optimizer, **kwargs):
        if len(replay_buffer) < batch_size:
            return None, None
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
        obs_batch = obs_batch.to(self.device).detach()
        act_batch = torch.tensor(act_batch, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rew_batch = torch.tensor(rew_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs_batch = next_obs_batch.to(self.device)

        not_done_mask = 1 - torch.tensor(done_mask, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_state_actions = self._get_action(next_obs_batch, eps=0.0, tau=0.0)[0]
            next_Q = self._get_q_value(target_network, next_obs_batch, next_state_actions) * not_done_mask
            target_Q = rew_batch + (gamma * next_Q)
        current_Q = self._get_q_value(self.network, obs_batch, act_batch)

        loss = F.smooth_l1_loss(current_Q, target_Q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        td_error = (current_Q - target_Q).mean().item()

        return loss.item(), td_error

    def _new_game_started(self):
        pass

    def _get_text_command(self, action):
        return self.actions[action[0]]

    def _create_network(self):
        return copy.deepcopy(self.network).to(self.device)

    @abstractmethod
    def _get_action(self, state, **kwargs):
        pass

    @abstractmethod
    def _get_q_value(self, network, state, action):
        pass

    @staticmethod
    def _copy_network(network, target_network):
        target_network.load_state_dict(network.state_dict())
        return target_network

    def _create_optimizer(self, lr):
        return optim.RMSprop(self.network.parameters(),
                             lr=lr,
                             eps=0.00001,
                             momentum=0.0,
                             alpha=0.95,
                             centered=True)

    @staticmethod
    def _create_env(task, output_parser, word_tokenizer, game_location, success_reward, pomdp_mode, sparse_reward):
        if task == 'full':
            max_steps = 420
        else:
            max_steps = 200
        return ZorkEnv(task=task, output_parser=output_parser, word_tokenizer=word_tokenizer,
                       game_location=game_location, success_reward=success_reward, pomdp_mode=pomdp_mode,
                       max_steps=max_steps, sparse_reward=sparse_reward)

    @abstractmethod
    def save_model(self, project_name, results, **kwargs):
        pass

    def _append_and_plot(self, values, value, timestep, plot_name, vis):
        values.append((timestep, value))
        vis_plot(vis, self.name + ' ' + plot_name, values)
        return values

    def _parse_state(self, state):
        if state is None:
            return torch.zeros((1, 2, self.input_width, self.input_length), dtype=torch.float32).to(self.device)
        return self.parser([state[0], state[1]])

    def _create_action_embeddings(self, action_parser):
        action_embeddings = []
        for action in self.actions:
            action_embeddings.append(action_parser([tokenizer(action)])[0, 0])
        return torch.stack(action_embeddings)

    def _tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            self.device)
        return torch.index_select(a, dim, order_index)
