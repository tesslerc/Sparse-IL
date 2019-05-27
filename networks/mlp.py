import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBow(nn.Module):
    def __init__(self, dict_size, output_size, hidden_layers, history_size):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(dict_size * 2 * history_size, hidden_layers[0]))
        for idx in range(len(hidden_layers) - 1):
            self.linears.append(nn.Linear(hidden_layers[idx], hidden_layers[idx + 1]))

        self.linears.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        x_relu = x.view(x.size(0), -1)
        for idx in range(len(self.linears)):
            x = self.linears[idx](x_relu)
            x_relu = F.relu(x)
        return x

    def get_action_stream(self, x):
        return x

    def get_state_stream(self, x):
        return self.forward(x)

    def get_q(self, s, a):
        return torch.bmm(s.unsqueeze(1), a.unsqueeze(-1))


class MlpDRRN(nn.Module):
    """
    He et. al (2016), http://www.aclweb.org/anthology/P16-1153
    """
    def __init__(self, dict_size, action_size, hidden_layers, history_size):
        super().__init__()
        self.state_stream = nn.ModuleList()
        self.action_stream = nn.ModuleList()

        self.state_stream.append(nn.Linear(dict_size * 2 * history_size, hidden_layers[0]))
        self.action_stream.append(nn.Linear(action_size, hidden_layers[0]))
        for idx in range(len(hidden_layers) - 1):
            self.state_stream.append(nn.Linear(hidden_layers[idx], hidden_layers[idx + 1]))
            self.action_stream.append(nn.Linear(hidden_layers[idx], hidden_layers[idx + 1]))

    def forward(self, s, a):
        s = self.get_state_stream(s)
        a = self.get_action_stream(a)

        return torch.bmm(s.unsqueeze(1), a.unsqueeze(-1))

    def _single_stream(self, x, module_list):
        x_relu = x.view(x.size(0), -1)
        for idx in range(len(module_list)):
            x = module_list[idx](x_relu)
            x_relu = F.relu(x)
        return x

    def get_action_stream(self, a):
        a = self._single_stream(a, self.action_stream)
        return a

    def get_state_stream(self, s):
        s = self._single_stream(s, self.state_stream)
        return s

    def get_q(self, s, a):
        return torch.bmm(s.unsqueeze(1), a.unsqueeze(-1))
