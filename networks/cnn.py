import torch.nn as nn
import torch.nn.functional as F
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):
    def __init__(self, embedding_size, output_size, history_size, input_width):
        super().__init__()

        self.input_tokens = input_width * 2  # both state and inventory
        self.embedding_size = embedding_size
        self.hist = history_size
        self.output_size = output_size

        kernels = (2, 4)
        n_filters = 64

        self.net = nn.ModuleList()
        self.net.append(nn.Conv2d(in_channels=self.hist, out_channels=n_filters,
                                  kernel_size=(kernels[0], self.embedding_size)
                                  )
                        )
        self.net.append(nn.ReLU())
        for idx in range(len(kernels) - 1):
            self.net.append(nn.Conv2d(in_channels=n_filters, out_channels=n_filters,
                                      kernel_size=(kernels[idx + 1], 1)
                                      )
                            )
            self.net.append(nn.ReLU())

        self.net.append(Flatten())
        self.net.append(nn.Linear(8064, 512))
        self.net.append(nn.ReLU())
        self.net.append(nn.Linear(512, self.output_size))

    def forward(self, x):
        x = x.view(x.size(0), self.hist, self.input_tokens, self.embedding_size)
        for idx in range(len(self.net)):
            x = self.net[idx](x)

        return x


class TextCNN(nn.Module):
    def __init__(self, embedding_size, output_size, history_size, input_width):
        super().__init__()

        self.input_tokens = input_width * 2  # both state and inventory
        self.input_width = input_width
        self.embedding_size = embedding_size
        self.hist = history_size
        self.output_size = output_size
        self.n_filters = 500

        self.conv2 = nn.Conv2d(self.hist, self.n_filters, (2, self.embedding_size))
        self.conv3 = nn.Conv2d(self.hist, self.n_filters, (3, self.embedding_size))
        self.conv4 = nn.Conv2d(self.hist, self.n_filters, (4, self.embedding_size))
        self.conv6 = nn.Conv2d(self.hist, self.n_filters, (6, self.embedding_size))

        self.linear1 = nn.Linear(4 * self.n_filters, self.output_size)

    def forward(self, x):
        x = x.view(x.size(0), self.hist, self.input_tokens, self.embedding_size)
        # Convolution
        x2 = F.relu(self.conv2(x)).squeeze(3)
        x3 = F.relu(self.conv3(x)).squeeze(3)
        x4 = F.relu(self.conv4(x)).squeeze(3)
        x6 = F.relu(self.conv6(x)).squeeze(3)

        # Pooling
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
        x6 = F.max_pool1d(x6, x6.size(2)).squeeze(2)

        # capture and concatenate the features
        x = torch.cat([x2, x3, x4, x6], 1)

        # project the features to the labels
        x = self.linear1(x)

        return x

    def get_action_stream(self, x):
        return x

    def get_state_stream(self, x):
        return self.forward(x)

    def get_q(self, s, a):
        return torch.bmm(s.unsqueeze(1), a.unsqueeze(-1))


class TextCNN_BIG(nn.Module):
    def __init__(self, embedding_size, output_size, history_size, input_width):
        super().__init__()

        self.input_tokens = input_width * 2  # both state and inventory
        self.input_width = input_width
        self.embedding_size = embedding_size
        self.hist = history_size
        self.output_size = output_size
        self.n_filters = 500

        self.conv2 = nn.Conv2d(self.hist, self.n_filters, (2, self.embedding_size))
        self.conv3 = nn.Conv2d(self.hist, self.n_filters, (3, self.embedding_size))
        self.conv4 = nn.Conv2d(self.hist, self.n_filters, (4, self.embedding_size))
        self.conv6 = nn.Conv2d(self.hist, self.n_filters, (6, self.embedding_size))

        self.linear1 = nn.Linear(4 * self.n_filters, self.output_size)

    def forward(self, x):
        x = x.view(x.size(0), self.hist, self.input_tokens, self.embedding_size)
        # Convolution
        x2 = F.relu(self.conv2(x)).squeeze(3)
        x3 = F.relu(self.conv3(x)).squeeze(3)
        x4 = F.relu(self.conv4(x)).squeeze(3)
        x6 = F.relu(self.conv6(x)).squeeze(3)

        # Pooling
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
        x6 = F.max_pool1d(x6, x6.size(2)).squeeze(2)

        # capture and concatenate the features
        x = torch.cat([x2, x3, x4, x6], 1)

        # project the features to the labels
        x = self.linear1(x)

        return x

    def get_action_stream(self, x):
        return x

    def get_state_stream(self, x):
        return self.forward(x)

    def get_q(self, s, a):
        return torch.bmm(s.unsqueeze(1), a.unsqueeze(-1))


class TextCNN_DRRN(nn.Module):
    """
    He et. al (2016), http://www.aclweb.org/anthology/P16-1153
    """
    def __init__(self, embedding_size, hidden_size, history_size, input_width, action_length):
        super().__init__()
        self.input_tokens = input_width * 2  # both state and inventory
        self.embedding_size = embedding_size
        self.action_length = action_length
        self.hist = history_size
        self.hidden_size = hidden_size
        self.n_filters = 100

        self.sconv2 = nn.Conv2d(self.hist, self.n_filters, (2, self.embedding_size))
        self.sconv3 = nn.Conv2d(self.hist, self.n_filters, (3, self.embedding_size))
        self.sconv4 = nn.Conv2d(self.hist, self.n_filters, (4, self.embedding_size))
        self.sconv6 = nn.Conv2d(self.hist, self.n_filters, (6, self.embedding_size))

        self.aconv2 = nn.Conv2d(self.hist, self.n_filters, (2, self.embedding_size))
        self.aconv3 = nn.Conv2d(self.hist, self.n_filters, (3, self.embedding_size))
        self.aconv4 = nn.Conv2d(self.hist, self.n_filters, (4, self.embedding_size))

        self.state_linear = nn.Linear(4 * self.n_filters, self.hidden_size)

        self.action_linear = nn.Linear(3 * self.hidden_size, self.hidden_size)

    def forward(self, s, a):
        s = self.get_state_stream(s)
        a = self.get_action_stream(a)
        return torch.bmm(s.view(s.shape[0], 1, -1), a.view(s.shape[0], -1, 1))

    def get_state_stream(self, s):
        s = s.view(s.size(0), self.hist, self.input_tokens, self.embedding_size)
        # Convolution
        s2 = F.relu(self.sconv2(s)).squeeze(3)
        s3 = F.relu(self.sconv3(s)).squeeze(3)
        s4 = F.relu(self.sconv4(s)).squeeze(3)
        s6 = F.relu(self.sconv6(s)).squeeze(3)

        # Pooling
        s2 = F.max_pool1d(s2, s2.size(2)).squeeze(2)
        s3 = F.max_pool1d(s3, s3.size(2)).squeeze(2)
        s4 = F.max_pool1d(s4, s4.size(2)).squeeze(2)
        s6 = F.max_pool1d(s6, s6.size(2)).squeeze(2)

        # capture and concatenate the features
        s = torch.cat([s2, s3, s4, s6], 1)

        # project the features to the labels
        s = self.state_linear(s)
        return s

    def get_action_stream(self, a):
        a = a.view(a.size(0), 1, self.action_length, self.embedding_size)
        # Convolution
        a2 = F.relu(self.aconv2(a)).squeeze(3)
        a3 = F.relu(self.aconv3(a)).squeeze(3)
        a4 = F.relu(self.aconv4(a)).squeeze(3)

        # Pooling
        a2 = F.max_pool1d(a2, a2.size(2)).squeeze(2)
        a3 = F.max_pool1d(a3, a3.size(2)).squeeze(2)
        a4 = F.max_pool1d(a4, a4.size(2)).squeeze(2)

        # capture and concatenate the features
        a = torch.cat([a2, a3, a4], 1)

        # project the features to the labels
        a = self.action_linear(a)
        return a

    def get_q(self, s, a):
        return torch.bmm(s.unsqueeze(1), a.unsqueeze(-1))
