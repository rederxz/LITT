import math

import torch
import torch.nn as nn

from .model_crnn import DataConsistencyInKspace


class ConvST(nn.Module):
    def __init__(self, input_size, output_size, space_kernel_size, time_kernel_size):
        super(ConvST, self).__init__()
        inter_size = math.ceil(
            (space_kernel_size ** 2 * time_kernel_size * input_size * output_size)
            / (space_kernel_size ** 2 * input_size + time_kernel_size * output_size)
        )
        self.conv_space = nn.Conv2d(input_size, inter_size, space_kernel_size, padding='same')
        self.conv_time = nn.Conv1d(inter_size, output_size, time_kernel_size, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        """
        Args:
            x: [num_seqs, batch_size, channel, width, height]

        Returns:
            [num_seqs, batch_size, channel, width, height]

        """
        t, batch_size, input_size, width, height = x.shape

        # space convolution
        x = x.reshape(t * batch_size, input_size, width, height)
        x = self.conv_space(x)
        x = x.reshape(t, batch_size, -1, width, height)

        x = self.leaky_relu(x)

        # time convolution
        x = x.permute(1, 3, 4, 2, 0)
        x = x.reshape(batch_size * width * height, -1, t)
        x = self.conv_time(x)
        x = x.reshape(batch_size, width, height, -1, t)
        x = x.permute(4, 0, 3, 1, 2)

        # x = self.leaky_relu(x)

        return x


class CNN_ST(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, space_kernel_size=3, time_kernel_size=5, n_iteration=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CNN_ST, self).__init__()
        self.n_iteration = n_iteration
        self.hidden_size = hidden_size

        self.conv_st_1 = ConvST(input_size, hidden_size, space_kernel_size, time_kernel_size)
        self.conv_st_1_i = ConvST(hidden_size, hidden_size, space_kernel_size, time_kernel_size)
        self.conv_2 = nn.Conv2d(hidden_size, hidden_size, space_kernel_size, padding='same')
        self.conv_2_i = nn.Conv2d(hidden_size, hidden_size, space_kernel_size, padding='same')
        self.conv_3 = nn.Conv2d(hidden_size, hidden_size, space_kernel_size, padding='same')
        self.conv_3_i = nn.Conv2d(hidden_size, hidden_size, space_kernel_size, padding='same')
        self.conv_4 = nn.Conv2d(hidden_size, hidden_size, space_kernel_size, padding='same')
        self.conv_4_i = nn.Conv2d(hidden_size, hidden_size, space_kernel_size, padding='same')
        self.conv_5 = nn.Conv2d(hidden_size, input_size, space_kernel_size, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

        self.dcs = [DataConsistencyInKspace(norm='ortho') for _ in range(n_iteration)]

    def forward(self, x, k, m):
        """
        :param x: input in image domain, [batch_size, 2, width, height, t]
        :param k: initially sampled elements in k-space, [batch_size, 2, width, height, t]
        :param m: corresponding nonzero location, [batch_size, 2, width, height, t]
        :return: reconstruction result, [batch_size, 2, width, height, t]
        """
        n_batch, n_ch, width, height, t = x.shape

        hidden_last_iteration = dict()
        hidden_last_iteration['conv_st_1'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['conv_2'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['conv_3'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['conv_4'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])

        for i in range(self.n_iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]

            hidden_last_iteration['conv_st_1'] = self.leaky_relu(
                self.conv_st_1(x) + self.conv_st_1_i(hidden_last_iteration['conv_st_1'])
            )
            hidden_last_iteration['conv_2'] = self.leaky_relu(
                # -> [t * batch_size, 2, width, height]
                self.conv_2(hidden_last_iteration['conv_st_1'].view(-1, self.hidden_size, width, height))
                + self.conv_2_i(hidden_last_iteration['conv_2'])
            )
            hidden_last_iteration['conv_3'] = self.leaky_relu(
                self.conv_3(hidden_last_iteration['conv_2'])
                + self.conv_3_i(hidden_last_iteration['conv_3'])
            )
            hidden_last_iteration['conv_4'] = self.leaky_relu(
                self.conv_4(hidden_last_iteration['conv_3'])
                + self.conv_4_i(hidden_last_iteration['conv_4'])
            )

            out = self.conv_5(hidden_last_iteration['conv_4'])
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x
