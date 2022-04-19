from collections import deque

import torch
import torch.nn as nn

from model import DataConsistencyInKspace


class CRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation):
        """
        Convolutional RNN cell that evolves over both time and iterations
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        """
        super(CRNNCell, self).__init__()
        # input2hidden conv
        self.input2hidden = nn.Conv2d(input_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        # hidden(from the neighbour frame)2hidden conv
        self.hidden_t2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden_t):
        """
        :param input: the input from the previous layer, with shape [batch_size, input_size, x, y]
        :param hidden_t: torch tensor or an iterable container of torch tensor(s), the hidden states of the neighbour
        frame(s), with shape [batch_size, hidden_size, x, y]
        :return: hidden state with shape [batch_size, hidden_size, width, height]
        """
        hidden_from_input = self.input2hidden(input)
        hidden_from_hidden_t = self.hidden_t2hidden(hidden_t)
        # print(hidden_from_input.shape)
        # print(hidden_from_hidden_t.shape)
        hidden_out = self.leaky_relu(hidden_from_input + hidden_from_hidden_t)

        return hidden_out


class CRNN_t(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation, uni_direction=False):
        """
        Recurrent Convolutional RNN layer over time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(CRNN_t, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.uni_direction = uni_direction
        self.crnn_cell = CRNNCell(self.input_size, self.hidden_size, self.kernel_size, self.dilation)

    def forward(self, input):
        """
        :param input: the input from the previous layer, [num_seqs, batch_size, channel, width, height]
        :return: hidden state, [num_seqs, batch_size, hidden_size, width, height]
        """
        nt, nb, nc, nx, ny = input.shape
        hid_init = input.new_zeros([nb, self.hidden_size, nx, ny])

        # forward
        output = []
        hidden = hid_init
        for i in range(nt):  # past time frame
            hidden = self.crnn_cell(input[i], hidden)
            output.append(hidden)
        output = torch.stack(output, dim=0)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden = hid_init
            for i in range(nt):  # future time frame
                hidden = self.crnn_cell(input[nt - i - 1], hidden)
                output_b.append(hidden)
            output_b = torch.stack(output_b[::-1], dim=0)
            output = output + output_b

        return output


class CRNN_V2(nn.Module):
    def __init__(self, input_size=2,
                 hidden_size=64,
                 kernel_size=3,
                 dilation=3,
                 iteration=4,
                 uni_direction=False):
        """
        CRNN_V2
        1. only use time connections, no iteration connections
        2. dilation convolution
        3. leaky relu
        4. inner residual connection
        """
        super(CRNN_V2, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.uni_direction = uni_direction

        self.crnn_t_1 = CRNN_t(input_size, hidden_size, kernel_size, 1, uni_direction)
        self.crnn_t_2 = CRNN_t(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
        self.crnn_t_3 = CRNN_t(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
        self.crnn_t_4 = CRNN_t(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
        self.conv4_x = nn.Conv2d(hidden_size, input_size, kernel_size, padding='same')
        self.dcs = [DataConsistencyInKspace(norm='ortho') for _ in range(iteration)]

    def forward(self, x, k, m):
        """
        :param x: input in image domain, [batch_size, 2, width, height, t]
        :param k: initially sampled elements in k-space, [batch_size, 2, width, height, t]
        :param m: corresponding nonzero location, [batch_size, 2, width, height, t]
        :return: reconstruction result, [batch_size, 2, width, height, t]
        """
        n_batch, n_ch, width, height, t = x.shape

        for i in range(self.iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]

            out = self.crnn_t_1(x)
            out = self.crnn_t_2(out) + out
            out = self.crnn_t_3(out) + out
            out = self.crnn_t_4(out) + out

            out = out.view(-1, self.hidden_size, width, height)  # -> [t * batch_size, hidden_size, width, height]
            out = self.conv4_x(out)
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x
