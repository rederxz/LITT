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
        self.input2hidden = nn.Conv2d(input_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.hidden_t2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.hidden_i2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden_t, hidden_i):
        """
        :param input: the input from the previous layer, with shape [batch_size, input_size, x, y]
        :param hidden_t: torch tensor, the hidden state of the neighbour, with shape [batch_size, hidden_size, x, y]
        :param hidden_i: torch tensor, the hidden state from the last iteration, with shape [batch_size, hidden_size, x, y]
        :return: hidden state with shape [batch_size, hidden_size, width, height]
        """
        hidden_from_input = self.input2hidden(input)
        hidden_from_hidden_t = self.hidden_t2hidden(hidden_t)
        hidden_from_hidden_i = self.hidden_i2hidden(hidden_i)
        hidden_out = self.leaky_relu(hidden_from_input + hidden_from_hidden_t + hidden_from_hidden_i)

        return hidden_out


class CRNN_t_i(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation, uni_direction=False):
        """
        Recurrent Convolutional RNN layer over time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(CRNN_t_i, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.uni_direction = uni_direction
        self.crnn_cell = CRNNCell(self.input_size, self.hidden_size, self.kernel_size, self.dilation)

    def forward(self, input, hidden_i):
        """
        Args:
            input: [num_seqs, batch_size, channel, width, height]
            hidden_i: [num_seqs, batch_size, hidden_size, width, height]

        Returns:
            [num_seqs, batch_size, hidden_size, width, height]
        """
        nt, nb, nc, nx, ny = input.shape
        hid_init = input.new_zeros([nb, self.hidden_size, nx, ny])

        # forward
        output = []
        hidden_t = hid_init
        for i in range(nt):  # past time frame
            hidden_t = self.crnn_cell(input[i], hidden_t, hidden_i[i])
            output.append(hidden_t)
        output = torch.stack(output, dim=0)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden_t = hid_init  # FIXME: 这里hidden_t和hid_init已经绑定，修改hidden_t会影响到hid_init！！！
            for i in range(nt):  # future time frame
                hidden_t = self.crnn_cell(input[nt - i - 1], hidden_t, hidden_i[nt - i - 1])
                output_b.append(hidden_t)
            output_b = torch.stack(output_b[::-1], dim=0)
            output = output + output_b

        return output


class CRNN_i(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation):
        """
        Recurrent Convolutional RNN layer over time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(CRNN_i, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.input2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.hidden_i2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, input, hidden_i):
        """
        Args:
            input: [t * batch_size, hidden_size, width, height]
            hidden_i: [t * batch_size, hidden_size, width, height]

        Returns:
            [t * batch_size, hidden_size, width, height]
        """

        x = self.leaky_relu(self.input2hidden(input) + self.hidden_i2hidden(hidden_i))

        return x


class CRNN_V2(nn.Module):
    def __init__(self, input_size=2,
                 hidden_size=64,
                 kernel_size=3,
                 dilation=3,
                 iteration=4,
                 uni_direction=False):
        """
        CRNN_V2
        1. only use CRNN_t_i
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

        self.crnn_t_i_1 = CRNN_t_i(input_size, hidden_size, kernel_size, 1, uni_direction)
        self.crnn_t_i_2 = CRNN_t_i(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
        self.crnn_t_i_3 = CRNN_t_i(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
        self.crnn_t_i_4 = CRNN_t_i(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
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

        hidden_last_iteration = dict()
        hidden_last_iteration['crnn_t_i_1'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_t_i_2'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_t_i_3'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_t_i_4'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])

        for i in range(self.iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]

            hidden_last_iteration['crnn_t_i_1'] = self.crnn_t_i_1(x, hidden_last_iteration['crnn_t_i_1'])
            hidden_last_iteration['crnn_t_i_2'] = self.crnn_t_i_2(
                hidden_last_iteration['crnn_t_i_1'], hidden_last_iteration['crnn_t_i_2']
            ) + hidden_last_iteration['crnn_t_i_1']
            hidden_last_iteration['crnn_t_i_3'] = self.crnn_t_i_3(
                hidden_last_iteration['crnn_t_i_2'], hidden_last_iteration['crnn_t_i_3']
            ) + hidden_last_iteration['crnn_t_i_2']
            hidden_last_iteration['crnn_t_i_4'] = self.crnn_t_i_4(
                hidden_last_iteration['crnn_t_i_3'], hidden_last_iteration['crnn_t_i_4']
            ) + hidden_last_iteration['crnn_t_i_3']

            out = hidden_last_iteration['crnn_t_i_4']\
                .view(-1, self.hidden_size, width, height)  # -> [t * batch_size, hidden_size, width, height]
            out = self.conv4_x(out)
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x


class CRNN_V2_without_inner_res(nn.Module):
    def __init__(self, input_size=2,
                 hidden_size=64,
                 kernel_size=3,
                 dilation=3,
                 iteration=4,
                 uni_direction=False):
        """
        CRNN_V2
        1. only use CRNN_t_i
        2. dilation convolution
        3. leaky relu
        4. inner residual connection
        """
        super(CRNN_V2_without_inner_res, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.uni_direction = uni_direction

        self.crnn_t_i_1 = CRNN_t_i(input_size, hidden_size, kernel_size, 1, uni_direction)
        self.crnn_t_i_2 = CRNN_t_i(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
        self.crnn_t_i_3 = CRNN_t_i(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
        self.crnn_t_i_4 = CRNN_t_i(hidden_size, hidden_size, kernel_size, dilation, uni_direction)
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

        hidden_last_iteration = dict()
        hidden_last_iteration['crnn_t_i_1'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_t_i_2'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_t_i_3'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_t_i_4'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])

        for i in range(self.iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]

            hidden_last_iteration['crnn_t_i_1'] = self.crnn_t_i_1(x, hidden_last_iteration['crnn_t_i_1'])
            hidden_last_iteration['crnn_t_i_2'] = self.crnn_t_i_2(
                hidden_last_iteration['crnn_t_i_1'], hidden_last_iteration['crnn_t_i_2']
            )
            hidden_last_iteration['crnn_t_i_3'] = self.crnn_t_i_3(
                hidden_last_iteration['crnn_t_i_2'], hidden_last_iteration['crnn_t_i_3']
            )
            hidden_last_iteration['crnn_t_i_4'] = self.crnn_t_i_4(
                hidden_last_iteration['crnn_t_i_3'], hidden_last_iteration['crnn_t_i_4']
            )

            out = hidden_last_iteration['crnn_t_i_4']\
                .view(-1, self.hidden_size, width, height)  # -> [t * batch_size, hidden_size, width, height]
            out = self.conv4_x(out)
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x


class CRNN_V2_1_3_1(nn.Module):
    def __init__(self, input_size=2,
                 hidden_size=64,
                 kernel_size=3,
                 dilation=3,
                 iteration=4,
                 uni_direction=False):
        """
        CRNN_V2_1_3_1
        1. dilation convolution
        2. leaky relu
        """
        super(CRNN_V2_1_3_1, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.uni_direction = uni_direction

        self.crnn_t_i = CRNN_t_i(input_size, hidden_size, kernel_size, 1, uni_direction)
        self.crnn_i_1 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_2 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_3 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
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

        hidden_last_iteration = dict()
        hidden_last_iteration['crnn_t_i'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_1'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_2'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_3'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])

        for i in range(self.iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]
            hidden_last_iteration['crnn_t_i'] = self.crnn_t_i(x, hidden_last_iteration['crnn_t_i'])
            out = hidden_last_iteration['crnn_t_i']\
                .view(-1, self.hidden_size, width, height)  # -> [t * batch_size, hidden_size, width, height]
            hidden_last_iteration['crnn_i_1'] = self.crnn_i_1(
                out, hidden_last_iteration['crnn_i_1']
            )
            hidden_last_iteration['crnn_i_2'] = self.crnn_i_2(
                hidden_last_iteration['crnn_i_1'], hidden_last_iteration['crnn_i_2']
            )
            hidden_last_iteration['crnn_i_3'] = self.crnn_i_3(
                hidden_last_iteration['crnn_i_2'], hidden_last_iteration['crnn_i_3']
            )
            out = self.conv4_x(hidden_last_iteration['crnn_i_3'])
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x


class CRNN_V2_no_dilation(nn.Module):
    def __init__(self, input_size=2,
                 hidden_size=64,
                 kernel_size=3,
                 dilation=1,
                 iteration=5,
                 uni_direction=False):
        """
        CRNN_V2_no_dilation
        1. leaky relu
        """
        super(CRNN_V2_no_dilation, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.uni_direction = uni_direction

        self.crnn_t_i = CRNN_t_i(input_size, hidden_size, kernel_size, 1, uni_direction)
        self.crnn_i_1 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_2 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_3 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
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

        hidden_last_iteration = dict()
        hidden_last_iteration['crnn_t_i'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_1'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_2'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_3'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])

        for i in range(self.iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]
            hidden_last_iteration['crnn_t_i'] = self.crnn_t_i(x, hidden_last_iteration['crnn_t_i'])
            out = hidden_last_iteration['crnn_t_i']\
                .view(-1, self.hidden_size, width, height)  # -> [t * batch_size, hidden_size, width, height]
            hidden_last_iteration['crnn_i_1'] = self.crnn_i_1(
                out, hidden_last_iteration['crnn_i_1']
            )
            hidden_last_iteration['crnn_i_2'] = self.crnn_i_2(
                hidden_last_iteration['crnn_i_1'], hidden_last_iteration['crnn_i_2']
            )
            hidden_last_iteration['crnn_i_3'] = self.crnn_i_3(
                hidden_last_iteration['crnn_i_2'], hidden_last_iteration['crnn_i_3']
            )
            out = self.conv4_x(hidden_last_iteration['crnn_i_3'])
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x


class CRNN_t_i_fix(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation, uni_direction=False):
        """
        Recurrent Convolutional RNN layer over time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(CRNN_t_i_fix, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.uni_direction = uni_direction
        self.crnn_cell = CRNNCell(self.input_size, self.hidden_size, self.kernel_size, self.dilation)

    def forward(self, input, hidden_i):
        """
        Args:
            input: [num_seqs, batch_size, channel, width, height]
            hidden_i: [num_seqs, batch_size, hidden_size, width, height]

        Returns:
            [num_seqs, batch_size, hidden_size, width, height]
        """
        nt, nb, nc, nx, ny = input.shape

        # forward
        output = []
        hidden_t = input.new_zeros([nb, self.hidden_size, nx, ny])
        for i in range(nt):  # past time frame
            hidden_t = self.crnn_cell(input[i], hidden_t, hidden_i[i])
            output.append(hidden_t)
        output = torch.stack(output, dim=0)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden_t = input.new_zeros([nb, self.hidden_size, nx, ny])
            for i in range(nt):  # future time frame
                hidden_t = self.crnn_cell(input[nt - i - 1], hidden_t, hidden_i[nt - i - 1])
                output_b.append(hidden_t)
            output_b = torch.stack(output_b[::-1], dim=0)
            output = output + output_b

        return output


class CRNN_V2_fix(nn.Module):
    def __init__(self, input_size=2,
                 hidden_size=64,
                 kernel_size=3,
                 dilation=1,
                 iteration=5,
                 uni_direction=False):
        """
        CRNN_V2_fix
        1. leaky relu
        2. fix CRNN_t_i bug
        """
        super(CRNN_V2_fix, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.uni_direction = uni_direction

        self.crnn_t_i = CRNN_t_i_fix(input_size, hidden_size, kernel_size, 1, uni_direction)
        self.crnn_i_1 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_2 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_3 = CRNN_i(hidden_size, hidden_size, kernel_size, dilation)
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

        hidden_last_iteration = dict()
        hidden_last_iteration['crnn_t_i'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_1'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_2'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_3'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])

        for i in range(self.iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]
            hidden_last_iteration['crnn_t_i'] = self.crnn_t_i(x, hidden_last_iteration['crnn_t_i'])
            out = hidden_last_iteration['crnn_t_i']\
                .view(-1, self.hidden_size, width, height)  # -> [t * batch_size, hidden_size, width, height]
            hidden_last_iteration['crnn_i_1'] = self.crnn_i_1(
                out, hidden_last_iteration['crnn_i_1']
            )
            hidden_last_iteration['crnn_i_2'] = self.crnn_i_2(
                hidden_last_iteration['crnn_i_1'], hidden_last_iteration['crnn_i_2']
            )
            hidden_last_iteration['crnn_i_3'] = self.crnn_i_3(
                hidden_last_iteration['crnn_i_2'], hidden_last_iteration['crnn_i_3']
            )
            out = self.conv4_x(hidden_last_iteration['crnn_i_3'])
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x


class CRNNCell_no_leaky(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation):
        """
        Convolutional RNN cell that evolves over both time and iterations
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        """
        super(CRNNCell_no_leaky, self).__init__()
        self.input2hidden = nn.Conv2d(input_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.hidden_t2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.hidden_i2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_t, hidden_i):
        """
        :param input: the input from the previous layer, with shape [batch_size, input_size, x, y]
        :param hidden_t: torch tensor, the hidden state of the neighbour, with shape [batch_size, hidden_size, x, y]
        :param hidden_i: torch tensor, the hidden state from the last iteration, with shape [batch_size, hidden_size, x, y]
        :return: hidden state with shape [batch_size, hidden_size, width, height]
        """
        hidden_from_input = self.input2hidden(input)
        hidden_from_hidden_t = self.hidden_t2hidden(hidden_t)
        hidden_from_hidden_i = self.hidden_i2hidden(hidden_i)
        hidden_out = self.relu(hidden_from_input + hidden_from_hidden_t + hidden_from_hidden_i)

        return hidden_out


class CRNN_t_i_no_leaky(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation, uni_direction=False):
        """
        Recurrent Convolutional RNN layer over time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(CRNN_t_i_no_leaky, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.uni_direction = uni_direction
        self.crnn_cell = CRNNCell_no_leaky(self.input_size, self.hidden_size, self.kernel_size, self.dilation)

    def forward(self, input, hidden_i):
        """
        Args:
            input: [num_seqs, batch_size, channel, width, height]
            hidden_i: [num_seqs, batch_size, hidden_size, width, height]

        Returns:
            [num_seqs, batch_size, hidden_size, width, height]
        """
        nt, nb, nc, nx, ny = input.shape

        # forward
        output = []
        hidden_t = input.new_zeros([nb, self.hidden_size, nx, ny])
        for i in range(nt):  # past time frame
            hidden_t = self.crnn_cell(input[i], hidden_t, hidden_i[i])
            output.append(hidden_t)
        output = torch.stack(output, dim=0)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden_t = input.new_zeros([nb, self.hidden_size, nx, ny])
            for i in range(nt):  # future time frame
                hidden_t = self.crnn_cell(input[nt - i - 1], hidden_t, hidden_i[nt - i - 1])
                output_b.append(hidden_t)
            output_b = torch.stack(output_b[::-1], dim=0)
            output = output + output_b

        return output


class CRNN_i_no_leaky(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation):
        """
        Recurrent Convolutional RNN layer over time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(CRNN_i_no_leaky, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.input2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.hidden_i2hidden = nn.Conv2d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_i):
        """
        Args:
            input: [t * batch_size, hidden_size, width, height]
            hidden_i: [t * batch_size, hidden_size, width, height]

        Returns:
            [t * batch_size, hidden_size, width, height]
        """

        x = self.relu(self.input2hidden(input) + self.hidden_i2hidden(hidden_i))

        return x


class CRNN_V2_no_leaky(nn.Module):
    def __init__(self, input_size=2,
                 hidden_size=64,
                 kernel_size=3,
                 dilation=1,
                 iteration=5,
                 uni_direction=False):
        super(CRNN_V2_no_leaky, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.uni_direction = uni_direction

        self.crnn_t_i = CRNN_t_i_no_leaky(input_size, hidden_size, kernel_size, 1, uni_direction)
        self.crnn_i_1 = CRNN_i_no_leaky(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_2 = CRNN_i_no_leaky(hidden_size, hidden_size, kernel_size, dilation)
        self.crnn_i_3 = CRNN_i_no_leaky(hidden_size, hidden_size, kernel_size, dilation)
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

        hidden_last_iteration = dict()
        hidden_last_iteration['crnn_t_i'] = x.new_zeros([t, n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_1'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_2'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])
        hidden_last_iteration['crnn_i_3'] = x.new_zeros([t * n_batch, self.hidden_size, width, height])

        for i in range(self.iteration):

            x = x.permute(4, 0, 1, 2, 3)  # [batch_size, 2, width, height, t] -> [t, batch_size, 2, width, height]
            hidden_last_iteration['crnn_t_i'] = self.crnn_t_i(x, hidden_last_iteration['crnn_t_i'])
            out = hidden_last_iteration['crnn_t_i']\
                .view(-1, self.hidden_size, width, height)  # -> [t * batch_size, hidden_size, width, height]
            hidden_last_iteration['crnn_i_1'] = self.crnn_i_1(
                out, hidden_last_iteration['crnn_i_1']
            )
            hidden_last_iteration['crnn_i_2'] = self.crnn_i_2(
                hidden_last_iteration['crnn_i_1'], hidden_last_iteration['crnn_i_2']
            )
            hidden_last_iteration['crnn_i_3'] = self.crnn_i_3(
                hidden_last_iteration['crnn_i_2'], hidden_last_iteration['crnn_i_3']
            )
            out = self.conv4_x(hidden_last_iteration['crnn_i_3'])
            out = out.view(-1, n_batch, 2, width, height)  # -> [t, batch_size, 2, width, height]

            out = out + x

            out = out.permute(1, 2, 3, 4, 0)  # [t, batch_size, 2, width, height] -> [batch_size, 2, width, height, t]

            x = self.dcs[i](out, k, m)

        return x

