from collections import deque

import torch
import torch.nn as nn

from .model_crnn import DataConsistencyInKspace


class CRNNcell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, multi_hidden_t=1):
        """
        Convolutional RNN cell that evolves over both time and iterations
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param multi_hidden_t: ...
        """
        super(CRNNcell, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.multi_hidden_t = multi_hidden_t
        # image2hidden conv
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # hidden(from the neighbour frame)2hidden conv
        self.h2h = nn.ModuleList()
        for i in range(multi_hidden_t):
            self.h2h.append(nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2))
        # hidden(from the previous iter)2hidden conv
        self.ih2ih = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_i, hidden_t):
        """
        :param input: the input from the previous layer, with shape [batch_size, input_size, x, y]
        :param hidden_i: the hidden states of the previous iteration, with shape [batch_size, input_size, x, y]
        :param hidden_t: torch tensor or an iterable container of torch tensor(s), the hidden states of the neighbour
        frame(s), with shape [batch_size, hidden_size, x, y]
        :return: hidden state with shape [batch_size, hidden_size, width, height]
        """
        in_to_hid = self.i2h(input)
        ih_to_ih = self.ih2ih(hidden_i)

        # make sure the given hidden_t matches self.multi_hidden_t
        if not isinstance(hidden_t, deque):
            hidden_t = deque([hidden_t])
        assert self.multi_hidden_t == len(hidden_t)

        hid_to_hid = torch.zeros_like(ih_to_ih)
        for i in range(self.multi_hidden_t):
            hid_to_hid += self.h2h[i](hidden_i.new_zeros((*hidden_i.shape[:1], self.hidden_size, *hidden_i.shape[2:]))
                                      if hidden_t[i] is None else hidden_t[i])

        hidden_out = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden_out


class resCRNN_t_i(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, uni_direction=False, multi_hidden_t=1):
        """
        Recurrent Convolutional RNN layer over iterations and time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(resCRNN_t_i, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.uni_direction = uni_direction
        self.multi_hidden_t = multi_hidden_t
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size, multi_hidden_t)
        self.bottleneck = nn.Conv2d(hidden_size, input_size, kernel_size, padding=kernel_size // 2)

    def forward(self, input, hidden_i):
        """
        :param input: the input from the previous layer, [num_seqs, batch_size, channel, width, height]
        :param hidden_i: the hidden state from the previous iteration, [num_seqs, batch_size, hidden_size, width, height]
        :return: hidden state, [num_seqs, batch_size, hidden_size, width, height]
        """
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = input.new_zeros(size_h)

        # forward
        output = []
        hidden_t_pool = deque([None] * self.multi_hidden_t, maxlen=self.multi_hidden_t)
        for i in range(nt):  # past time frame
            hidden = self.CRNN_model(input[i], hidden_i[i], hidden_t_pool)
            # Note that the following two sentences do not make memory overhead double
            hidden_t_pool.append(hidden)
            hidden = self.bottleneck(hidden)
            output.append(hidden)
        output = torch.stack(output, dim=0)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden_t_pool = deque([None] * self.multi_hidden_t, maxlen=self.multi_hidden_t)
            for i in range(nt):  # future time frame
                hidden = self.CRNN_model(input[nt - i - 1], hidden_i[nt - i - 1], hidden_t_pool)
                hidden_t_pool.append(hidden)
                hidden = self.bottleneck(hidden)
                output_b.append(hidden)
            output_b = torch.stack(output_b[::-1], dim=0)
            output = output + output_b

        output = output + input

        return output


class resCRNN(nn.Module):
    def __init__(self, n_ch=2, nf=48, ks=3, nc=5, nd=4, uni_direction=False):
        """
        Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
        unidirectional version
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        :param uni_direction: ...
        """
        super(resCRNN, self).__init__()
        self.n_ch = n_ch
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks
        self.uni_direction = uni_direction

        self.crnn_t_i_1 = resCRNN_t_i(n_ch, nf, ks, uni_direction)
        self.crnn_t_i_2 = resCRNN_t_i(n_ch, nf, ks, uni_direction)
        self.crnn_t_i_3 = resCRNN_t_i(n_ch, nf, ks, uni_direction)
        self.conv4 = nn.Conv2d(n_ch, n_ch, ks, padding=ks // 2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m):
        """
        :param x: input in image domain, [batch_size, 2, width, height, n_seq]
        :param k: initially sampled elements in k-space, [batch_size, 2, width, height, n_seq]
        :param m: corresponding nonzero location, [batch_size, 2, width, height, n_seq]
        :return: reconstruction result, [batch_size, 2, width, height, n_seq]
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq, n_batch, n_ch, width, height]
        hid_init = x.new_zeros(size_h)  # the initial zero-valued hidden state (the same device and dtype as x)

        # initialize hidden states
        for j in range(self.nd - 1):  # except for the last vanilla CNN layer, all layers maintain a hidden state
            net['t0_x%d' % j] = hid_init  # 't' means iteration here

        # iterate
        for i in range(1, self.nc + 1):  # i: number of iteration
            x = x.permute(4, 0, 1, 2, 3)  # [n_seq, batch, n_ch, width, height]

            # 3 layers of resCRNN-t-i
            net['t%d_x0' % i] = self.crnn_t_i_1(x, net['t%d_x0' % (i - 1)])
            net['t%d_x1' % i] = self.crnn_t_i_2(net['t%d_x0' % i], net['t%d_x1' % (i - 1)])
            net['t%d_x2' % i] = self.crnn_t_i_3(net['t%d_x1' % i], net['t%d_x2' % (i - 1)])

            # 1 layer of vanilla CNN
            net['t%d_x3' % i] = self.conv4(
                net['t%d_x2' % i].view(-1, n_ch, width, height)
            ).view(n_seq, n_batch, n_ch, width, height)

            # outer shortcut
            net['t%d_out' % i] = x + net['t%d_x3' % i]

            net['t%d_out' % i] = net['t%d_out' % i].permute(1, 2, 3, 4, 0)  # (batch_size, n_ch, width, height, n_seq)

            x = self.dcs[i - 1].perform(net['t%d_out' % i], k, m)  # data consistency layer

        return x

    def queue_forward(self, x, k, m, h=None):
        raise NotImplementedError
