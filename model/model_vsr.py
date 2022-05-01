import torch
import torch.nn as nn

from model_crnn import DataConsistencyInKspace


class CRNNcell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        """
        Convolutional RNN cell that evolves over both time and iterations
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        """
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        # image2hidden conv
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # hidden(from the neighbour frame)2hidden conv
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # hidden(from the previous iter)2hidden conv
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.fuse = nn.Conv2d(hidden_size*3, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_i, hidden_t):
        """
        :param input: the input from the previous layer, with shape [batch_size, input_size, x, y]
        :param hidden_i: the hidden states of the previous iteration, with shape [batch_size, hidden_size, x, y]
        :param hidden_t: the hidden states of the neighbour frame, with shape [batch_size, hidden_size, x, y]
        :return: hidden state with shape [batch_size, hidden_size, width, height]
        """
        in_to_hid = self.relu(self.i2h(input))
        ih_to_ih = self.relu(self.ih2ih(hidden_i))
        hid_to_hid = self.relu(self.h2h(torch.zeros_like(hidden_i) if hidden_t is None else hidden_t))
        aggregation = torch.cat([in_to_hid, hid_to_hid, ih_to_ih], dim=1)
        hidden_out = self.relu(self.fuse(aggregation))

        return hidden_out


class CRNN_t_i(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, uni_direction=False):
        """
        Recurrent Convolutional RNN layer over iterations and time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(CRNN_t_i, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.uni_direction = uni_direction
        self.CRNN_cell = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)
        if not self.uni_direction:
            self.fuse = nn.Conv2d(hidden_size*2, hidden_size, kernel_size, padding=self.kernel_size // 2)
            self.relu = nn.ReLU(inplace=True)

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
        hidden = hid_init
        for i in range(nt):  # past time frame
            hidden = self.CRNN_cell(input[i], hidden_i[i], hidden)
            output.append(hidden)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden = hid_init
            for i in range(nt):  # future time frame
                hidden = self.CRNN_cell(input[nt - i - 1], hidden_i[nt - i - 1], hidden)
                output_b.append(hidden)
            output_b = output_b[::-1]
            for i in range(nt):
                aggregation = torch.cat([output[i], output_b[i]], dim=1)
                output[i] = self.relu(self.fuse(aggregation))

        output = torch.stack(output, dim=0)

        return output


class CRNN(nn.Module):
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5, uni_direction=False):
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
        super(CRNN, self).__init__()
        self.n_ch = n_ch
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks
        self.uni_direction = uni_direction

        self.crnn_t_i = CRNN_t_i(n_ch, nf, ks, uni_direction)
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding=ks // 2)
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
        size_h = [n_seq * n_batch, self.nf, width, height]
        hid_init = x.new_zeros(size_h)  # the initial zero-valued hidden state (the same device and dtype as x)

        # initialize hidden states
        for j in range(self.nd - 1):  # except for the last vanilla CNN layer, all layers maintain a hidden state
            net['t0_x%d' % j] = hid_init  # 't' means iteration here

        # iterate
        for i in range(1, self.nc + 1):  # i: number of iteration
            x = x.permute(4, 0, 1, 2, 3)  # [n_seq, batch, n_ch, width, height]

            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)] \
                .view(n_seq, n_batch, self.nf, width, height)  # [n_seq, n_batch, self.nf, width, height] as required by CRNN_t_i

            # 1 layer of CRNN-t-i
            net['t%d_x0' % i] = self.crnn_t_i(x, net['t%d_x0' % (i - 1)])

            net['t%d_x0' % i] = net['t%d_x0' % i] \
                .view(-1, self.nf, width, height)  # [n_seq * n_batch, self.nf, width, height] as required by following CNN layers

            # 3 layers of CRNN-i
            net['t%d_x1' % i] = self.relu(self.conv1_x(net['t%d_x0' % i]) + self.conv1_h(net['t%d_x1' % (i - 1)]))
            net['t%d_x2' % i] = self.relu(self.conv2_x(net['t%d_x1' % i]) + self.conv2_h(net['t%d_x2' % (i - 1)]))
            net['t%d_x3' % i] = self.relu(self.conv3_x(net['t%d_x2' % i]) + self.conv3_h(net['t%d_x3' % (i - 1)]))

            # 1 layer of vanilla CNN
            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            # shortcut connection
            net['t%d_out' % i] = x + net['t%d_x4' % i].view(n_seq, n_batch, n_ch, width, height)

            net['t%d_out' % i] = net['t%d_out' % i].permute(1, 2, 3, 4, 0)  # (batch_size, n_ch, width, height, n_seq)

            x = self.dcs[i - 1].perform(net['t%d_out' % i], k, m)  # data consistency layer

        return x

    def queue_forward(self, x, k, m, h=None):
        raise NotImplementedError
