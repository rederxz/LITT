from collections import deque

import torch
import torch.nn as nn


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4:  # input is 2D
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5:  # input is 3D
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        x = torch.view_as_complex(x.contiguous())
        k = torch.fft.fft2(x, norm=self.normalized)
        k = torch.view_as_real(k)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        out = torch.view_as_complex(out.contiguous())
        x_res = torch.fft.ifft2(out, norm=self.normalized)
        x_res = torch.view_as_real(x_res)

        if x_res.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x_res.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


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
        self.kernel_size = kernel_size
        self.multi_hidden_t = multi_hidden_t
        # image2hidden conv
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # hidden(from the neighbour frame)2hidden conv
        self.h2h = nn.ModuleList()
        for i in range(multi_hidden_t):
            self.h2h.append(nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2))
        # hidden(from the previous iter)2hidden conv
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_i, hidden_t):
        """
        :param input: the input from the previous layer, with shape [batch_size, input_size, x, y]
        :param hidden_i: the hidden states of the previous iteration, with shape [batch_size, hidden_size, x, y]
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
            hid_to_hid += self.h2h[i](torch.zeros_like(hidden_i) if hidden_t[i] is None else hidden_t[i])

        hidden_out = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden_out


class CRNN_t_i(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, uni_direction=False, multi_hidden_t=1):
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
        self.multi_hidden_t = multi_hidden_t
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size, multi_hidden_t)

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
            output.append(hidden)
        output = torch.stack(output, dim=0)

        if not self.uni_direction:
            # backward
            output_b = []
            hidden_t_pool = deque([None] * self.multi_hidden_t, maxlen=self.multi_hidden_t)
            for i in range(nt):  # future time frame
                hidden = self.CRNN_model(input[nt - i - 1], hidden_i[nt - i - 1], hidden_t_pool)
                hidden_t_pool.append(hidden)
                output_b.append(hidden)
            output_b = torch.stack(output_b[::-1], dim=0)
            output = output + output_b

        return output


class CRNN(nn.Module):
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5, uni_direction=False, multi_hidden_t=1):
        """
        Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
        unidirectional version
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        :param uni_direction: ...
        :param multi_hidden_t: ...
        """
        super(CRNN, self).__init__()
        self.n_ch = n_ch
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks
        self.uni_direction = uni_direction
        self.multi_hidden_t = multi_hidden_t

        self.crnn_t_i = CRNN_t_i(n_ch, nf, ks, uni_direction, multi_hidden_t)
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
        """
        perform prediction one by one, helped by the hidden state of the previous frame
        equivalent to doing forward with T=1 and initial CRNN-i hidden state given
        **only meaningful when unidirectional**
        :param x: input in image domain, [batch_size, 2, width, height, 1]
        :param k: initially sampled elements in k-space, [batch_size, 2, width, height, 1]
        :param m: corresponding nonzero location, [batch_size, 2, width, height, 1]
        :param h: a dict, each value corresponds to each iteration's hidden_t of the previous frame(s), each value
        is an iterable container whose element(s) has shape [batch_size, hidden_size, width, height]
        :return: reconstruction result, [batch_size, 2, width, height, 1]
                hidden states, a dict, each value has shape [batch_size, hidden_size, width, height]
        """
        assert self.uni_direction and x.shape[-1] == 1

        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq * n_batch, self.nf, width, height]
        hid_init = x.new_zeros(size_h)  # the initial zero-valued hidden state (the same device and dtype as x)

        # initialize hidden states
        for j in range(self.nd - 1):  # except for the last vanilla CNN layer, all layers maintain a hidden state
            net['t0_x%d' % j] = hid_init  # 't' means iteration, 'x' means layers

        # initialize a hidden_t pool if not given
        if h is None:
            h = {}
            for i in range(1, self.nc + 1):
                h['t%d_x0' % i] = deque([None] * self.multi_hidden_t, maxlen=self.multi_hidden_t)

        # for convenience
        x = x.view(-1, self.n_ch, width, height)  # [1 * batch, n_ch, width, height]
        k = k.view(-1, self.n_ch, width, height)  # [1 * batch, n_ch, width, height]
        m = m.view(-1, self.n_ch, width, height)  # [1 * batch, n_ch, width, height]

        # iterate
        for i in range(1, self.nc + 1):  # i: number of iteration
            # directly call the CRNN cell
            net['t%d_x0' % i] = self.crnn_t_i.CRNN_model(x, net['t%d_x0' % (i - 1)], h['t%d_x0' % i])
            h['t%d_x0' % i].append(net['t%d_x0' % i])  # update corresponding hidden_t pool

            # 3 layers of CRNN-i
            net['t%d_x1' % i] = self.relu(self.conv1_x(net['t%d_x0' % i]) + self.conv1_h(net['t%d_x1' % (i - 1)]))
            net['t%d_x2' % i] = self.relu(self.conv2_x(net['t%d_x1' % i]) + self.conv2_h(net['t%d_x2' % (i - 1)]))
            net['t%d_x3' % i] = self.relu(self.conv3_x(net['t%d_x2' % i]) + self.conv3_h(net['t%d_x3' % (i - 1)]))

            # 1 layer of vanilla CNN
            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            # shortcut connection
            net['t%d_out' % i] = x + net['t%d_x4' % i]

            x = self.dcs[i - 1].perform(net['t%d_out' % i], k, m)  # data consistency layer

        return x[..., None], h


class resCRNNcell(CRNNcell):
    def __init__(self, input_size, hidden_size, kernel_size, multi_hidden_t=1):
        """
        Just use identity mapping as the hidden2hidden connection.
        """
        super().__init__(input_size, hidden_size, kernel_size, multi_hidden_t)
        self.ih2ih = nn.Identity()


class resCRNN_t_i(CRNN_t_i):
    def __init__(self, input_size, hidden_size, kernel_size, uni_direction=False, multi_hidden_t=1):
        """
        Recurrent Convolutional RNN layer over iterations and time
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param uni_direction: ...
        """
        super(resCRNN_t_i, self).__init__(input_size, hidden_size, kernel_size, uni_direction, multi_hidden_t)
        self.CRNN_model = resCRNNcell(self.input_size, self.hidden_size, self.kernel_size, multi_hidden_t)


class resCRNN(CRNN):
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5, uni_direction=False, multi_hidden_t=1):
        """
        Just use identity mapping as the hidden2hidden connection.
        """
        super().__init__(n_ch, nf, ks, nc, nd, uni_direction, multi_hidden_t)
        self.conv1_h = nn.Identity()
        self.conv2_h = nn.Identity()
        self.conv3_h = nn.Identity()


class CRNNcell_t(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, multi_hidden_t=1):
        """
        Only connect in time dim
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.multi_hidden_t = multi_hidden_t
        # image2hidden conv
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # hidden(from the neighbour frame)2hidden conv
        self.h2h = nn.ModuleList()
        for i in range(multi_hidden_t):
            self.h2h.append(nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_i, hidden_t):
        """
        :param input: the input from the previous layer, with shape [batch_size, input_size, x, y]
        :param hidden_i: the hidden states of the previous iteration, with shape [batch_size, hidden_size, x, y]
        :param hidden_t: torch tensor or an iterable container of torch tensor(s), the hidden states of the neighbour
        frame(s), with shape [batch_size, hidden_size, x, y]
        :return: hidden state with shape [batch_size, hidden_size, width, height]
        """
        in_to_hid = self.i2h(input)

        # make sure the given hidden_t matches self.multi_hidden_t
        if not isinstance(hidden_t, deque):
            hidden_t = deque([hidden_t])
        assert self.multi_hidden_t == len(hidden_t)

        hid_to_hid = torch.zeros_like(hidden_i)
        for i in range(self.multi_hidden_t):
            hid_to_hid += self.h2h[i](torch.zeros_like(hidden_i) if hidden_t[i] is None else hidden_t[i])

        hidden_out = self.relu(in_to_hid + hid_to_hid)

        return hidden_out


class CRNN_t(CRNN_t_i):
    def __init__(self, input_size, hidden_size, kernel_size, uni_direction=False, multi_hidden_t=1):
        super().__init__(input_size, hidden_size, kernel_size, uni_direction, multi_hidden_t)
        self.CRNN_model = CRNNcell_t(self.input_size, self.hidden_size, self.kernel_size, multi_hidden_t)


class CRNN_T(nn.Module):
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5, uni_direction=False, multi_hidden_t=1):
        """
        Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
        unidirectional version
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        :param uni_direction: ...
        :param multi_hidden_t: ...
        """
        super().__init__()
        self.n_ch = n_ch
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks
        self.uni_direction = uni_direction
        self.multi_hidden_t = multi_hidden_t

        self.crnn_t = CRNN_t(n_ch, nf, ks, uni_direction, multi_hidden_t)
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding=ks // 2)
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

            # 1 layer of CRNN-t
            net['t%d_x0' % i] = self.crnn_t(x, net['t%d_x0' % (i - 1)])

            net['t%d_x0' % i] = net['t%d_x0' % i] \
                .view(-1, self.nf, width, height)  # [n_seq * n_batch, self.nf, width, height] as required by following CNN layers

            # 3 layers of CRNN-i
            net['t%d_x1' % i] = self.relu(self.conv1_x(net['t%d_x0' % i]))
            net['t%d_x2' % i] = self.relu(self.conv2_x(net['t%d_x1' % i]))
            net['t%d_x3' % i] = self.relu(self.conv3_x(net['t%d_x2' % i]))

            # 1 layer of vanilla CNN
            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            # shortcut connection
            net['t%d_out' % i] = x + net['t%d_x4' % i].view(n_seq, n_batch, n_ch, width, height)

            net['t%d_out' % i] = net['t%d_out' % i].permute(1, 2, 3, 4, 0)  # (batch_size, n_ch, width, height, n_seq)

            x = self.dcs[i - 1].perform(net['t%d_out' % i], k, m)  # data consistency layer

        return x

    def queue_forward(self, x, k, m, h=None):
        pass

