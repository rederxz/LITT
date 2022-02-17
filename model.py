import numpy as np
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


def get_add_neighbour_op(nc, frame_dist, divide_by_n, clipped):
    max_sample = max(frame_dist) *2 + 1

    # for non-clipping, increase the input circularly
    if clipped:
        padding = (max_sample//2, 0, 0)
    else:
        padding = 0

    # expect data to be in this format: (n, nc, nt, nx, ny) (due to FFT)
    conv = nn.Conv3d(in_channels=nc, out_channels=nc*len(frame_dist),
                     kernel_size=(max_sample, 1, 1),
                     stride=1, padding=padding, bias=False)

    # Although there is only 1 parameter, need to iterate as parameters return generator
    conv.weight.requires_grad = False

    # kernel has size nc=2, nc'=8, kt, kx, ky
    for i, n in enumerate(frame_dist):
        m = max_sample // 2
        #c = 1 / (n * 2 + 1) if divide_by_n else 1
        c = 1
        wt = np.zeros((2, max_sample, 1, 1), dtype=np.float32)
        wt[0, m-n:m+n+1] = c
        wt2 = np.zeros((2, max_sample, 1, 1), dtype=np.float32)
        wt2[1, m-n:m+n+1] = c

        conv.weight.data[2*i] = torch.from_numpy(wt)
        conv.weight.data[2*i+1] = torch.from_numpy(wt2)

    conv.cuda()
    return conv


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = torch.fft(x, 2, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.ifft(out, 2, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


class KspaceFillNeighbourLayer(nn.Module):
    '''
    k-space fill layer - The input data is assumed to be in k-space grid.

    The input data is assumed to be in k-space grid.
    This layer should be invoked from AverageInKspaceLayer
    '''
    def __init__(self, frame_dist, divide_by_n=False, clipped=True, **kwargs):
        # frame_dist is the extent that data sharing goes.
        # e.g. current frame is 3, frame_dist = 2, then 1,2, and 4,5 are added for reconstructing 3
        super(KspaceFillNeighbourLayer, self).__init__()
        print("fr_d={}, divide_by_n={}, clippd={}".format(frame_dist, divide_by_n, clipped))
        if 0 not in frame_dist:
            raise ValueError("There suppose to be a 0 in fr_d in config file!")
            frame_dist = [0] + frame_dist # include ID

        self.frame_dist  = frame_dist
        self.n_samples   = [1 + 2*i for i in self.frame_dist]
        self.divide_by_n = divide_by_n
        self.clipped     = clipped
        self.op = get_add_neighbour_op(2, frame_dist, divide_by_n, clipped)

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, k, mask):
        '''

        Parameters
        ------------------------------
        inputs: two 5d tensors, [kspace_data, mask], each of shape (n, 2, NT, nx, ny)

        Returns
        ------------------------------
        output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
        shape becomes (n* (len(frame_dist), 2, nt, nx, ny)
        '''
        max_d = max(self.frame_dist)
        k_orig = k
        mask_orig = mask
        if not self.clipped:
            # pad input along nt direction, which is circular boundary condition. Otherwise, just pad outside
            # places with 0 (zero-boundary condition)
            k = torch.cat([k[:,:,-max_d:], k, k[:,:,:max_d]], 2)
            mask = torch.cat([mask[:,:,-max_d:], mask, mask[:,:,:max_d]], 2)

        # start with x, then copy over accumulatedly...
        res = self.op(k)
        if not self.divide_by_n:
            # divide by n basically means for each kspace location, if n non-zero values from neighboring
            # time frames contributes to it, then divide this entry by n (like a normalization)
            res_mask = self.op(mask)
            res = res / res_mask.clamp(min=1)
        else:
            res_mask = self.op(torch.ones_like(mask))
            res = res / res_mask.clamp(min=1)

        res = data_consistency(res,
                               k_orig.repeat(1,len(self.frame_dist),1,1,1),
                               mask_orig.repeat(1,len(self.frame_dist),1,1,1))

        nb, nc_ri, nt, nx, ny = res.shape # here ri_nc is complicated with data sharing replica and real-img dimension
        res = res.reshape(nb, nc_ri//2, 2, nt, nx, ny)
        return res


class AveragingInKspace(nn.Module):
    '''
    Average-in-k-space layer

    First transforms the representation in Fourier domain,
    then performs averaging along temporal axis, then transforms back to image
    domain. Works only for 5D tensor (see parameter descriptions).


    Parameters
    -----------------------------
    incomings: two 5d tensors, [kspace_data, mask], each of shape (n, 2, nx, ny, nt)

    data_shape: shape of the incoming tensors: (n, 2, nx, ny, nt) (This is for convenience)

    frame_dist: a list of distances of neighbours to sample for each averaging channel
        if frame_dist=[1], samples from [-1, 1] for each temporal frames
        if frame_dist=[3, 5], samples from [-3,-2,...,0,1,...,3] for one,
                                           [-5,-4,...,0,1,...,5] for the second one

    divide_by_n: bool - Decides how averaging will be done.
        True => divide by number of neighbours (=#2*frame_dist+1)
        False => divide by number of nonzero contributions

    clipped: bool - By default the layer assumes periodic boundary condition along temporal axis.
        True => Averaging will be clipped at the boundary, no circular references.
        False => Averages with circular referencing (i.e. at t=0, gets contribution from t=nt-1, so on).

    Returns
    ------------------------------
    output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
            shape becomes (n, (len(frame_dist))* 2, nx, ny, nt)
    '''

    def __init__(self, frame_dist, divide_by_n=False, clipped=True, norm='ortho'):
        super(AveragingInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.frame_dist = frame_dist
        self.divide_by_n = divide_by_n
        self.kavg = KspaceFillNeighbourLayer(frame_dist, divide_by_n, clipped)

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, mask):
        """
        x    - input in image space, shape (n, 2, nx, ny, nt)
        mask - corresponding nonzero location
        """
        mask = mask.permute(0, 1, 4, 2, 3)

        x = x.permute(0, 4, 2, 3, 1) # put t to front, in convenience for fft
        k = torch.fft(x, 2, normalized=self.normalized)
        k = k.permute(0, 4, 1, 2, 3) # then put ri to the front, then t

        # data sharing
        # nc is the numpy of copies of kspace, specified by frame_dist
        out = self.kavg.perform(k, mask)
        # after datasharing, it is nb, nc, 2, nt, nx, ny

        nb, nc, _, nt, nx, ny = out.shape # , jo's version

        # out.shape: [nb, 2*len(frame_dist), nt, nx, ny]
        # we then detatch confused real/img channel and replica kspace channel due to datasharing (nc)
        out = out.permute(0,1,3,4,5,2) # jo version, split ri and nc, put ri to the back for ifft
        x_res = torch.ifft(out, 2, normalized=self.normalized)


        # now nb, nc, nt, nx, ny, ri, put ri to channel position, and after nc (i.e. within each nc)
        x_res = x_res.permute(0,1,5,3,4,2).reshape(nb, nc*2, nx,ny, nt)# jo version

        return x_res


class CRNNcell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        """
        Convolutional RNN cell that evolves over both time and iterations
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        :param multi_hidden_t: whether use multi hidden states
        """
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        # image2hidden conv
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # hidden2hidden conv
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # hidden(from the previous iter)2hidden conv
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        """
        :param input: the input from the previous layer
        :param hidden_iteration: the hidden states of the previous iteration
        :param hidden: the hidden states of the previous frame or the next frame
        :return: hidden state with shape [batch_size, hidden_size, width, height]
        """
        in_to_hid = self.i2h(input)
        ih_to_ih = self.ih2ih(hidden_iteration)
        hid_to_hid = self.h2h_0(hidden)
        hidden_out = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden_out


class UniCRNNlayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        """
        Unidirectional Convolutional RNN layer (only forward direction)
        :param input_size: channels of inputs
        :param hidden_size: channels of hidden states
        :param kernel_size: the kernel size of CNN
        """
        super(UniCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration):
        """
        :param input: the input from the previous layer, [num_seqs, batch_size, channel, width, height]
        :param input_iteration: the hidden state from the previous iteration, [num_seqs, batch_size, hidden_size, width, height]
        :return: hidden state, [num_seqs, batch_size, hidden_size, width, height]
        """
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = torch.zeros(size_h)  # the initial zero-valued hidden state

        # forward
        output_f = []
        hidden = hid_init
        for i in range(nt):  # past time frame
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden[None, ...])
        output = torch.cat(output_f, dim=0)

        return output


class CRNN_MRI_UniDir(nn.Module):
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        """
        Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
        unidirectional version
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI_UniDir, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.unicrnn = UniCRNNlayer(n_ch, nf, ks)  # unidirectional
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
        hid_init = torch.zeros(size_h)  # the initial zero-valued hidden state

        # initialize hidden states
        for j in range(self.nd - 1):  # except for the last vanilla CNN layer, all layers maintain a hidden state
            net['t0_x%d' % j] = hid_init  # 't' means iteration here

        # iterate
        for i in range(1, self.nc + 1):  # i: number of iteration
            x = x.permute(4, 0, 1, 2, 3)  # [n_seq, batch, n_ch, width, height]
            # x = x.contiguous()  # TODO

            # 1 layer of uniCRNN-t-i
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)] \
                .view(n_seq, n_batch, self.nf, width, height)  # [n_seq, n_batch, self.nf, width, height]
            net['t%d_x0' % i] = self.unicrnn(x, net['t%d_x0' % (i - 1)])
            net['t%d_x0' % i] = net['t%d_x0' % i] \
                .view(-1, self.nf, width, height)  # [n_seq * n_batch, self.nf, width, height]

            # 3 layers of CRNN-i
            net['t%d_x1' % i] = self.relu(self.conv1_x(net['t%d_x0' % i]) + self.conv1_h(net['t%d_x1' % (i - 1)]))
            net['t%d_x2' % i] = self.relu(self.conv2_x(net['t%d_x1' % i]) + self.conv2_h(net['t%d_x2' % (i - 1)]))
            net['t%d_x3' % i] = self.relu(self.conv3_x(net['t%d_x2' % i]) + self.conv3_h(net['t%d_x3' % (i - 1)]))

            # 1 layer of vanilla CNN
            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            # shortcut connection
            net['t%d_out' % i] = x + net['t%d_x4' % i]\
                .view(n_seq, n_batch, self.nf, width, height)

            net['t%d_out' % i] = net['t%d_out' % i].permute(1, 2, 3, 4, 0)  # (batch_size, n_ch, width, height, n_seq)
            # net['t%d_out' % i].contiguous()  # TODO

            x = self.dcs[i - 1].perform(net['t%d_out' % i], k, m)  # data consistency layer

            # # clean up i-1
            # if test:
            #     to_delete = [key for key in net if ('t%d' % (i - 1)) in key]
            #     for elt in to_delete:
            #         del net[elt]
            #     torch.cuda.empty_cache()

        return x

    def forward_1by1(self, x, k, m, h):
        """
        perform predict one by one, helped by the hidden state of the previous frame
        equivalent to doing forward with T=1 and initial CRNN-i hidden state given
        :param x: input in image domain, [batch_size, 2, width, height, 1]
        :param k: initially sampled elements in k-space, [batch_size, 2, width, height, 1]
        :param m: corresponding nonzero location, [batch_size, 2, width, height, 1]
        :param h: hidden states over each iteration of the previous frame
        :return: reconstruction result, [batch_size, 2, width, height, 1]
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq * n_batch, self.nf, width, height]
        hid_init = torch.zeros(size_h)  # the initial zero-valued hidden state

        # initialize hidden states
        for j in range(self.nd - 1):  # except for the last vanilla CNN layer, all layers maintain a hidden state
            net['t0_x%d' % j] = hid_init  # 't' means iteration, 'x' means layers

        # iterate
        for i in range(1, self.nc + 1):  # i: number of iteration
            x = x.permute(4, 0, 1, 2, 3)  # [n_seq, batch, n_ch, width, height]
            # x = x.contiguous()  # TODO

            # 1 layer of uniCRNN-t-i
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)] \
                .view(n_seq, n_batch, self.nf, width, height)  # [n_seq, n_batch, self.nf, width, height]
            net['t%d_x0' % i] = self.unicrnn.CRNN_model(x, net['t%d_x0' % (i - 1)], h)  # directly call the CRNN cell
            net['t%d_x0' % i] = net['t%d_x0' % i] \
                .view(-1, self.nf, width, height)  # [n_seq * n_batch, self.nf, width, height]

            # 3 layers of CRNN-i
            net['t%d_x1' % i] = self.relu(self.conv1_x(net['t%d_x0' % i]) + self.conv1_h(net['t%d_x1' % (i - 1)]))
            net['t%d_x2' % i] = self.relu(self.conv2_x(net['t%d_x1' % i]) + self.conv2_h(net['t%d_x2' % (i - 1)]))
            net['t%d_x3' % i] = self.relu(self.conv3_x(net['t%d_x2' % i]) + self.conv3_h(net['t%d_x3' % (i - 1)]))

            # 1 layer of vanilla CNN
            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            # shortcut connection
            net['t%d_out' % i] = x + net['t%d_x4' % i]\
                .view(n_seq, n_batch, self.nf, width, height)

            net['t%d_out' % i] = net['t%d_out' % i].permute(1, 2, 3, 4, 0)  # (batch_size, n_ch, width, height, n_seq)
            # net['t%d_out' % i].contiguous()  # TODO

            x = self.dcs[i - 1].perform(net['t%d_out' % i], k, m)  # data consistency layer

        return x
