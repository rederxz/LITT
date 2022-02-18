import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model import DataConsistencyInKspace
import matplotlib.pyplot as plt


def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128

    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) \
                   + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)

        x = x.transpose(newshape)

    x = np.ascontiguousarray(x).view(dtype=ctype)
    return x.reshape(x.shape[:-1])


def c2r(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x


def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def to_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 2, 3, 1))
    elif x.ndim == 5:  # n 4D inputs. reorder axes
        x = np.transpose(x, (0, 3, 4, 1, 2))

    if mask:  # Hacky solution
        x = x*(1+1j)

    x = c2r(x)

    return x


def from_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n, 2, nx, ny[, nt]).
    Reshapes to (n, [nt, ]nx, ny)
    """

    if x.ndim == 5:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 1, 4, 2, 3))
    elif x.ndim == 6:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 1, 4, 5, 2, 3))
    if mask:
        x = mask_r2c(x)
    else:
        x = r2c(x)

    return x


def phase_diff(img0, img1):
    # img0 give basic phase and img1_diff minus the phase
    a = img0[:, 0:1, :, :, :]
    b = img0[:, 1:2, :, :, :]
    img1_diff_a = a*img1[:, 0:1, :, :, :] + b*img1[:, 1:2, :, :, :]
    img1_diff_b = a*img1[:, 1:2, :, :, :] - b*img1[:, 0:1, :, :, :]

    img1_diff = torch.cat((img1_diff_a, img1_diff_b), 1)
    # tmp0 = from_tensor_format(img0.data.cpu().numpy())
    # tmp1 = from_tensor_format(img1.data.cpu().numpy())
    # tmp2 = from_tensor_format(img1_diff.data.cpu().numpy())
    # im0 = np.angle(np.squeeze(np.concatenate([tmp0[0, 0, :, :], tmp1[0, 0, :, :], tmp2[0, 0, :, :]], 1)))
    # plt.imshow(im0)
    # plt.show()
    # im0 = np.abs(np.squeeze(np.concatenate([tmp0[0, 0, :, :], tmp1[0, 0, :, :], tmp2[0, 0, :, :]], 1)))
    # plt.imshow(im0)
    # plt.show()

    return img1_diff


def phase_add(img0, img1):
    a = img0[:, 0:1, :, :, :]
    b = img0[:, 1:2, :, :, :]
    img1_diff_a = a*img1[:, 0:1, :, :, :] - b*img1[:, 1:2, :, :, :]
    img1_diff_b = a*img1[:, 1:2, :, :, :] + b*img1[:, 0:1, :, :, :]

    img1_diff = torch.cat((img1_diff_a, img1_diff_b), 1)
    # tmp0 = from_tensor_format(img0.data.cpu().numpy())
    # tmp1 = from_tensor_format(img1.data.cpu().numpy())
    # tmp2 = from_tensor_format(img1_diff.data.cpu().numpy())
    # im0 = np.angle(np.squeeze(np.concatenate([tmp0[0, 0, :, :], tmp1[0, 0, :, :], tmp2[0, 0, :, :]], 1)))
    # plt.imshow(im0)
    # plt.show()
    # im0 = np.abs(np.squeeze(np.concatenate([tmp0[0, 0, :, :], tmp1[0, 0, :, :], tmp2[0, 0, :, :]], 1)))
    # plt.imshow(im0)
    # plt.show()
    return img1_diff



class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size, multi_hidden_t=1):
        super(CRNNcell, self).__init__()
        self.multi_hidden_t = multi_hidden_t
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h_0 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        if multi_hidden_t>1:
            self.h2h_1 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
            if multi_hidden_t > 2:
                self.h2h_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
                if multi_hidden_t > 3:
                    self.h2h_3 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
                    if multi_hidden_t > 4:
                        self.h2h_4 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
                        if multi_hidden_t > 5:
                            self.h2h_5 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)

        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        ih_to_ih = self.ih2ih(hidden_iteration)
        if self.multi_hidden_t==1:
            hid_to_hid = self.h2h_0(hidden)
            hidden_out = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        else:
            hidden_out = in_to_hid + ih_to_ih
            for t_hidden in range(self.multi_hidden_t):
                hidden_temp = hidden[...,t_hidden]
                if t_hidden==0:
                    h2h_temp = self.h2h_0
                elif t_hidden==1:
                    h2h_temp = self.h2h_1
                elif t_hidden==2:
                    h2h_temp = self.h2h_2
                elif t_hidden==3:
                    h2h_temp = self.h2h_3
                elif t_hidden==4:
                    h2h_temp = self.h2h_4
                elif t_hidden==5:
                    h2h_temp = self.h2h_5
                hidden_out = hidden_out + h2h_temp(hidden_temp)

            hidden_out = self.relu(hidden_out)

        return hidden_out

class CRNNcell_ME(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell_ME, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv3d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # add iteration hidden connection
        self.ih2ih = nn.Conv3d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)
        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden

class UniCRNNlayer(nn.Module):
    """
    Unidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(UniCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False, useCPU=False):
        ## jieying 20211017 add useCPU
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []

        # forward
        hidden = hid_init
        for i in range(nt):  #past time frame
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)
        output_f = torch.cat(output_f)

        output = output_f

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny) #fill in a new size

        return output

class UniCRNNlayer_ME(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(UniCRNNlayer_ME, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False, useCPU=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        # forward
        hidden = hid_init
        for i in range(nt):  # past time frame
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        output = output_f
        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output

class UniCRNNlayer_frame1by1(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size,multi_hidden_t=1):
        super(UniCRNNlayer_frame1by1, self).__init__()
        self.multi_hidden_t = multi_hidden_t
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size,self.multi_hidden_t)

    def forward(self, input, input_iteration,hid_init_ip, first_frame=False, test=False, useCPU=False):
        # hid_init_ip=None if first_frame==true
        # nt should be 1
        nb, nc, nx, ny = input.shape

        if first_frame:
            if self.multi_hidden_t==1:
                size_h = [nb, self.hidden_size, nx, ny]
            else:
                size_h = [nb, self.hidden_size, nx, ny, self.multi_hidden_t]
            hid_init = torch.zeros(size_h)
        else:
            hid_init = hid_init_ip
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(hid_init).to(torch.device('cpu'))
                else:
                    hid_init = Variable(hid_init).cuda()
        else:
            if useCPU:
                hid_init = Variable(hid_init).to(torch.device('cpu'))
            else:
                hid_init = Variable(hid_init).cuda()
        # forward
        # for i in range(nt):  # past time frame
        hidden = self.CRNN_model(input, input_iteration, hid_init)

        if self.multi_hidden_t==1:
            output = hidden
            output = output.view(nb, self.hidden_size, nx, ny)
        else:
            output=[]
            for t_hidden in range(self.multi_hidden_t-1):
                output.append(hid_init[...,t_hidden+1].unsqueeze(dim=4))
            output.append(hidden.view(nb, self.hidden_size, nx, ny).unsqueeze(dim=4))
            output = torch.cat(output,dim=4)

        return output

class UniCRNNlayer_ME_frame1by1(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size,multi_hidden_t=1):
        super(UniCRNNlayer_ME_frame1by1, self).__init__()
        self.multi_hidden_t = multi_hidden_t
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size,self.multi_hidden_t)

    def forward(self, input, input_iteration, hid_init_ip, first_frame=False, test=False, useCPU=False):
        # hid_init_ip=None if first_frame==true
        # nt should be 1
        nb, nc, nx, ny = input.shape
        if first_frame:
            if self.multi_hidden_t == 1:
                size_h = [nb, self.hidden_size, nx, ny]
            else:
                size_h = [nb, self.hidden_size, nx, ny, self.multi_hidden_t]
            hid_init = torch.zeros(size_h)
        else:
            hid_init = hid_init_ip
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(hid_init).to(torch.device('cpu'))
                else:
                    hid_init = Variable(hid_init).cuda()
        else:
            if useCPU:
                hid_init = Variable(hid_init).to(torch.device('cpu'))
            else:
                hid_init = Variable(hid_init).cuda()

        # forward
        # for i in range(nt):  # past time frame
        hidden = self.CRNN_model(input, input_iteration, hid_init)

        if self.multi_hidden_t==1:
            output = hidden
            output = output.view(nb, self.hidden_size, nx, ny)
        else:
            output=[]
            for t_hidden in range(self.multi_hidden_t-1):
                output.append(hid_init[...,t_hidden+1].unsqueeze(dim=4))
            output.append(hidden.view(nb, self.hidden_size, nx, ny).unsqueeze(dim=4))
            output = torch.cat(output,dim=4)

        return output

class CRNN_MRI_UniDir(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        """
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
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, test=False,useCPU=False):
        ## jieying 20211017 add useCPU
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq*n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        for i in range(1,self.nc+1): #i: number of iteration

            x = x.permute(4,0,1,2,3) #(n_seq, batch, n_ch, width, height)
            x = x.contiguous()
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_seq, n_batch,self.nf,width, height)
            net['t%d_x0'%i] = self.unicrnn(x, net['t%d_x0'%(i-1)], test, useCPU)
            net['t%d_x0'%i] = net['t%d_x0'%i].view(-1,self.nf,width, height) #n_seq*n_batch, self.nf, width, height

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])  #previous iteration
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1,n_ch,width, height)
            net['t%d_out'%i] = x + net['t%d_x4'%i]  #residual #n_seq*n_batch, self.nf, width, height

            net['t%d_out'%i] = net['t%d_out'%i].view(-1,n_batch, n_ch, width, height) #(n_seq, batch_size, n_ch, width, height)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,3,4,0)#(batch_size, n_ch, width, height, n_seq)
            net['t%d_out'%i].contiguous()
            net['t%d_dcs'%i] = self.dcs[i-1].perform(net['t%d_out'%i], k, m)  #data consistency layer
            x = net['t%d_dcs'%i]

            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        return net['t%d_dcs'%i]

class CRNN_MRI_UniDir_ME(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """

    def __init__(self, n_ch=2, nf=64, ks=3, nc=9, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI_UniDir_ME, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.unicrnn = UniCRNNlayer_ME(n_ch, nf, ks)
        self.conv1_x = nn.Conv3d(nf, nf, ks, padding=ks // 2)
        self.conv1_h = nn.Conv3d(nf, nf, ks, padding=ks // 2)
        self.conv2_x = nn.Conv3d(nf, nf, ks, padding=ks // 2)
        self.conv2_h = nn.Conv3d(nf, nf, ks, padding=ks // 2)
        self.conv3_x = nn.Conv3d(nf, nf, ks, padding=ks // 2)
        self.conv3_h = nn.Conv3d(nf, nf, ks, padding=ks // 2)
        self.conv4_x = nn.Conv3d(nf, n_ch, ks, padding=ks // 2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, test=False, useCPU=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq * n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd - 1):
            net['t0_x%d' % j] = hid_init

        for i in range(1, self.nc + 1):  # i: no of iteration

            x = x.permute(5, 0, 1, 2, 3, 4)
            x = x.contiguous()
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_seq, n_batch, self.nf, width, height)
            net['t%d_x0' % i] = self.unicrnn(x, net['t%d_x0' % (i - 1)], test, useCPU)
            net['t%d_x0' % i] = net['t%d_x0' % i].view(-1, self.nf, width, height)

            net['t%d_x1' % i] = self.conv1_x(net['t%d_x0' % i])
            net['t%d_h1' % i] = self.conv1_h(net['t%d_x1' % (i - 1)])
            net['t%d_x1' % i] = self.relu(net['t%d_h1' % i] + net['t%d_x1' % i])

            net['t%d_x2' % i] = self.conv2_x(net['t%d_x1' % i])
            net['t%d_h2' % i] = self.conv2_h(net['t%d_x2' % (i - 1)])
            net['t%d_x2' % i] = self.relu(net['t%d_h2' % i] + net['t%d_x2' % i])

            net['t%d_x3' % i] = self.conv3_x(net['t%d_x2' % i])
            net['t%d_h3' % i] = self.conv3_h(net['t%d_x3' % (i - 1)])
            net['t%d_x3' % i] = self.relu(net['t%d_h3' % i] + net['t%d_x3' % i])

            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            x = x.view(-1, n_ch, width, height)
            net['t%d_out' % i] = x + net['t%d_x4' % i]

            net['t%d_out' % i] = net['t%d_out' % i].view(-1, n_batch, n_ch, width, height)
            net['t%d_out' % i] = net['t%d_out' % i].permute(1, 2, 3, 4, 5, 0)
            net['t%d_out' % i].contiguous()
            net['t%d_out' % i] = self.dcs[i - 1].perform(net['t%d_out' % i], k, m)
            x = net['t%d_out' % i]
            # x_tmp = x.detach().cpu().numpy()
            # plt.imshow(abs(np.concatenate([x_tmp[0, 0, :, :,0], x_tmp[0, 1, :, :,0]], 1)), cmap='gray')
            # plt.show()

            # clean up i-1
            if test:
                to_delete = [key for key in net if ('t%d' % (i - 1)) in key]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        return net['t%d_out' % i]

class CRNN_MRI_UniDir_frame1by1_multiHidden(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5, multi_hidden_t=1):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI_UniDir_frame1by1_multiHidden, self).__init__()
        self.multi_hidden_t = multi_hidden_t
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.unicrnn = UniCRNNlayer_frame1by1(n_ch, nf, ks, self.multi_hidden_t)  # unidirectional
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, hidden, first_frame=False,test=False,useCPU=False):
        net = {}
        n_batch, n_ch, width, height = x.size()
        size_h = [n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        hidden_for_next_frame = []
        for i in range(1,self.nc+1): #i: number of iteration
            x = x.contiguous()
            if first_frame:
                hidden_iter = None
            else:
                hidden_iter = hidden[(i-1)*n_batch:i*n_batch]
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_batch, self.nf, width, height)
            hidden_out = self.unicrnn(x, net['t%d_x0' % (i - 1)], hidden_iter, first_frame, test, useCPU)
            hidden_for_next_frame.append(hidden_out)
            if self.multi_hidden_t==1:
                net['t%d_x0' % i] = hidden_out.view(-1, self.nf, width, height)
            else:
                net['t%d_x0' % i] = hidden_out[...,-1].view(-1, self.nf, width, height)

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])  #previous iteration
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1,n_ch,width, height)
            net['t%d_out'%i] = x + net['t%d_x4'%i]  #residual #n_seq*n_batch, self.nf, width, height

            net['t%d_out' % i].contiguous()
            net['t%d_out' % i] = self.dcs[i - 1].perform(net['t%d_out' % i], k, m, use_echo_dim=False)
            x = net['t%d_out' % i]

            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        hidden_for_next_frame = torch.cat(hidden_for_next_frame)
        return net['t%d_out' % i], hidden_for_next_frame

class CRNN_MRI_UniDir_ME_frame1by1_multiHidden(nn.Module):
    """
    so far the single-echo model is the same as the multi-echo model except for the n_channel.
    just get prepared for the possible modification in the multi-echo model
    """

    def __init__(self, n_ch=2, nf=64, ks=3, nc=9, nd=5, multi_hidden_t=1):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI_UniDir_ME_frame1by1_multiHidden, self).__init__()
        self.multi_hidden_t = multi_hidden_t
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.unicrnn = UniCRNNlayer_ME_frame1by1(n_ch, nf, ks, self.multi_hidden_t)
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

    def forward(self, x, k, m, hidden, first_frame=False,use_echo_dim=False, test=False, useCPU=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height = x.size()
        size_h = [n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd - 1):
            net['t0_x%d' % j] = hid_init

        hidden_for_next_frame = []
        for i in range(1, self.nc + 1):  # i: no of iteration
            x = x.contiguous()
            if first_frame:
                hidden_iter = None
            else:
                hidden_iter = hidden[(i-1)*n_batch:i*n_batch]
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_batch, self.nf, width, height)
            hidden_out = self.unicrnn(x, net['t%d_x0' % (i - 1)], hidden_iter, first_frame, test, useCPU)
            hidden_for_next_frame.append(hidden_out)
            if self.multi_hidden_t==1:
                net['t%d_x0' % i] = hidden_out.view(-1, self.nf, width, height)
            else:
                net['t%d_x0' % i] = hidden_out[...,-1].view(-1, self.nf, width, height)

            net['t%d_x1' % i] = self.conv1_x(net['t%d_x0' % i])
            net['t%d_h1' % i] = self.conv1_h(net['t%d_x1' % (i - 1)])
            net['t%d_x1' % i] = self.relu(net['t%d_h1' % i] + net['t%d_x1' % i])

            net['t%d_x2' % i] = self.conv2_x(net['t%d_x1' % i])
            net['t%d_h2' % i] = self.conv2_h(net['t%d_x2' % (i - 1)])
            net['t%d_x2' % i] = self.relu(net['t%d_h2' % i] + net['t%d_x2' % i])

            net['t%d_x3' % i] = self.conv3_x(net['t%d_x2' % i])
            net['t%d_h3' % i] = self.conv3_h(net['t%d_x3' % (i - 1)])
            net['t%d_x3' % i] = self.relu(net['t%d_h3' % i] + net['t%d_x3' % i])

            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            # x = x.view(-1, n_ch, width, height)
            net['t%d_out' % i] = x + net['t%d_x4' % i]

            net['t%d_out' % i].contiguous()
            net['t%d_out' % i] = self.dcs[i - 1].perform(net['t%d_out' % i], k, m, use_echo_dim)
            x = net['t%d_out' % i]
            # x_tmp = x.detach().cpu().numpy()
            # plt.imshow(abs(np.concatenate([x_tmp[0, 0, :, :,0], x_tmp[0, 1, :, :,0]], 1)), cmap='gray')
            # plt.show()

            # clean up i-1
            if test:
                to_delete = [key for key in net if ('t%d' % (i - 1)) in key]
                for elt in to_delete:
                    del net[elt]
                torch.cuda.empty_cache()

        hidden_for_next_frame = torch.cat(hidden_for_next_frame)
        return net['t%d_out' % i], hidden_for_next_frame

class CRNN_MRI_UniDir_frame1by1(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI_UniDir_frame1by1, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.unicrnn = UniCRNNlayer_frame1by1(n_ch, nf, ks)  # unidirectional
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, hidden, first_frame=False,test=False,useCPU=False):
        net = {}
        n_batch, n_ch, width, height = x.size()
        size_h = [n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        hidden_for_next_frame = []
        for i in range(1,self.nc+1): #i: number of iteration
            x = x.contiguous()
            if first_frame:
                hidden_iter = None
            else:
                hidden_iter = torch.unsqueeze(hidden[i-1],0)
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_batch, self.nf, width, height)
            hidden_out = self.unicrnn(x, net['t%d_x0' % (i - 1)], hidden_iter, first_frame, test, useCPU)
            hidden_for_next_frame.append(hidden_out)
            net['t%d_x0' % i] = hidden_out.view(-1, self.nf, width, height)

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])  #previous iteration
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1,n_ch,width, height)
            net['t%d_out'%i] = x + net['t%d_x4'%i]  #residual #n_seq*n_batch, self.nf, width, height

            net['t%d_out' % i].contiguous()
            net['t%d_out' % i] = self.dcs[i - 1].perform(net['t%d_out' % i], k, m, use_echo_dim=False)
            x = net['t%d_out' % i]

            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        hidden_for_next_frame = torch.cat(hidden_for_next_frame)
        return net['t%d_out' % i], hidden_for_next_frame

class CRNN_MRI_UniDir_ME_frame1by1(nn.Module):
    """
    so far the single-echo model is the same as the multi-echo model except for the n_channel.
    just get prepared for the possible modification in the multi-echo model
    """

    def __init__(self, n_ch=2, nf=64, ks=3, nc=9, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI_UniDir_ME_frame1by1, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.unicrnn = UniCRNNlayer_ME_frame1by1(n_ch, nf, ks)
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

    def forward(self, x, k, m, hidden, first_frame=False,use_echo_dim=False, test=False, useCPU=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height = x.size()
        size_h = [n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd - 1):
            net['t0_x%d' % j] = hid_init

        hidden_for_next_frame = []
        for i in range(1, self.nc + 1):  # i: no of iteration
            x = x.contiguous()
            if first_frame:
                hidden_iter = None
            else:
                hidden_iter = torch.unsqueeze(hidden[i-1],0)
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_batch, self.nf, width, height)
            hidden_out = self.unicrnn(x, net['t%d_x0' % (i - 1)], hidden_iter, first_frame, test, useCPU)
            hidden_for_next_frame.append(hidden_out)
            net['t%d_x0' % i] = hidden_out.view(-1, self.nf, width, height)

            net['t%d_x1' % i] = self.conv1_x(net['t%d_x0' % i])
            net['t%d_h1' % i] = self.conv1_h(net['t%d_x1' % (i - 1)])
            net['t%d_x1' % i] = self.relu(net['t%d_h1' % i] + net['t%d_x1' % i])

            net['t%d_x2' % i] = self.conv2_x(net['t%d_x1' % i])
            net['t%d_h2' % i] = self.conv2_h(net['t%d_x2' % (i - 1)])
            net['t%d_x2' % i] = self.relu(net['t%d_h2' % i] + net['t%d_x2' % i])

            net['t%d_x3' % i] = self.conv3_x(net['t%d_x2' % i])
            net['t%d_h3' % i] = self.conv3_h(net['t%d_x3' % (i - 1)])
            net['t%d_x3' % i] = self.relu(net['t%d_h3' % i] + net['t%d_x3' % i])

            net['t%d_x4' % i] = self.conv4_x(net['t%d_x3' % i])

            # x = x.view(-1, n_ch, width, height)
            net['t%d_out' % i] = x + net['t%d_x4' % i]

            net['t%d_out' % i].contiguous()
            net['t%d_out' % i] = self.dcs[i - 1].perform(net['t%d_out' % i], k, m, use_echo_dim)
            x = net['t%d_out' % i]
            # x_tmp = x.detach().cpu().numpy()
            # plt.imshow(abs(np.concatenate([x_tmp[0, 0, :, :,0], x_tmp[0, 1, :, :,0]], 1)), cmap='gray')
            # plt.show()

            # clean up i-1
            if test:
                to_delete = [key for key in net if ('t%d' % (i - 1)) in key]
                for elt in to_delete:
                    del net[elt]
                torch.cuda.empty_cache()

        hidden_for_next_frame = torch.cat(hidden_for_next_frame)
        return net['t%d_out' % i], hidden_for_next_frame

class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False, useCPU=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):  #past time frame
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt): #future time frame
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i -1], hidden)

            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output

class CRNN_MRI_diff(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, n_ch=2, nf=64, ks=3, nc=6, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI_diff, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.bcrnn = BCRNNlayer(n_ch, nf, ks)
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, b, test=False, useCPU=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        b   - image contains background phase info
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq*n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                if useCPU:
                    hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
                else:
                    hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            if useCPU:
                hid_init = Variable(torch.zeros(size_h)).to(torch.device('cpu'))
            else:
                hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        for i in range(1,self.nc+1): #i: no of iteration
            # x_tmp = from_tensor_format(x.data.cpu().numpy())
            # plt.imshow(abs(np.concatenate([x_tmp[0, 0, :, :], x_tmp[0, 1, :, :]], 1)), cmap='gray')
            # plt.show()
            # plt.imshow(np.angle(np.concatenate([x_tmp[0, 0, :, :], x_tmp[0, 1, :, :]], 1)), cmap='gray')
            # plt.show()
            x = phase_diff(b, x)
            # x_tmp = from_tensor_format(x.data.cpu().numpy())
            # plt.imshow(abs(np.concatenate([x_tmp[0, 0, :, :], x_tmp[0, 1, :, :]], 1)), cmap='gray')
            # plt.show()
            # plt.imshow(np.angle(np.concatenate([x_tmp[0, 0, :, :], x_tmp[0, 1, :, :]], 1)), cmap='gray')
            # plt.show()

            x = x.permute(4,0,1,2,3)
            x = x.contiguous()
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_seq, n_batch,self.nf,width, height)
            net['t%d_x0'%i] = self.bcrnn(x, net['t%d_x0'%(i-1)], test,useCPU)
            net['t%d_x0'%i] = net['t%d_x0'%i].view(-1,self.nf,width, height)


            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1,n_ch,width, height)
            net['t%d_out'%i] =torch.squeeze(x) + net['t%d_x4'%i]
            x_tmp = from_tensor_format(net['t%d_out'%i].data.cpu().numpy())
            # plt.imshow(abs(np.concatenate([x_tmp[1, :, :], x_tmp[0, :, :]], 1)), cmap='gray')
            # plt.show()
            if test & (i == self.nc):
                phase1 = np.angle(x_tmp[1, :, :])
                phase2 = np.angle(x_tmp[0, :, :])
                phase3 = phase1 - phase2
                plt.imshow(np.concatenate([phase1,phase1,phase3], 1), cmap='gray')
                plt.show()

            x0 = phase_add(b,net['t%d_out'%i].unsqueeze(0).permute(0,2,3,4,1))
            net['t%d_out'%i] = torch.squeeze(x0.permute(4,1,2,3,0))
            # x_tmp = from_tensor_format(net['t%d_out'%i].data.cpu().numpy())
            # plt.imshow(abs(np.concatenate([x_tmp[1, :, :], x_tmp[0, :, :]], 1)), cmap='gray')
            # plt.show()
            # plt.imshow(np.angle(np.concatenate([x_tmp[1, :, :], x_tmp[0, :, :]], 1)), cmap='gray')
            # plt.show()

            net['t%d_out'%i] = net['t%d_out'%i].view(-1,n_batch, n_ch, width, height)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,3,4,0)
            net['t%d_out'%i].contiguous()
            net['t%d_out'%i] = self.dcs[i-1].perform(net['t%d_out'%i], k, m)
            x = net['t%d_out'%i]
            x_tmp = x.detach().cpu().numpy()
            # plt.imshow(abs(np.concatenate([x_tmp[0, 0, :, :, 0], x_tmp[0, 1, :, :, 0]], 1)), cmap='gray')
            # plt.show()

            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        return net['t%d_out'%i]
