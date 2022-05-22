import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_crnn import DataConsistencyInKspace, k2i


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, n_f=64, k_s=3, dilation=1):
        super(ResidualBlock_noBN, self).__init__()
        self.conv_1 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.conv_2 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.conv_2(out)
        return x + out


class ResidualBlock_SiLU_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, n_f=64, k_s=3, dilation=1):
        super(ResidualBlock_SiLU_noBN, self).__init__()
        self.conv_1 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.conv_2 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.silu(out)
        out = self.conv_2(out)
        return x + out


class ProductBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, n_f=64, k_s=3, dilation=1):
        super(ProductBlock_noBN, self).__init__()
        self.conv_1 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.conv_2 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.conv_2(out)
        return x * out


class ProductBlock_noBN_v2(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, n_f=64, k_s=3, dilation=1):
        super(ProductBlock_noBN_v2, self).__init__()
        self.conv_1 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.conv_2 = nn.Conv2d(n_f, n_f, k_s, dilation=dilation, padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.sig(out)
        return x * out


class ConvBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, n_f=64, k_s=3):
        super(ConvBlock_noBN, self).__init__()
        self.conv_1 = nn.Conv2d(n_f, n_f, k_s, padding='same')
        self.conv_2 = nn.Conv2d(n_f, n_f, k_s, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.conv_2(out)
        return out


class ESA(nn.Module):

    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4

        def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
            if not padding and stride == 1:
                padding = kernel_size // 2
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                             groups=groups)

        conv = default_conv

        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, f):
        c1_ = (self.conv1(f))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RLFB(nn.Module):
    def  __init__(self, n_f=64):
        super(RLFB, self).__init__()

        self.c1 = nn.Conv2d(n_f, n_f, 3, padding='same')
        self.c2 = nn.Conv2d(n_f, n_f, 3, padding='same')
        self.c3 = nn.Conv2d(n_f, n_f, 3, padding='same')

        self.fusion = nn.Conv2d(n_f, n_f, 1)

        self.esa = ESA(n_f)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.c1(x))
        out = self.act(self.c2(out))
        out = self.act(self.c3(out))
        out = x + out

        out = self.fusion(out)
        out = self.esa(out, out)

        return out


def make_layer(block_func, n_layers):
    layers = list()
    for _ in range(n_layers):
        layers.append(block_func())
    return nn.Sequential(*layers)


class RRN(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        self.conv_1 = nn.Conv2d(n_ch * 2 + n_h + n_ch, n_h, 3, padding='same')
        self.residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.relu = nn.ReLU(inplace=True)

        self.dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        network_input = torch.cat([x, x_l, h, o], dim=1)
        hidden = self.conv_1(network_input)
        hidden = self.relu(hidden)
        hidden = self.residual_blocks(hidden)
        output_h = self.conv_h(hidden)
        output_h = self.relu(output_h)
        output_o = self.conv_o(hidden)
        output_o = self.dc(output_o, k, m)

        return output_o, output_h


class RRN_two_stage(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_RLFB(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=2):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_RLFB, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv_1 = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s1_rlfb = make_layer(lambda: RLFB(n_f=n_h), n_blocks)
        self.s1_conv_2 = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s1_conv_3 = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv_1 = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s2_rlfb = make_layer(lambda: RLFB(n_f=n_h), n_blocks)
        self.s2_conv_2 = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s2_conv_3 = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.s1_conv_1(stage_1_input)
        hidden = self.s1_conv_2(self.s1_rlfb(hidden)) + hidden
        hidden = self.s1_conv_3(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.s2_conv_1(stage_2_input)
        hidden = self.s2_conv_2(self.s2_rlfb(hidden)) + hidden
        hidden = self.s2_conv_3(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_silu(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_silu, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_SiLU_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_SiLU_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.silu = nn.SiLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.silu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.silu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.silu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_tk(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_tk, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, o=None, tk=None):
        if o is None:
            o = torch.zeros_like(x)
        if tk is None:
            tk = torch.zeros_like(k)

        tk = k * m + tk * (1 - m)
        x_tk = k2i(tk)

        stage_1_input = torch.cat([x, x_tk], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)

        return output_o, tk


class RRN_two_stage_tk_res(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_tk_res, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, o=None, tk=None):
        if o is None:
            o = torch.zeros_like(x)
        if tk is None:
            tk = torch.zeros_like(k)

        tk = k * m + tk * (1 - m)
        x_tk = k2i(tk)

        stage_1_input = torch.cat([x, x_tk], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden) + x_tk, k, m)

        return output_o, tk


def low_res_diff(k, k_ref, center=8, centred=False):
    """
    Args:
        k: [batch_size, 2, width, height]
        k_ref: [batch_size, 2, width, height]
        center: center lines

    Returns:

    """
    mask = torch.zeros_like(k)
    mask[..., k.shape[-2] // 2 - center // 2: k.shape[-2] // 2 + center // 2, :] = 1
    if not centred:
        mask = torch.fft.ifftshift(mask, dim=(-1, -2))

    lr_k = torch.view_as_complex((mask * k).permute(0, 2, 3, 1).contiguous())
    lr_k_ref = torch.view_as_complex((mask * k_ref).permute(0, 2, 3, 1).contiguous())

    lr_img = torch.fft.ifft2(lr_k, norm='ortho')
    lr_img_ref = torch.fft.ifft2(lr_k_ref, norm='ortho')

    diff = torch.view_as_real(lr_img - lr_img_ref).permute(0, 3, 1, 2)

    return diff


class RRN_two_stage_diff_guided(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5, center_lines=8):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_diff_guided, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        self.low_res_diff = lambda k, k_ref: low_res_diff(k, k_ref, center=center_lines)

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, k_ref, h=None, o=None):

        n_b, n_ch, width, height = x.shape
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        diff = self.low_res_diff(k, k_ref)

        stage_1_input = torch.cat([x, diff, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_product_block(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_product_block, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_product_blocks = make_layer(lambda: ProductBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_product_blocks = make_layer(lambda: ProductBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_product_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_product_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_product_block_v2(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_product_block_v2, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_product_blocks = make_layer(lambda: ProductBlock_noBN_v2(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_product_blocks = make_layer(lambda: ProductBlock_noBN_v2(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_product_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_product_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_dilation(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, dilation=1, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_dilation, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, dilation=dilation, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s, dilation=dilation), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, dilation=dilation, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, dilation=dilation, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s, dilation=dilation), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, dilation=dilation, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, dilation=dilation, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_post_relu(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_post_relu, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.relu(self.s1_residual_blocks(hidden))
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.relu(self.s2_residual_blocks(hidden))
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_only_hidden_relay(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_only_hidden_relay, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, h=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])

        stage_1_input = torch.cat([x, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_only_rec_relay(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_only_rec_relay, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, o], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)

        return output_o


class RRN_two_stage_res(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_res, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden) + o, k, m)  # predict the residual
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_only_outer_res(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_only_outer_res, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_conv_blocks = make_layer(lambda: ConvBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_conv_blocks = make_layer(lambda: ConvBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_conv_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden) + x, k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_conv_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden) + stage_1_output, k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_v2(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_v2, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s1_conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output_o = self.s1_dc(self.s1_conv_o(hidden), k, m)
        stage_1_output_h = self.relu(self.s1_conv_h(hidden))

        stage_2_input = torch.cat([stage_1_output_o, o, stage_1_output_h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_two_stage_v3(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_v3, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, hidden], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class Temporal_Feature_Aggregation(nn.Module):
    def __init__(self, weight):
        super(Temporal_Feature_Aggregation, self).__init__()
        self.weight = weight

    def forward(self, cur_feat, last_feat):
        coherence = abs(torch.nn.functional.cosine_similarity(cur_feat, last_feat)[:, None])

        feat = cur_feat + self.weight * coherence * (last_feat - cur_feat)

        return feat


class RRN_two_stage_tfa(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5, weight=0.5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_two_stage_tfa, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        self.tfa = Temporal_Feature_Aggregation(weight=0.5)

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s2_conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output_o = self.s1_dc(self.s1_conv_o(hidden), k, m)

        hidden = self.tfa(hidden, h)

        stage_2_input = torch.cat([stage_1_output_o, o, hidden], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h


class RRN_multi_stage(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_multi_stage, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        # stage 3
        self.s3_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s3_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s3_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s3_dc = DataConsistencyInKspace(norm='ortho')

        # stage 4
        self.s4_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s4_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s4_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s4_dc = DataConsistencyInKspace(norm='ortho')

        # stage 5
        self.s5_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s5_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s5_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s5_conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s5_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, h], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        stage_2_output = self.s2_dc(self.s2_conv_o(hidden), k, m)

        stage_3_input = torch.cat([stage_2_output, o, h], dim=1)
        hidden = self.relu(self.s3_conv(stage_3_input))
        hidden = self.s3_residual_blocks(hidden)
        stage_3_output = self.s3_dc(self.s3_conv_o(hidden), k, m)

        stage_4_input = torch.cat([stage_3_output, o, h], dim=1)
        hidden = self.relu(self.s4_conv(stage_4_input))
        hidden = self.s4_residual_blocks(hidden)
        stage_4_output = self.s4_dc(self.s4_conv_o(hidden), k, m)

        stage_5_input = torch.cat([stage_4_output, o, h], dim=1)
        hidden = self.relu(self.s5_conv(stage_5_input))
        hidden = self.s5_residual_blocks(hidden)
        output_o = self.s5_dc(self.s5_conv_o(hidden), k, m)
        output_h = self.relu(self.s5_conv_h(hidden))

        return output_o, output_h


class RRN_multi_stage_v3(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_multi_stage_v3, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        # stage 3
        self.s3_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s3_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s3_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s3_dc = DataConsistencyInKspace(norm='ortho')

        # stage 4
        self.s4_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s4_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s4_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s4_dc = DataConsistencyInKspace(norm='ortho')

        # stage 5
        self.s5_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, 3, padding='same')
        self.s5_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s5_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s5_conv_h = nn.Conv2d(n_h, n_h, 3, padding='same')
        self.s5_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.relu(self.s1_conv(stage_1_input))
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, hidden], dim=1)
        hidden = self.relu(self.s2_conv(stage_2_input))
        hidden = self.s2_residual_blocks(hidden)
        stage_2_output = self.s2_dc(self.s2_conv_o(hidden), k, m)

        stage_3_input = torch.cat([stage_2_output, o, hidden], dim=1)
        hidden = self.relu(self.s3_conv(stage_3_input))
        hidden = self.s3_residual_blocks(hidden)
        stage_3_output = self.s3_dc(self.s3_conv_o(hidden), k, m)

        stage_4_input = torch.cat([stage_3_output, o, hidden], dim=1)
        hidden = self.relu(self.s4_conv(stage_4_input))
        hidden = self.s4_residual_blocks(hidden)
        stage_4_output = self.s4_dc(self.s4_conv_o(hidden), k, m)

        stage_5_input = torch.cat([stage_4_output, o, hidden], dim=1)
        hidden = self.relu(self.s5_conv(stage_5_input))
        hidden = self.s5_residual_blocks(hidden)
        output_o = self.s5_dc(self.s5_conv_o(hidden), k, m)
        output_h = self.relu(self.s5_conv_h(hidden))

        return output_o, output_h


class RRN_relay(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_relay, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        self.conv_1 = nn.Conv2d(n_ch, n_h, 3, padding='same')
        self.residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.relu = nn.ReLU(inplace=True)

        self.relay_dc = DataConsistencyInKspace(norm='ortho')
        self.dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, o=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        o = self.relay_dc(o, k, m)  # relay step

        hidden = self.relu(self.conv_1(o))
        hidden = self.residual_blocks(hidden)
        output_o = self.dc(self.conv_o(hidden), k, m)

        return output_o


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionFuse(nn.Module):
    def __init__(self, n_h):
        super(AttentionFuse, self).__init__()

        self.ca = ChannelAttention(n_h * 2)
        self.sa = SpatialAttention()

    def forward(self, hidden, last_hidden):
        union = torch.cat([hidden, last_hidden], dim=1)
        c_att = self.ca(union)
        union = c_att * union
        s_att = self.sa(union)
        hidden = hidden * (1 - s_att) + last_hidden * s_att

        return hidden


class RRN_att_fuse(nn.Module):
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
        """
        Args:
            n_ch: input channel
            n_h: hidden size
        """
        super(RRN_att_fuse, self).__init__()
        self.n_ch = n_ch
        self.n_h = n_h
        self.n_blocks = n_blocks

        # feature extraction stage
        self.s1_conv = nn.Conv2d(n_ch, n_h, 3, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)

        # att module
        self.att_fuse = AttentionFuse(n_h)

        # reconstruction stage
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, 3, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k, m, h=None):
        """
        Args:
            x: the aliased image, the current and the last frame [batch_size, 2, width, height]
            k: initially sampled elements in k-space, [batch_size, 2, width, height]
            m: corresponding nonzero location, [batch_size, 2, width, height]
            x_l: the last aliased image, the current and the last frame [batch_size, 2, width, height]
            h: hidden from the last frame,  [batch_size, hidden_size, width, height]
            o: reconstruction result of the last frame, [batch_size, 2, width, height]

        Returns:
            reconstruction result, [batch_size, 2, width, height]
            output_hidden, [batch_size, hidden_size, width, height]

        """
        n_b, n_ch, width, height = x.shape
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])

        hidden = self.relu(self.s1_conv(x))
        hidden = self.s1_residual_blocks(hidden)

        hidden_fuse = self.att_fuse(hidden, h)

        hidden = self.s2_residual_blocks(hidden_fuse)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)

        return output_o, hidden_fuse
