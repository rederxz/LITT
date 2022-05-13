import torch
import torch.nn as nn

from .model_crnn import DataConsistencyInKspace


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, n_f=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv_1 = nn.Conv2d(n_f, n_f, 3, padding='same')
        self.conv_2 = nn.Conv2d(n_f, n_f, 3, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.conv_2(out)
        return x + out


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
    def __init__(self, n_ch=2, n_h=64, n_blocks=5):
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
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)
        output_h = self.relu(self.s2_conv_h(hidden))

        return output_o, output_h
