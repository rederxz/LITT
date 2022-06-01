import torch
import torch.nn as nn

from .model_crnn import DataConsistencyInKspace
from .model_zy import ResidualBlock_noBN, make_layer


class RRN_plain(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        super(RRN_plain, self).__init__()
        self.n_h = n_h

        self.proj_conv = nn.Conv2d(n_ch + n_h + n_ch + n_ch, n_h, k_s, padding='same')
        self.residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        inputs = torch.cat([x, x_l, o, h], dim=1)
        hidden = self.proj_conv(inputs)
        hidden = self.residual_blocks(hidden)
        output_o = self.dc(self.rec_conv(hidden), k, m)

        return output_o, hidden


class RRN_plain_two_stage(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        super(RRN_plain_two_stage, self).__init__()
        self.n_h = n_h

        self.s1_proj_conv = nn.Conv2d(n_ch + n_h + n_ch + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        self.s2_proj_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        inputs = torch.cat([x, x_l, o, h], dim=1)
        hidden = self.s1_proj_conv(inputs)
        hidden = self.s1_residual_blocks(hidden)
        rec = self.s1_dc(self.s1_rec_conv(hidden), k, m)

        inputs = torch.cat([rec, hidden], dim=1)
        hidden = self.s2_proj_conv(inputs)
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_rec_conv(hidden), k, m)

        return output_o, hidden


class RRN_plain_three_stage(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        super(RRN_plain_three_stage, self).__init__()
        self.n_h = n_h

        self.s1_proj_conv = nn.Conv2d(n_ch + n_h + n_ch + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        self.s2_proj_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.s3_proj_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s3_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s3_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s3_dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        inputs = torch.cat([x, x_l, o, h], dim=1)
        hidden = self.s1_proj_conv(inputs)
        hidden = self.s1_residual_blocks(hidden)
        rec = self.s1_dc(self.s1_rec_conv(hidden), k, m)

        inputs = torch.cat([rec, hidden], dim=1)
        hidden = self.s2_proj_conv(inputs)
        hidden = self.s2_residual_blocks(hidden)
        rec = self.s2_dc(self.s2_rec_conv(hidden), k, m)

        inputs = torch.cat([rec, hidden], dim=1)
        hidden = self.s3_proj_conv(inputs)
        hidden = self.s3_residual_blocks(hidden)
        output_o = self.s3_dc(self.s3_rec_conv(hidden), k, m)

        return output_o, hidden


class RRN_plain_five_stage(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        super(RRN_plain_five_stage, self).__init__()
        self.n_h = n_h

        self.s1_proj_conv = nn.Conv2d(n_ch + n_h + n_ch + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        self.s2_proj_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

        self.s3_proj_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s3_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s3_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s3_dc = DataConsistencyInKspace(norm='ortho')

        self.s4_proj_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s4_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s4_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s4_dc = DataConsistencyInKspace(norm='ortho')

        self.s5_proj_conv = nn.Conv2d(n_ch + n_h, n_h, k_s, padding='same')
        self.s5_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s5_rec_conv = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s5_dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        inputs = torch.cat([x, x_l, o, h], dim=1)
        hidden = self.s1_proj_conv(inputs)
        hidden = self.s1_residual_blocks(hidden)
        rec = self.s1_dc(self.s1_rec_conv(hidden), k, m)

        inputs = torch.cat([rec, hidden], dim=1)
        hidden = self.s2_proj_conv(inputs)
        hidden = self.s2_residual_blocks(hidden)
        rec = self.s2_dc(self.s2_rec_conv(hidden), k, m)

        inputs = torch.cat([rec, hidden], dim=1)
        hidden = self.s3_proj_conv(inputs)
        hidden = self.s3_residual_blocks(hidden)
        rec = self.s3_dc(self.s3_rec_conv(hidden), k, m)

        inputs = torch.cat([rec, hidden], dim=1)
        hidden = self.s4_proj_conv(inputs)
        hidden = self.s4_residual_blocks(hidden)
        rec = self.s4_dc(self.s4_rec_conv(hidden), k, m)

        inputs = torch.cat([rec, hidden], dim=1)
        hidden = self.s5_proj_conv(inputs)
        hidden = self.s5_residual_blocks(hidden)
        output_o = self.s5_dc(self.s5_rec_conv(hidden), k, m)

        return output_o, hidden


class RRN_two_stage_v4(nn.Module):
    def __init__(self, n_ch=2, n_h=64, k_s=3, n_blocks=5):
        super(RRN_two_stage_v4, self).__init__()
        self.n_h = n_h

        # stage 1
        self.s1_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s1_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s1_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s1_dc = DataConsistencyInKspace(norm='ortho')

        # stage 2
        self.s2_conv = nn.Conv2d(n_ch + n_h + n_ch, n_h, k_s, padding='same')
        self.s2_residual_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_h, k_s=k_s), n_blocks)
        self.s2_conv_o = nn.Conv2d(n_h, n_ch, k_s, padding='same')
        self.s2_dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, x_l=None, h=None, o=None):
        n_b, n_ch, width, height = x.shape
        if x_l is None:
            x_l = x.new_zeros([n_b, n_ch, width, height])
        if h is None:
            h = x.new_zeros([n_b, self.n_h, width, height])
        if o is None:
            o = x.new_zeros([n_b, n_ch, width, height])

        stage_1_input = torch.cat([x, x_l, h], dim=1)
        hidden = self.s1_conv(stage_1_input)
        hidden = self.s1_residual_blocks(hidden)
        stage_1_output = self.s1_dc(self.s1_conv_o(hidden), k, m)

        stage_2_input = torch.cat([stage_1_output, o, hidden], dim=1)
        hidden = self.s2_conv(stage_2_input)
        hidden = self.s2_residual_blocks(hidden)
        output_o = self.s2_dc(self.s2_conv_o(hidden), k, m)

        return output_o, hidden
