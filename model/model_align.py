import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_crnn import DataConsistencyInKspace


def make_layer(block_func, n_layers):
    layers = list()
    for _ in range(n_layers):
        layers.append(block_func())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """
    def __init__(self, n_f=64, k_s=3):
        super(ResidualBlock_noBN, self).__init__()
        self.conv_1 = nn.Conv2d(n_f, n_f, k_s, padding='same')
        self.conv_2 = nn.Conv2d(n_f, n_f, k_s, padding='same')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.lrelu(out)
        out = self.conv_2(out)
        return x + out


class Encoder(nn.Module):
    def __init__(self, n_ch=2, n_f=64, n_blocks=5):
        super(Encoder, self).__init__()
        self.conv_down_sample = torch.nn.Sequential(
            nn.Conv2d(n_ch, n_f * 2, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(n_f * 2, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(n_f, n_f * 2, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(n_f * 2, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(n_f, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.res_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_f), n_blocks)
        self.conv_pyramid_1 = torch.nn.Sequential(
            nn.Conv2d(n_f, n_f, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(n_f, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv_pyramid_2 = torch.nn.Sequential(
            nn.Conv2d(n_f, n_f, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(n_f, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.pyramid_fuse_conv = nn.Sequential(
            nn.Conv2d(n_f * 3, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        fea_lr = self.conv_down_sample(x)

        fea_lr_1x = self.res_blocks(fea_lr)
        fea_lr_2x = self.conv_pyramid_1(fea_lr_1x)
        fea_lr_4x = self.conv_pyramid_2(fea_lr_2x)

        fea_lr_2x_up = F.interpolate(fea_lr_2x, size=fea_lr_1x.shape[-2:], mode='bilinear', align_corners=False)
        fea_lr_4x_up = F.interpolate(fea_lr_4x, size=fea_lr_1x.shape[-2:], mode='bilinear', align_corners=False)

        print(fea_lr_1x.shape)
        print(fea_lr_2x_up.shape)
        print(fea_lr_4x_up.shape)

        out = self.pyramid_fuse_conv(torch.cat([fea_lr_1x, fea_lr_2x_up, fea_lr_4x_up], dim=1))

        return out


class TimeAdapter(nn.Module):
    def __init__(self, patch_size=3, alpha=-1.0):
        super(TimeAdapter,self).__init__()
        self.patch_size = patch_size
        self.alpha = alpha

    def forward(self, nbr, ref):

        b, c, h, w = ref.size()
        ref_clone = ref.detach().clone()
        ref_flat = ref_clone.view(b, c, -1, h * w).permute(0, 3, 2, 1).contiguous().view(b * h * w, -1, c)
        ref_flat = torch.nn.functional.normalize(ref_flat, p=2, dim=-1)

        weight_diff = (nbr - ref) ** 2
        weight_diff = torch.exp(self.alpha * weight_diff)

        pad = self.patch_size // 2
        nbr_pad = torch.nn.functional.pad(nbr, (pad, pad, pad, pad), mode='reflect')
        nbr = torch.nn.functional.unfold(nbr_pad, kernel_size=self.patch_size).view(b, c, -1, h * w)
        nbr = torch.nn.functional.normalize(nbr, p=2, dim=1)
        nbr = nbr.permute(0, 3, 1, 2).contiguous().view(b * h * w, c, -1)
        d = torch.matmul(ref_flat, nbr).squeeze(1)
        weight_temporal = torch.nn.functional.softmax(d, -1)
        agg_fea = torch.einsum('bc,bnc->bn', weight_temporal, nbr).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        agg_fea = agg_fea * weight_diff

        return agg_fea


class Fuser(nn.Module):
    def __init__(self, n_f=64, num=2):
        super(Fuser, self).__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(n_f * num, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        out = self.conv_fuse(x)

        return out


class Decoder(nn.Module):
    def __init__(self, n_f=64, n_ch=2, n_blocks=5):
        super(Decoder, self).__init__()
        self.res_blocks = make_layer(lambda: ResidualBlock_noBN(n_f=n_f), n_blocks)
        self.conv_up_sample = nn.Sequential(
            nn.Conv2d(n_f, n_f * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(n_f, n_f * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(n_f, n_f, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(n_f, n_ch, 1, 1)
        )

    def forward(self, x):
        rec_fea = self.res_blocks(x)
        rec_fea = self.conv_up_sample(rec_fea)
        out = self.conv_out(rec_fea)

        return out


class TARN(nn.Module):
    def __init__(self, n_ch=2, n_f=64, enc_blocks=5, dec_blocks=5):
        super(TARN, self).__init__()
        self.encode = Encoder(n_ch=n_ch, n_f=n_f, n_blocks=enc_blocks)
        self.adapt = TimeAdapter()
        self.fuse = Fuser(n_f=n_f, num=2)
        self.decode = Decoder(n_ch=n_ch, n_f=n_f, n_blocks=dec_blocks)

        self.dc = DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m, last_fea=None):
        fea = self.encode(x)

        if last_fea is None:
            last_fea = torch.zeros_like(fea)
        fea_ada = self.adapt(last_fea, fea)
        fea_cat = torch.cat([fea, fea_ada], dim=1)
        fea = self.fuse(fea_cat)

        rec = self.decode(fea)
        rec = self.dc(rec, k, m)

        return rec, fea
