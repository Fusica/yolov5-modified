# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
from models.coat import SerialBlock
from models.common import Conv, Bottleneck, C3, DWConv, autopad
from models.SwinTransformer import SwinTransformerBlock
from models.cswin import CSWinTransformer
from utils import torch_utils
from utils.activations import AconC

from utils.downloads import attempt_download

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.ops import deform_conv2d
import time


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model  # return ensemble


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, d=1, act=True):
        super(DSConv, self).__init__()
        self.DConv = nn.Conv2d(c1, c1, k, s, p, d, c1)
        self.PConv = nn.Conv2d(c1, c2, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.PConv(self.DConv(x))))


class DSConv_A(DSConv):
    def __init__(self, c1, c2, k=3, s=1, p=1, d=1, act=True):
        super(DSConv_A, self).__init__(c1, c2, k, s, p, d, act)
        if act:
            self.act = AconC(c2)
        else:
            self.act = nn.Identity()


class DWConv_A(DWConv):
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super(DWConv_A, self).__init__(c1, c2, k, s, act)
        if act:
            self.act = AconC(c2)
        else:
            self.act = nn.Identity()


class DConv(nn.Module):
    """
    Dilation Convolution
    Didn't work as expected
    """

    def __init__(self, c1, c2, k=5, s=7, p=1, d=7, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, d)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Pool(nn.Module):
    def __init__(self, c1, c2, pool_size=2, act=True):
        super().__init__()
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.conv = nn.Conv2d(c1, c2, 1, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(self.pool(x))))


class Reshape(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ECA(nn.Module):
    """Constructs an ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # print(y.shape,y.squeeze(-1).shape,y.squeeze(-1).transpose(-1, -2).shape)
        # Two different branches of ECA module
        # 50*C*1*1
        # 50*C*1
        # 50*1*C
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.silu(y)

        return x * y.expand_as(x)


class PoolECA(nn.Module):
    def __init__(self, c1, c2, pool_size=2, act=True):
        super(PoolECA, self).__init__()
        self.pool = Pool(c1, c2, pool_size, act)
        self.eca = ECA(c2, c2)

    def forward(self, x):
        return self.eca(self.pool(x))


class CropLayer(nn.Module):
    # E.g., (-1, 0) means this layer should crop the first and last rows of the feature map.
    # And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride, padding, deploy=False, dilation=1, groups=1, padding_mode='zeros'):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = DSConv(c1, c2, kernel_size, stride, padding, dilation)
        else:
            self.square_conv = DSConv(c1, c2, kernel_size, stride, padding, dilation)
            self.square_bn = nn.BatchNorm2d(num_features=c2)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = DSConv(c1, c2, (3, 1), stride, ver_conv_padding, dilation)
            self.hor_conv = DSConv(c1, c2, (1, 3), stride, hor_conv_padding, dilation)
            self.ver_bn = nn.BatchNorm2d(num_features=c2)
            self.hor_bn = nn.BatchNorm2d(num_features=c2)

    # forward函数
    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs


class ACBlocks(nn.Module):
    def __init__(self, c1, c2):
        super(ACBlocks, self).__init__()
        self.acblock1 = ACBlock(c1, c2, 3, 2, 1)
        self.acblock2 = ACBlock(c2, c2, 3, 1, 1, True)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.acblock2(self.acblock1(x))))


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, out_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              out_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, )


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.MaxPool2d(
            pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PFBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, c2, down_sampling=True, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.SiLU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.conv = nn.Conv2d(dim, c2, 3, 2, 1)
        self.downsample = down_sampling

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
            if self.downsample:
                x = self.conv(x)
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if self.downsample:
                x = self.conv(x)
        return x


class PF(nn.Module):
    # module with PoolformerBlock()
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.pf = nn.Sequential(*(PFBlock(c1, c2, False) for _ in range(n)))

    def forward(self, x):
        return self.pf(x)


class DFConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DFConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding if type(stride) == tuple else (padding, padding)

        # init weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=kernel_size,
                                          stride=stride, padding=self.padding, bias=True)

        # init
        self.reset_parameters()
        self._init_weight()

    def reset_parameters(self):
        n = self.in_channels * (self.kernel_size ** 2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = deform_conv2d(input=x, offset=offset, weight=self.weight, bias=self.bias, padding=self.padding, mask=mask,
                          stride=self.stride)
        return x


class BottleneckDF(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DFConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3DF(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, e=e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(BottleneckDF(c_, c_, shortcut, 1.0) for _ in range(n)))


class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=3, shortcut=True, g=1, e=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_ // 32, n)


class Gap(nn.Module):
    def __init__(self, c1, c2):
        super(Gap, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(self.gap(x))


class PixShuffle(nn.Module):
    def __init__(self, c1, c2, factor=2, shuffle=True):
        super(PixShuffle, self).__init__()
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        if shuffle:
            self.shuffle = nn.PixelShuffle(factor)
        else:
            self.shuffle = nn.PixelUnshuffle(factor)

    def forward(self, x):
        return self.act(self.bn(self.shuffle(x)))


class C3Conv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.cv4 = Conv(c1, c2, 3, 2)

    def forward(self, x):
        return self.cv4(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))


# add CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


# 标准卷积层 + CBAM
class ConvCBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvCBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class CBAMBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16,
                 kernel_size=7):  # ch_in, ch_out, shortcut, groups, expansion
        super(CBAMBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        out = self.channel_attention(x1) * x1
        out = self.spatial_attention(out) * out
        return x + out if self.add else out


class C3CBAM(C3):
    # C3 module with CBAMBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut) for _ in range(n)))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        pool_h = nn.AdaptiveAvgPool2d((h, 1))
        x_h = pool_h(x)
        # c*H*1
        # C*1*h
        pool_w = nn.AdaptiveAvgPool2d((1, w))
        x_w = pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class CABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=32):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.ca=CoordAtt(c1,c2,ratio)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // ratio)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        n, c, h, w = x.size()
        # c*1*W
        pool_h = nn.AdaptiveAvgPool2d((h, 1))
        x_h = pool_h(x)
        # c*H*1
        # C*1*h
        pool_w = nn.AdaptiveAvgPool2d((1, w))
        x_w = pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = x1 * a_w * a_h

        # out=self.ca(x1)*x1
        return x + out if self.add else out


class C3CA(C3):
    # C3 module with CABottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut) for _ in range(n)))


class Upsample(nn.Module):
    def __init__(self, c1, c2, n=1):
        super(Upsample, self).__init__()
        assert c1 > c2, "c1 must larger than c2"
        self.num = n
        self.inchannel = c1
        self.outchannel = c2
        self.upsample = self.make_upsample()

    def make_upsample(self):
        inchannel, outchannel, layer = [], [], []
        if self.num == 1:
            inchannel = [self.inchannel]
            outchannel = [self.outchannel]
        elif self.num == 2:
            inchannel = [self.inchannel, self.inchannel // 2]
            outchannel = [self.inchannel // 2, self.outchannel]
        elif self.num == 3:
            inchannel = [self.inchannel, self.inchannel // 2, self.inchannel // 4]
            outchannel = [self.inchannel // 2, self.inchannel // 4, self.outchannel]
        elif self.num == 4:
            inchannel = [self.inchannel, self.inchannel // 2, self.inchannel // 4, self.inchannel // 8]
            outchannel = [self.inchannel // 2, self.inchannel // 4, self.inchannel // 8, self.outchannel]

        for i in range(self.num):
            layer.append(nn.Upsample(scale_factor=2.))
            layer.append(nn.BatchNorm2d(inchannel[i]))
            layer.append(nn.SiLU())
            layer.append(DWConv(inchannel[i], outchannel[i], 1, 1, act=False))
        return nn.Sequential(*layer)

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, c1, c2, n=1):
        super(Downsample, self).__init__()
        assert c1 < c2, "c1 must smaller than c2"
        self.num = n
        self.inchannel = c1
        self.outchannel = c2
        self.downsample = self.make_downsample()

    def make_downsample(self):
        inchannel, outchannel, layer = [], [], []
        if self.num == 1:
            inchannel = [self.inchannel]
            outchannel = [self.outchannel]
        elif self.num == 2:
            inchannel = [self.inchannel, self.inchannel * 2]
            outchannel = [self.inchannel * 2, self.outchannel]
        elif self.num == 3:
            inchannel = [self.inchannel, self.inchannel * 2, self.inchannel * 4]
            outchannel = [self.inchannel * 2, self.inchannel * 4, self.outchannel]
        elif self.num == 4:
            inchannel = [self.inchannel, self.inchannel * 2, self.inchannel * 4, self.inchannel * 8]
            outchannel = [self.inchannel * 2, self.inchannel * 4, self.inchannel * 8, self.outchannel]

        for i in range(self.num):
            layer.append(DSConv(inchannel[i], outchannel[i], 3, 2, 1))
        return nn.Sequential(*layer)

    def forward(self, x):
        return self.downsample(x)


class BottleneckDS(Bottleneck):
    # Deep Separate convolution Bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DSConv(c_, c2, 3, 1)


class C3DS(C3):
    # CSP Bottleneck with 3 Deep Separate convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BottleneckDS(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class Add_Bi(nn.Module):  # pairwise add
    # Concatenate a list of tensors along dimension
    def __init__(self, n=2):
        super(Add_Bi, self).__init__()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.w4 = nn.Parameter(torch.ones(5, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # self.act= nn.SiLU()  #这里原本用silu，但是用途应该是保证权重是0-1之间 所以改成relu
        self.act = nn.ReLU()

    def forward(self, x):  # mutil-layer 1-3 layers
        # print("bifpn:",x.shape)
        if len(x) == 2:
            # w = self.relu(self.w1)
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.act(weight[0] * x[0] + weight[1] * x[1])
        elif len(x) == 3:
            # w = self.relu(self.w2)
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2])
        elif len(x) == 4:
            w = self.w3
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2] + weight[3] * x[3])
        elif len(x) == 5:
            w = self.w4
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2] + weight[3] * x[3] + weight[3] * x[3])
        return x


class Add_weight(nn.Module):
    def __init__(self, n=2):
        super(Add_weight, self).__init__()
        self.num = n

    def forward(self, x):
        if self.num == 2:
            return x[0] + x[1]
        elif self.num == 3:
            return x[0] + x[1] + x[2]
        elif self.num == 4:
            return x[0] + x[1] + x[2] + x[3]


class ConvACON(Conv):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvACON, self).__init__(c1, c2, k, s, p, g, act)
        if act:
            self.act = AconC(c2)
        else:
            self.act = nn.Identity()


class HRBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super(HRBlock, self).__init__()
        assert c1 == c2, "must match channels"
        self.extract = nn.Sequential(
            Conv(c1, c2, 3, 1, act=False),
            Conv(c2, c2, 3, 1)
        )
        self.residual = shortcut and c1 == c2

    def forward(self, x):
        return x + self.extract(x) if self.residual else self.extract(x)


class HRStage(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super(HRStage, self).__init__()
        self.m = nn.Sequential(*(HRBlock(c1, c2, shortcut) for _ in range(4)))

    def forward(self, x):
        return self.m(x)


class DGBlock(nn.Module):
    def __init__(self, c1, c2):
        super(DGBlock, self).__init__()


class Involution(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = c1
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvACON(c1, c1 // reduction_ratio, 1, 1)
        # nn.Sequential(
        #     nn.Conv2d(c1, c1 // reduction_ratio, 1, 1, 0),
        #     nn.BatchNorm2d(c1 // reduction_ratio),
        #     nn.ReLU()
        # )
        self.conv2 = ConvACON(c1 // reduction_ratio, kernel_size ** 2 * self.groups, 1, 1, act=False)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class BottleneckInvolution(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=4.0):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvACON(c1, c_, 1, 1)
        self.cv2 = nn.Sequential(
            Involution(c_, c_, 7, 1),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )
        self.cv3 = ConvACON(c_, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C3Involution(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BottleneckInvolution(c_, c_) for _ in range(n)))


class CSWinBlock(CSWinTransformer):
    """
    default:
    img_size=[160, 80, 40, 20], embed_dim=[96, 144, 192, 240], split_size=[8, 16, 20, 20], depth=[2, 2, 6, 2]
    """

    def __init__(self, c1, c2, stage, embed_dim, split_size, depth):
        super(CSWinBlock, self).__init__(c1, c2, stage, embed_dim, split_size, depth)


class DiBlock(nn.Module):
    def __init__(self, c1, c2, n, shortcut=True):
        super(DiBlock, self).__init__()
        assert c1 == c2, "c1 and c2 must be equal"
        self.shortcut = shortcut
        c_ = int(c1 // 2)
        self.deformable = DFConv(c_, c_)
        self.involution = C3Involution(c_, c_, n)
        self.conv = DWConv_A(c1, c2, 3, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = AconC(c2)

    def forward(self, x):
        residual = x
        x = torch.chunk(x, 2, dim=1)
        df_output = self.deformable(x[0])
        iv_output = self.involution(x[1])
        cv_output = self.conv(residual)

        return self.act(self.bn(torch.cat((df_output, iv_output), dim=1) + cv_output))


class CTGBlock(nn.Module):
    """
    Inspired by iFormer, use two main path to extract features, Involution, Maxpool, CoaT
    """

    def __init__(self, c1, c2, n, shortcut=True):
        super(CTGBlock, self).__init__()
        assert c1 == c2, "c1 and c2 must be equal"
        self.shortcut = shortcut
        c_h, c_l = None, None
        if n == 1:
            # 160*160 [3/8, 3/8, 1/4]
            c_h = int(c1 // 8 * 3)
            c_l = int(c1 // 1 / 4)
        elif n == 2:
            # 80*80 [1/4, 1/4, 2/4]
            c_h = int(c1 // 4)
            c_l = int(c1 // 2)
        elif n == 3:
            # 40*40 [1/8, 1/8, 3/4]
            c_h = int(c1 // 8)
            c_l = int(c1 // 4 * 3)
        self.involution = BottleneckInvolution(c_h, c_h)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.conv = ConvACON(c_h, c_h, 3, 1, 1)
        self.coat = CoaTBlock(c_l, c_l, 2)
        self.coordatt = CoordAtt(c1, c2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = AconC(c2)
        self.n = n

    def forward(self, x):
        residual = x
        if self.n == 1:
            x_h = int(x.shape[1] // 8 * 3)
            x_l = int(x.shape[1] // 4)
            x = torch.split(x, [x_h, x_h, x_l], dim=1)
        elif self.n == 2:
            x_h = int(x.shape[1] // 4)
            x_l = int(x.shape[1] // 2)
            x = torch.split(x, [x_h, x_h, x_l], dim=1)
        elif self.n == 3:
            x_h = int(x.shape[1] // 8)
            x_l = int(x.shape[1] // 4 * 3)
            x = torch.split(x, [x_h, x_h, x_l], dim=1)
        else:
            assert "only operate on first 3 maps"
        x_h1, x_h2 = x[0], x[1]  # x_h[0], x_h[1]: maxpool and involution
        x_l = x[2]
        output_h1 = self.conv(self.maxpool(x_h1))
        output_h2 = self.involution(x_h2)
        output_l = self.coat(x_l)
        output_add = self.coordatt(self.gap(residual))

        return residual + self.act(self.bn(output_add + torch.cat((output_h1, output_h2, output_l), dim=1)))


class CoaTBlock(nn.Module):
    def __init__(self, c1, c2, num_layers):
        super(CoaTBlock, self).__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.tr = nn.Sequential(*(SerialBlock(c2, 8) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(0, 2, 1)  # B, N, C
        return self.tr(p).permute(0, 2, 1).reshape(b, self.c2, w, h)


class C3CoaT(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = CoaTBlock(c_, c_, n)


class LKA(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.conv0 = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)
        # padding is too big
        self.conv_spatial1 = nn.Conv2d(c1, c1, 5, stride=1, padding=8, groups=c1, dilation=4)
        self.conv_spatial = nn.Conv2d(c1, c1, 7, stride=1, padding=9, groups=c1, dilation=3)
        self.conv_spatial3 = nn.Conv2d(c1, c1, 9, stride=1, padding=9, groups=c1, dilation=3)
        self.conv1 = nn.Conv2d(c1, c1, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_CABlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super(LKA_CABlock, self).__init__()
        assert c1 == c2, "mustn't change channels"
        c_ = int(c1 // 2)
        c__ = int(c_ * 2)
        self.conv1 = nn.Conv2d(c_, c__, 1)
        self.lka = LKA(c__)
        self.conv2 = nn.Conv2d(c__, c_, 3, 1, 1)
        self.ca = CoordAtt(c_, c_)
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        x = torch.chunk(x, 2, 1)
        output_1 = self.conv2(self.lka(self.conv1(x[0])))
        output_2 = self.ca(x[1])
        if self.shortcut:
            return residual + torch.cat((output_1, output_2), dim=1)
        else:
            return torch.cat((output_1, output_2), dim=1)


class selayer(nn.Module):
    def __init__(self, c1, c2, ratio=16) -> None:
        super(selayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class HRBlock_SE(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super(HRBlock_SE, self).__init__()
        assert c1 == c2, "must match channels"
        c_ = int(c1 * 0.5)
        self.extract = nn.Sequential(
            DSConv_A(c1, c_, 3, 1, act=False),
            selayer(c_, c_),
            DSConv(c_, c2, 3, 1)
        )
        self.bn = nn.BatchNorm2d(c1)
        self.residual = shortcut and c1 == c2

    def forward(self, x):
        return self.bn(x) + self.extract(x) if self.residual else self.extract(x)


class HRStage_SE(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super(HRStage_SE, self).__init__()
        self.m = nn.Sequential(*(HRBlock_SE(c1, c2, shortcut) for _ in range(4)))

    def forward(self, x):
        return self.m(x)


# test = HRStage_SE(128, 128)
# input = torch.rand(1, 128, 160, 160)
#
# startTime = time.time()
# output = test(input)
# endTime = time.time()
# print(endTime - startTime)
# print(output.shape)


# test = ADiCBlock(512, 512, 3)
#
# input = torch.rand(1, 512, 40, 40)
# start = time.time()
# output = test(input)
# end = time.time()
# print(output.shape)
# print(end-start)


# test = Add_Bi(5)
#
# input1 = [torch.rand(1, 64, 20, 20),
#           torch.rand(1, 64, 1, 1),
#           torch.rand(1, 64, 20, 20),
#           torch.rand(1, 64, 20, 20),
#           torch.rand(1, 64, 20, 20)]
#
# output1 = test(input1)
# print(output1.shape)
