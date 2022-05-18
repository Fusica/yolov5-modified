# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
from models.common import Conv
from models.common import autopad
from utils.downloads import attempt_download

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.ops import deform_conv2d


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


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


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).float()  # FP32 model
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
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
    def __init__(self, c1, c2, k, s, p, d=1, act=True):
        super(DSConv, self).__init__()
        self.DConv = nn.Conv2d(c1, c1, k, s, p, d, c1)
        self.PConv = nn.Conv2d(c1, c2, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.PConv(self.DConv(x))))


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
        self.relu = nn.ReLU()

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
        y = self.relu(y)

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


# class DFConv(nn.Module):
#     def __init__(self, c1, c2, k=3, s=1, p=(0, 0), act=True):
#         super(DFConv, self).__init__()
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.c1, self.c2, self.k, self.s, self.p = c1, c2, k, s, p
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         weight = torch.rand(self.c2, self.c1, self.k)
#         offset = torch.rand(B, 2 * self.k * self.k, H, W)
#         mask = torch.rand(B, self.k * self.k, H, W)
#         x = deform_conv2d(x, offset, weight, padding=autopad(self.k, self.p), mask=mask)
#         return self.act(self.bn(x))


class DFConv(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, padding=1, act=True, bias=False):
        super(DFConv, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(c1,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(c1,
                                        kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(c1,
                                      c2,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w)/4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.relu(self.modulator_conv(x))

        # self.regular_conv.weight = self.regular_conv.weight.half() if x.dtype == torch.float16 else \
        #     self.regular_conv.weight
        x = deform_conv2d(input=x,
                          offset=offset,
                          weight=self.regular_conv.weight,
                          bias=self.regular_conv.bias,
                          padding=(self.padding, self.padding),
                          mask=modulator,
                          stride=self.stride,
                          )
        return x


# class DFConv(nn.Module):
#     def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None, modulation=True):
#         """
#         Args:
#             modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
#         """
#         super(DFConv, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.zero_padding = nn.ZeroPad2d(padding)
#         self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
#
#         self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, stride=stride, padding=1)
#         nn.init.constant_(self.p_conv.weight, 0)
#
#         self.modulation = modulation
#         if modulation:
#             self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, stride=stride, padding=1)
#             nn.init.constant_(self.m_conv.weight, 0)
#
#     @staticmethod
#     def _set_lr(module, grad_input, grad_output):
#         grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
#         grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
#
#     def forward(self, x):
#         offset = self.p_conv(x)
#         if self.modulation:
#             m = torch.sigmoid(self.m_conv(x))
#
#         dtype = offset.data.type()
#         ks = self.kernel_size
#         N = offset.size(1) // 2
#
#         if self.padding:
#             x = self.zero_padding(x)
#
#         # (b, 2N, h, w)
#         p = self._get_p(offset, dtype)
#
#         # (b, h, w, 2N)
#         p = p.contiguous().permute(0, 2, 3, 1)
#         q_lt = p.detach().floor()
#         q_rb = q_lt + 1
#
#         q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
#         q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
#
#         # clip p
#         p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
#
#         # bilinear kernel (b, h, w, N)
#         g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
#         g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
#         g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
#         g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
#
#         # (b, c, h, w, N)
#         x_q_lt = self._get_x_q(x, q_lt, N)
#         x_q_rb = self._get_x_q(x, q_rb, N)
#         x_q_lb = self._get_x_q(x, q_lb, N)
#         x_q_rt = self._get_x_q(x, q_rt, N)
#
#         # (b, c, h, w, N)
#         x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
#                    g_rb.unsqueeze(dim=1) * x_q_rb + \
#                    g_lb.unsqueeze(dim=1) * x_q_lb + \
#                    g_rt.unsqueeze(dim=1) * x_q_rt
#
#         # modulation
#         if self.modulation:
#             m = m.contiguous().permute(0, 2, 3, 1)
#             m = m.unsqueeze(dim=1)
#             m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
#             x_offset *= m
#
#         x_offset = self._reshape_x_offset(x_offset, ks)
#         out = self.conv(x_offset)
#
#         return out
#
#     def _get_p_n(self, N, dtype):
#         p_n_x, p_n_y = torch.meshgrid(
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
#         # (2N, 1)
#         p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
#         p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
#
#         return p_n
#
#     def _get_p_0(self, h, w, N, dtype):
#         p_0_x, p_0_y = torch.meshgrid(
#             torch.arange(1, h*self.stride+1, self.stride),
#             torch.arange(1, w*self.stride+1, self.stride))
#         p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
#
#         return p_0
#
#     def _get_p(self, offset, dtype):
#         N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
#
#         # (1, 2N, 1, 1)
#         p_n = self._get_p_n(N, dtype)
#         # (1, 2N, h, w)
#         p_0 = self._get_p_0(h, w, N, dtype)
#         p = p_0 + p_n + offset
#         return p
#
#     def _get_x_q(self, x, q, N):
#         b, h, w, _ = q.size()
#         padded_w = x.size(3)
#         c = x.size(1)
#         # (b, c, h*w)
#         x = x.contiguous().view(b, c, -1)
#
#         # (b, h, w, N)
#         index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
#         # (b, c, h*w*N)
#         index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
#
#         x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
#
#         return x_offset
#
#     @staticmethod
#     def _reshape_x_offset(x_offset, ks):
#         b, c, h, w, N = x_offset.size()
#         x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
#         x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
#
#         return x_offset


class BottleneckDF(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DFConv(c1, c_, 1, 1, 0)
        self.cv2 = DFConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3DF(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, g=4)
        self.cv2 = Conv(c1, c_, 1, 1, g=4)
        self.cv3 = Conv(2 * c_, c2, 1, g=4)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckDF(c_, c_, shortcut, 1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# test = DFConv(64, 64, 3, 1, 1)
# input = torch.rand(2, 64, 20, 20)
# output = test(input)
# print(output.shape)
