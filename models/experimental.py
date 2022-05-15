# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
from models.common import Conv
from utils.downloads import attempt_download

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models.common import C3


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
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
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
            if t is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
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
    def __init__(self, c1, c2, n, act=True):
        super().__init__()
        self.pool = nn.MaxPool2d(n, n)
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
        self.sigmoid = nn.Sigmoid()

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
        y = self.sigmoid(y)

        return x * y.expand_as(x)


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
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
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


class PoolformerBlock(nn.Module):
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

    def __init__(self, c1, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

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
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class C3PF(C3):
    # C3 module with PoolformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=1.0):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.pf = nn.Sequential(*(PoolformerBlock(c2, c2) for _ in range(n)))
        self.m = self.pf


class Approx_GeLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_checkpointing = True

    def func(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        x = self.func(x)
        return x


def subtraction_gaussian_kernel_torch(q, k):
    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:])
    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = torch.ones(q.shape[-2:]) @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)


class SoftmaxFreeAttentionKernel(nn.Module):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, use_conv, max_iter=20, kernel_method="torch"):
        super().__init__()

        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads

        self.num_landmarks = num_landmark
        self.q_seq_len = q_len
        self.k_seq_len = k_len
        self.max_iter = max_iter

        if kernel_method == "torch":
            self.kernel_function = subtraction_gaussian_kernel_torch
        else:
            assert False, "please choose kernel method from torch"

        ratio = int(np.sqrt(self.q_seq_len // self.num_landmarks))
        if ratio == 1:
            self.Qlandmark_op = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.Qnorm_act = nn.Sequential(nn.LayerNorm(self.head_dim), nn.GELU())
        else:
            self.Qlandmark_op = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=ratio, stride=ratio, bias=False)
            self.Qnorm_act = nn.Sequential(nn.LayerNorm(self.head_dim), nn.GELU())

        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.num_head, out_channels=self.num_head,
                kernel_size=(self.use_conv, self.use_conv), padding=(self.use_conv // 2, self.use_conv // 2),
                bias=False,
                groups=self.num_head)

    def forward(self, Q, V):
        b, nhead, N, headdim, = Q.size()
        # Q: [b, num_head, N, head_dim]
        Q = Q / math.sqrt(math.sqrt(self.head_dim))
        K = Q
        if self.num_landmarks == self.q_seq_len:
            Q_landmarks = Q.reshape(b * self.num_head, int(np.sqrt(self.q_seq_len)) * int(np.sqrt(self.q_seq_len)) + 1,
                                    self.head_dim)
            Q_landmarks = self.Qlandmark_op(Q_landmarks)
            Q_landmarks = self.Qnorm_act(Q_landmarks).reshape(b, self.num_head, self.num_landmarks + 1, self.head_dim)
            K_landmarks = Q_landmarks
            attn = self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())
            attn = torch.exp(-attn / 2)
            X = torch.matmul(attn, V)

            h = w = int(np.sqrt(N))
            if self.use_conv:
                V_ = V[:, :, 1:, :]
                cls_token = V[:, :, 0, :].unsqueeze(2)
                V_ = V_.reshape(b, nhead, h, w, headdim)
                V_ = V_.permute(0, 4, 1, 2, 3).reshape(b * headdim, nhead, h, w)
                out = self.conv(V_).reshape(b, headdim, nhead, h, w).flatten(3).permute(0, 2, 3, 1)
                out = torch.cat([cls_token, out], dim=2)
                X += out
        else:
            Q_landmarks = Q.reshape(b * self.num_head, int(np.sqrt(self.q_seq_len)) * int(np.sqrt(self.q_seq_len)),
                                    self.head_dim).reshape(b * self.num_head, int(np.sqrt(self.q_seq_len)),
                                                           int(np.sqrt(self.q_seq_len)),
                                                           self.head_dim).permute(0, 3, 1, 2)
            Q_landmarks = self.Qlandmark_op(Q_landmarks)
            Q_landmarks = Q_landmarks.flatten(2).transpose(1, 2).reshape(b, self.num_head, self.num_landmarks,
                                                                         self.head_dim)
            Q_landmarks = self.Qnorm_act(Q_landmarks)
            K_landmarks = Q_landmarks

            kernel_1_ = self.kernel_function(Q, K_landmarks.transpose(-1, -2).contiguous())
            kernel_1_ = torch.exp(-kernel_1_ / 2)

            kernel_2_ = self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())
            kernel_2_ = torch.exp(-kernel_2_ / 2)

            kernel_3_ = kernel_1_.transpose(-1, -2)

            X = torch.matmul(torch.matmul(kernel_1_, self.newton_inv(kernel_2_)), torch.matmul(kernel_3_, V))

            h = w = int(np.sqrt(N))
            if self.use_conv:
                V = V.reshape(b, nhead, h, w, headdim)
                V = V.permute(0, 4, 1, 2, 3).reshape(b * headdim, nhead, h, w)
                X += self.conv(V).reshape(b, headdim, nhead, h, w).flatten(3).permute(0, 2, 3, 1)

        return X

    def newton_inv(self, mat):
        P = mat
        I = torch.eye(mat.size(-1), device=mat.device)
        alpha = 2 / (torch.max(torch.sum(mat, dim=-1)) ** 2)
        beta = 0.5
        V = alpha * P
        pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
        err_cnt = 0
        while pnorm > 1.01 and err_cnt < 10:
            alpha *= beta
            V = alpha * P
            pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
            err_cnt += 1

        for i in range(self.max_iter):
            V = 2 * V - V @ P @ V
        return V


class SoftmaxFreeAttention(nn.Module):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter=20, kernel_method="torch"):
        super().__init__()

        self.grad_checkpointing = True
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = SoftmaxFreeAttentionKernel(dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter,
                                               kernel_method)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, return_QKV=False):

        Q = self.split_heads(self.W_q(X))
        V = self.split_heads(self.W_v(X))
        attn_out = self.attn(Q, V)
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        if return_QKV:
            return out, (Q, V)
        else:
            return out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


class SoftmaxFreeTransformer(nn.Module):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size, drop_path=0., max_iter=20,
                 kernel_method="torch"):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(4 * dim)

        self.mha = SoftmaxFreeAttention(dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter, kernel_method)

        self.dropout1 = torch.nn.Dropout(p=drop_path)
        self.norm1 = nn.LayerNorm(self.dim)

        self.ff1 = nn.Linear(self.dim, self.hidden_dim)
        self.act = Approx_GeLU()
        self.ff2 = nn.Linear(self.hidden_dim, self.dim)

        self.dropout2 = torch.nn.Dropout(p=drop_path)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, X, return_QKV=False):

        if return_QKV:
            mha_out, QKV = self.mha(X, return_QKV=True)
        else:
            mha_out = self.mha(X)

        mha_out = self.norm1(X + self.dropout1(mha_out))
        ff_out = self.ff2(self.act(self.ff1(mha_out)))
        mha_out = self.norm2(mha_out + self.dropout2(ff_out))

        if return_QKV:
            return mha_out, QKV
        else:
            return mha_out


class SoftmaxFreeTrasnformerBlock(nn.Module):
    def __init__(self, dim, num_heads, H, W, drop_path=0., conv_size=3, max_iter=20, kernel_method="torch"):
        super().__init__()
        seq_len = 16
        self.att = SoftmaxFreeTransformer(dim, num_heads, int(H * W), int(H * W), seq_len, conv_size, drop_path,
                                          max_iter, kernel_method)

    def forward(self, x):
        x = self.att(x)
        return x


class Mlp_S(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_S(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=7, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if kernel_size == 7:
            self.proj = nn.Sequential(nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(32), nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32), nn.ReLU(),
                                      nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(embed_dim), nn.ReLU())
        elif kernel_size == 3:
            self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(embed_dim), nn.ReLU())
        else:
            self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(embed_dim), nn.ReLU())

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class SFTR(nn.Module):
    """Normally,
    embed_dims=[128, 256, 512]
    num_heads=[1, 2, 4]
    depths = [1, 3, 2]
    """

    def __init__(self, in_chans, out_channels, stage=None, embed_dims=None, num_heads=None, depths=None,
                 img_size=160, patch_size=2, drop_rate=0., drop_path_rate=0., newton_max_iter=20,
                 kernel_method="torch"):
        super().__init__()
        if depths is None:
            depths = [3, 6, 3]
        if num_heads is None:
            num_heads = [1, 4, 8]
        if embed_dims is None:
            embed_dims = [256, 512, 1024]
        self.depths = depths
        self.stage = stage

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, kernel_size=3, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, kernel_size=3, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, kernel_size=3, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, (img_size // 2)**2, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, (img_size // 4)**2, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, (img_size // 8)**2, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([SoftmaxFreeTrasnformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], H=80, W=80, conv_size=9,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([SoftmaxFreeTrasnformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], drop_path=dpr[cur + i], H=40, W=40, conv_size=5,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([SoftmaxFreeTrasnformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], drop_path=dpr[cur + i], H=20, W=20, conv_size=3,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[2])])

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]

        if self.stage == 1:
            # stage 1
            x, (H, W) = self.patch_embed1(x)
            x = x + self.pos_embed1
            x = self.pos_drop1(x)
            for blk in self.block1:
                x = blk(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            return x
        elif self.stage == 2:
            # stage 1
            x, (H, W) = self.patch_embed1(x)
            x = x + self.pos_embed1
            x = self.pos_drop1(x)
            for blk in self.block1:
                x = blk(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # stage 2
            x, (H, W) = self.patch_embed2(x)
            x = x + self.pos_embed2
            x = self.pos_drop2(x)
            for blk in self.block2:
                x = blk(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            return x
        elif self.stage == 3:
            # stage 1
            x, (H, W) = self.patch_embed1(x)
            x = x + self.pos_embed1
            x = self.pos_drop1(x)
            for blk in self.block1:
                x = blk(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # stage 2
            x, (H, W) = self.patch_embed2(x)
            x = x + self.pos_embed2
            x = self.pos_drop2(x)
            for blk in self.block2:
                x = blk(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # stage 3
            x, (H, W) = self.patch_embed3(x)
            x = x + self.pos_embed3
            x = self.pos_drop3(x)
            for blk in self.block3:
                x = blk(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            return x
        else:
            assert False, "please choose between (1, 3)"

    def forward(self, x):
        return self.forward_features(x)
