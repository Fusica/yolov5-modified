import torch

# a = torch.rand((2, 2, 1, 1))
# b = torch.rand((2, 2, 5, 5))
# print(a)
# print(b)
# print("-----------------")
# print(a+b)


# a = torch.arange(3).reshape((3, 1))
# b = torch.arange(4).reshape((1, 4))
# print(a + b)
# from torch import nn
#
#
# class DSConv(nn.Module):
#     def __init__(self, c1, c2, k, s, p=0, d=1, act=True):
#         super(DSConv, self).__init__()
#         self.DConv = nn.Conv2d(c1, c1, k, s, p, d, c1)
#         self.PConv = nn.Conv2d(c1, c2, 1)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#     def forward(self, x):
#         return self.act(self.bn(self.PConv(self.DConv(x))))


# dsconv = DSConv(3, 64, 6, 2, 2)
# conv = nn.Conv2d(3, 64, 6, 2, 2)

# input = torch.randn((3, 3, 32, 32))
#
# output1 = dsconv(input)
# print(output1.shape)
#
# output2 = conv(input)
# print(output2.shape)


# class CropLayer(nn.Module):
#     # E.g., (-1, 0) means this layer should crop the first and last rows of the feature map.
#     # And (0, -1) crops the first and last columns
#     def __init__(self, crop_set):
#         super(CropLayer, self).__init__()
#         self.rows_to_crop = - crop_set[0]
#         self.cols_to_crop = - crop_set[1]
#         assert self.rows_to_crop >= 0
#         assert self.cols_to_crop >= 0
#
#     def forward(self, input):
#         return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]
#
#
# class ACBlock(nn.Module):
#     def __init__(self, c1, c2, kernel_size, stride, padding, deploy=False, dilation=1, groups=1, padding_mode='zeros'):
#         super(ACBlock, self).__init__()
#         self.deploy = deploy
#         if deploy:
#             self.fused_conv = DSConv(c1, c2, kernel_size, stride, padding, dilation)
#         else:
#             self.square_conv = DSConv(c1, c2, kernel_size, stride, padding, dilation)
#             self.square_bn = nn.BatchNorm2d(num_features=c2)
#
#             center_offset_from_origin_border = padding - kernel_size // 2
#             ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
#             hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
#             if center_offset_from_origin_border >= 0:
#                 self.ver_conv_crop_layer = nn.Identity()
#                 ver_conv_padding = ver_pad_or_crop
#                 self.hor_conv_crop_layer = nn.Identity()
#                 hor_conv_padding = hor_pad_or_crop
#             else:
#                 self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
#                 ver_conv_padding = (0, 0)
#                 self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
#                 hor_conv_padding = (0, 0)
#             self.ver_conv = DSConv(c1, c2, (3, 1), stride, ver_conv_padding, dilation)
#             self.hor_conv = DSConv(c1, c2, (1, 3), stride, hor_conv_padding, dilation)
#             self.ver_bn = nn.BatchNorm2d(num_features=c2)
#             self.hor_bn = nn.BatchNorm2d(num_features=c2)
#
#     # forward函数
#     def forward(self, input):
#         if self.deploy:
#             return self.fused_conv(input)
#         else:
#             square_outputs = self.square_conv(input)
#             square_outputs = self.square_bn(square_outputs)
#             # print(square_outputs.size())
#             # return square_outputs
#             vertical_outputs = self.ver_conv_crop_layer(input)
#             vertical_outputs = self.ver_conv(vertical_outputs)
#             vertical_outputs = self.ver_bn(vertical_outputs)
#             # print(vertical_outputs.size())
#             horizontal_outputs = self.hor_conv_crop_layer(input)
#             horizontal_outputs = self.hor_conv(horizontal_outputs)
#             horizontal_outputs = self.hor_bn(horizontal_outputs)
#             # print(horizontal_outputs.size())
#             return square_outputs + vertical_outputs + horizontal_outputs
#
#
# acblock = ACBlock(3, 64, 6, 2, 2)
# input = torch.randn((3, 3, 64400, 6))
# output = acblock(input)
# print(output.shape)
from torch import nn
from torchvision.ops import deform_conv2d

input = torch.rand(4, 3, 10, 10)
kh, kw = 4, 4
weight = torch.rand(6, 3, kh, kw)
# offset and mask should have the same spatial size as the output
# of the convolution. In this case, for an input of 10, stride of 1
# and kernel size of 3, without padding, the output size is 8
offset = torch.rand(4, 2 * kh * kw, 7, 7)
mask = torch.rand(4, kh * kw, 8, 8)
# out = deform_conv2d(input, offset, weight, mask=None)

test = nn.Conv2d(3, 64, 3, 1, 1)
out = test(input)
print(out.shape)
