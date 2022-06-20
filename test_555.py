# window = {3: 2, 5: 3, 7: 3}
#
# head_splits = []
#
# for cur_window, cur_head_split in window.items():
#     head_splits.append(cur_head_split)
#
# print(head_splits)
import torch
import torch.nn as nn

#
#
# x1 = torch.rand(1, 8, 400, 16)
# x2 = torch.rand(1, 8, 400, 16)
#
# x3 = x1*x2
# x3 = F.pad(x3, (0, 0, 1, 0, 0, 0))
#
# print(x3.shape)

n_filter_list = (3, 48, 96, 192, 384)

test = nn.Sequential(
    *[nn.Sequential(
        nn.Conv2d(in_channels=n_filter_list[i],
                  out_channels=n_filter_list[i + 1],
                  kernel_size=3,  # hardcoding for now because that's what the paper used
                  stride=2,  # hardcoding for now because that's what the paper used
                  padding=1),  # hardcoding for now because that's what the paper used
    )
        for i in range(len(n_filter_list) - 1)
    ])

test.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1],
                                            out_channels=796,
                                            stride=1,  # hardcoding for now because that's what the paper used
                                            kernel_size=1,
                                            # hardcoding for now because that's what the paper used
                                            padding=0))  # hardcoding for now because that's what the paper used

x = torch.rand(1, 3, 160, 160)

x = test(x)
print(x.shape)
