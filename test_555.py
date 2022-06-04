# window = {3: 2, 5: 3, 7: 3}
#
# head_splits = []
#
# for cur_window, cur_head_split in window.items():
#     head_splits.append(cur_head_split)
#
# print(head_splits)
import torch
import torch.nn.functional as F


x1 = torch.rand(1, 8, 400, 16)
x2 = torch.rand(1, 8, 400, 16)

x3 = x1*x2
x3 = F.pad(x3, (0, 0, 1, 0, 0, 0))

print(x3.shape)
