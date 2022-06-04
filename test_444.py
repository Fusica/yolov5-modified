# print(-4//2)

import torch
from torch import nn

# test = nn.Sequential(
#     nn.Conv2d(3, 3, 3, 1, 1),
#     nn.Conv2d(3, 3, 3, 1, 1),
#     nn.Conv2d(3, 3, 3, 1, 1)
# )
#
# for conv in test:
#     b = conv.bias.view(1, -1)
#     b.data.fill_(1)
#     conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#
# print(b[0])

# values = ['a', 'b', 'c']
#
# for i, j in enumerate(values):
#     print(i, j)


# test = nn.Linear(96, 3*96)
#
# input = torch.rand(1, 400, 96)
#
# qkv = test(input).reshape(1, -1, 3, 96).permute(2, 0, 1, 3)
#
# print(qkv[2].shape)

x = torch.rand(1, 128, 160, 160)
# x = torch.chunk(x, 4, dim=1)
# print(x[1].shape)

# x = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3).unsqueeze(3).transpose(0, 3).reshape(1, 128, 160, 160)
# print(x.shape)

q = torch.rand(1, 128, 40, 20)
k = torch.rand(1, 128, 40, 20)
v = torch.rand(1, 128, 40, 20)

content = torch.matmul(q.permute(0, 1, 3, 2), k)
c1, c2, c3, c4 = content.size()

print(c3)
