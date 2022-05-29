# print(-4//2)

import torch
from torch import nn

from utils.aws.resume import f

# w = torch.zeros(3, 5)
# print(w)
#
# nn.init.uniform_(w)
# print(w)

# for i in range(3):
#     print(i)

x = torch.rand(3, 3, 2)
x = x.view(2, 9)
print(x)
