# Editor: Max

# Create Time: 4/26/22 18:55


# p, d, k, s = 1, 1, 1, 1
# for p in range(1, 10):
#     for d in range(1, 10):
#         for k in range(1, 9):
#             for s in range(1, 8):
#                 if 20 == (160 + 2 * p - d * (k - 1) - 1)/s + 1:
#                     print(p, d, k, s)

import torch
from torch import nn

m = nn.AdaptiveMaxPool2d((5,7))
x = nn.Conv2d(64, 128, 1, 1)
input = torch.randn(1, 64, 8, 9)
output = x(input)
print(output.shape)
