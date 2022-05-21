# print(-4//2)

import torch

# conv  = torch.nn.Conv2d(3, 64, 3, 2, 1)
#
# input = torch.rand((3, 3, 32, 32))
#
# outoput = conv(input)
#
# print(outoput.shape)
from torch import nn

test = nn.PixelShuffle(2)
input = torch.rand(1, 1024, 20, 20)

outoput = test(input)
print(outoput.shape)
