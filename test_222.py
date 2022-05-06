print(-4//2)

import torch

conv  = torch.nn.Conv2d(3, 64, 3, 2, 1)

input = torch.rand((3, 3, 32, 32))

outoput = conv(input)

print(outoput.shape)

