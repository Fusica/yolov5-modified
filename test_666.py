import torch

# test = torch.nn.GLU(1)

x = torch.rand(1, 128, 160, 160)

x = torch.chunk(x, 2, 1)

# x = test(x)
# print(x.shape)

# x = torch.split(x, [48, 48, 32], dim=1)
print(x[0].shape)
