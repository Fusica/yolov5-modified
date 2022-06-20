import torch

test = torch.nn.Linear(128, 128)

x = torch.rand(1, 128, 160, 160)

# x = test(x)
# print(x.shape)

# x = torch.split(x, [48, 48, 32], dim=1)
print(x.shape[1])
