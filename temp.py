import torch
import torch.nn as nn

x = torch.Tensor(0.01)
print(x)

# x = torch.rand(2,10,768)

# conv = nn.Conv1d(10,3136,kernel_size=1,stride=1)
# gelu = nn.GELU()
# linear = nn.Linear(768,96)

# x_conv = conv(x)
# print(x_conv.shape)
# x_gelu = gelu(x_conv)
# print(x_conv.shape)
# x_linear = linear(x_gelu)
# print(x_linear.shape)