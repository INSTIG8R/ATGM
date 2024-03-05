import torch
import torch.nn as nn

x = torch.rand(2,768,49)
y = torch.rand(2,384,196)

channel_conv = nn.Conv1d(in_channels=384,out_channels=384,kernel_size=1)
maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

x_up = channel_conv(y)

print(x_up.shape)

# x_down = maxpool(x_added)
# print(x_down.shape)
# x_added_2 = channel_conv(x_added)


# print(x_added_2.shape)