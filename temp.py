import torch
import torch.nn as nn

x = torch.rand(2,96,3136)
y = torch.rand(2,784,192)

channel_conv = nn.Conv1d(in_channels=96,out_channels=192,kernel_size=3,padding=1)
maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

x_added = channel_conv(x)

print(x_added.shape)

x_down = maxpool(x_added)
print(x_down.shape)
# x_added_2 = channel_conv(x_added)


# print(x_added_2.shape)