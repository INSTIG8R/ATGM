import torch
import torch.nn as nn
import math

x = torch.rand(2,96,3136)

channel_conv = nn.Conv1d(in_channels=96,out_channels=3,kernel_size=1)

x_up = channel_conv(x)

print(x_up.shape)

x.view

channel_conv = nn.ConvTranspose1d(in_channels=3,out_channels=3,kernel_size=4,stride=4)

x_up = channel_conv(x_up)

print(x_up.shape)

channel_conv = nn.ConvTranspose1d(in_channels=3,out_channels=3,kernel_size=4,stride=4)

x_up = channel_conv(x_up)

print(x_up.shape)

x_up = x_up.transpose(2,1)

h = int(math.sqrt(x_up.shape[1]))

x_up = x_up.view(x_up.shape[0],h,h,x_up.shape[2])

print(x_up.shape)


# x_down = maxpool(x_added)
# print(x_down.shape)
# x_added_2 = channel_conv(x_added)


# print(x_added_2.shape)