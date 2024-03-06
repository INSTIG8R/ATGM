import torch
import torch.nn as nn

from Modules.Reshape2bchw import Reshape2bchw

class Convolution(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
         
    def forward(self,x,maxpool=True):
        x = Reshape2bchw(x)
        # if x.shape[1]!=self.in_channels:
        #     x = x.transpose(2,1)
        x_conv = self.conv(x)
        if maxpool==True:
            x_conv = self.maxpool(x_conv)
        # print(x_conv.shape)
        b,c,h,w = x_conv.shape
        l = int(h**2)
        
        x_conv = x_conv.reshape(b,l,c)

        # if x_conv.shape[1]==self.out_channels:
        #     x_conv = x_conv.transpose(2,1)

        return x_conv
    
if __name__=="__main__":
    conv = Convolution(96,192)
    x = torch.rand(2,3136,96)
    x_conv = conv(x)

    print(x_conv.shape) # 2,784,192