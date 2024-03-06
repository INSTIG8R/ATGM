import torch
import torch.nn as nn

from Modules.SwinTransformer import SwinTransformerBlock
from Modules.Convolution import Convolution

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, num_heads):
        super().__init__()
        self.in_channels = in_channels
        self.swin = SwinTransformerBlock(dim=in_channels, input_resolution=input_resolution, num_heads=num_heads)
        self.conv = Convolution(in_channels=in_channels,out_channels=in_channels)

    def forward(self,x,depth=2,swin=False,conv=True):
        
        if swin:
            for i in range(depth):
                x = self.swin(x)
        if conv:
            if x.shape[1]!=self.in_channels:
                x = x.transpose(2,1)
            x = self.conv(x,maxpool=False)
            if x.shape[1]==self.in_channels:
                x = x.transpose(2,1)

        return x

if __name__ == "__main__":
    bottleneck = Bottleneck(in_channels=768, out_channels=768, input_resolution=(7,7), num_heads=24)
    x = torch.rand(2,49,768)
    x_bottle = bottleneck(x)

    print(x.shape)
