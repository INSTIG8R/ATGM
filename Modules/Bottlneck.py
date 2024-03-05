import torch
import torch.nn as nn

from SwinTransformer import SwinTransformerBlock
from Convolution import Convolution

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, num_heads):
        super().__init__()
        self.swin = SwinTransformerBlock(dim=in_channels, input_resolution=input_resolution, num_heads=num_heads)
        self.conv = Convolution(in_channels=in_channels,out_channels=in_channels)

    def forward(self,x,depth=2,swin=False,conv=False):
        if swin:
            for i in range(depth):
                x = self.swin(x)
        if conv:
            x = self.conv(x)

        return x

if __name__ == "__main__":
    bottleneck = Bottleneck(in_channels=768, out_channels=768, input_resolution=(7,7), num_heads=24)
    x = torch.rand(2,49,768)
    x_bottle = bottleneck(x,conv=True)

    print(x.shape)
