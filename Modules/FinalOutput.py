import torch
import torch.nn as nn
import math

class FinalOutput(nn.Module):
    def __init__(self, in_channel, out_channel):
        self.in_channel =  in_channel
        self.out_channel = out_channel
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,out_channels=out_channel,kernel_size=1),
            nn.ConvTranspose1d(in_channels=out_channel,out_channels=out_channel,kernel_size=4,stride=4),
            nn.ConvTranspose1d(in_channels=out_channel,out_channels=out_channel,kernel_size=4,stride=4)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        if x.shape[1]!=self.in_channel:
            x = x.transpose(2,1)
        x = self.conv(x)
        if x.shape[1]==self.out_channel:
            x = x.transpose(2,1)
        x = self.sigmoid(x)
        if len(x.shape)==3:
            h = int(math.sqrt(x.shape[1]))
            x = x.view(x.shape[0],h,h,x.shape[2])
        
        return x
    
# if __name__ == "__main__":
#     output = FinalOutput(in_channel=96, out_channel=1)

#     x = torch.rand(2,3136,96)

#     out = output(x)

#     print(out.shape) # 2,224,224,3