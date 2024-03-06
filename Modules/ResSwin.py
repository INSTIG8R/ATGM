import torch
import torch.nn as nn

from Modules.SwinTransformer import SwinTransformerBlock
from Modules.PatchMerging import PatchMerging
from Modules.Convolution import Convolution

class ResSwin(nn.Module):
    def __init__(self,input_dim,output_dim,input_resolution,num_heads,depth):
        super().__init__()
        self.depth = depth
        self.swin_block = SwinTransformerBlock(dim=input_dim, input_resolution=input_resolution, num_heads=num_heads)
        self.patch_merging = PatchMerging(dim=input_dim, input_resolution=input_resolution)
        self.convolution = Convolution(in_channels=input_dim,out_channels=output_dim)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        for i in range(self.depth):
            block_output = self.swin_block(x)
        input_patch_merged = self.patch_merging(block_output)
        input_convolution = self.convolution(x)
        input_joined = input_patch_merged + input_convolution
        output = self.relu(input_joined)

        return output
    
# if __name__=="__main__":
#     res_swin = ResSwin(input_dim=96,output_dim=192,input_resolution=(56,56),num_heads=3,depth=2)
#     x = torch.rand(2,3136,96)

#     out = res_swin(x)

#     print(out.shape) # 2,784,192



        


