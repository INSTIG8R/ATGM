import torch.nn as nn

from SwinTransformer import SwinTransformerBlock
from PatchMerging import PatchMerging

class ResSwin(nn.Module):
    def __init__(self,input_dim,output_dim,input_resolution,num_heads,depth):
        super().__init__()
        self.depth = depth
        self.swin_block = SwinTransformerBlock(dim=input_dim, input_resolution=input_resolution, num_heads=num_heads)
        self.patch_merging = PatchMerging(dim=input_dim, input_resolution=input_resolution)
        

    def forward(self,x):
        input = x
        for i in range(self.depth):
            block_output = self.swin_block(x)
        input_merged = self.patch_merging(block_output)
        


