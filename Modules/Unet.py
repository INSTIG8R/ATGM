import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchinfo import summary

# from BERT import BERT
from Modules.PatchEmbed import PatchEmbed
from Modules.GuideEncoder import GuideEncoder
from Modules.ResSwin import ResSwin
from Modules.Bottleneck import Bottleneck
from Modules.AttentionGate import AttentionGate
from Modules.Deconvolution import Deconvolution
from Modules.FinalOutput import FinalOutput

class Unet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        # self.BERT = BERT()
        self.PatchEmbed = PatchEmbed()
        self.GuideEncoder = GuideEncoder(in_channels=96, text_len=3136)
        self.ResSwin1 = ResSwin(input_dim=96,output_dim=192,input_resolution=(56,56),num_heads=3,depth=2)
        self.ResSwin2 = ResSwin(input_dim=192,output_dim=384,input_resolution=(28,28),num_heads=6,depth=6)
        self.ResSwin3 = ResSwin(input_dim=384,output_dim=768,input_resolution=(14,14),num_heads=12,depth=6)
        self.Bottleneck = Bottleneck(in_channels=768, out_channels=768, input_resolution=(7,7), num_heads=24)
        self.AttentionGate3 = AttentionGate(gate_input_channel=768,skip_input_channel=384)
        self.AttentionGate2 = AttentionGate(gate_input_channel=384,skip_input_channel=192)
        self.AttentionGate1 = AttentionGate(gate_input_channel=192,skip_input_channel=96)
        self.Decoder1 = Deconvolution(input_channel=768,skip_input_channel=384)
        self.Decoder2 = Deconvolution(input_channel=384,skip_input_channel=192)
        self.Decoder3 = Deconvolution(input_channel=192,skip_input_channel=96)
        self.FinalOutput = FinalOutput(in_channel=96, out_channel=num_classes)

    def forward(self,x,text):
        text_embed = text
        x_embed = self.PatchEmbed(x)
        x_guide = self.GuideEncoder(x_embed,text_embed)
        x_res1 = self.ResSwin1(x_guide)
        x_res2 = self.ResSwin2(x_res1)
        x_res3 = self.ResSwin3(x_res2)
        x_bottle = self.Bottleneck(x_res3)
        skip_2 = self.AttentionGate3(x_bottle,x_res2)
        x_dec1 = self.Decoder1(x_res3,skip_2)
        skip_1 = self.AttentionGate2(x_dec1,x_res1)
        x_dec2 = self.Decoder2(x_dec1,skip_1)
        skip_0 = self.AttentionGate1(x_dec2,x_embed)       
        x_dec3 = self.Decoder3(x_dec2,skip_0)
        x_out = self.FinalOutput(x_dec3)

        return x_out




if __name__ =="__main__":
    x = torch.rand(2,3,224,224)
    text = torch.rand(2,10,768)
    x, text = x.cuda(),text.cuda()

    unet = Unet(num_classes=1).cuda()

    # summary(unet, [(2,3,224,224),(2,10,768)])

    output = unet(x,text)
    print(output.shape)
    output = output.transpose(1,3)
    print(output.shape)


    plt.imshow(output[0,0].cpu().detach().numpy())
    plt.show()
