import torch
import torch.nn as nn
# from Modules.Unet import Unet
from matplotlib import pyplot as plt
from torchinfo import summary
import torchvision

conv_test = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
x= torch.rand(2,96,56,56)

x = conv_test(x)
print(x.shape)

# x = torch.rand(2,3,224,224)
# text = torch.rand(2,10,768)
# x, text = x.cuda(),text.cuda()

# unet = Unet(num_classes=1).cuda()
# output = unet(x,text)
# x = output.transpose(1,3)
# trans = torchvision.transforms.ToPILImage()
# out = trans(x[0])
# out.show()

# summary(unet, [(2,3,224,224),(2,10,768)])

# x = torch.randn(1, 1, 224, 224)
# trans = torchvision.transforms.ToPILImage()
# out = trans(x[0])
# out.show()


# print(output.shape)
# output = output.transpose(1,3)
# print(output.shape)

# plt.imshow(output[0,0].cpu().detach().numpy())

# plt.show()
