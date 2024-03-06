import torch
import torch.nn as nn
import math

def Reshape2bchw(x):
    h = int(math.sqrt(x.shape[1]))
    x = x.reshape(x.shape[0],x.shape[2],h,h)

    return x