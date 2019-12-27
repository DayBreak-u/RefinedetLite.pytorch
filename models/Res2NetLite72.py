import torch.nn as  nn
import torch.nn.functional as F
from  .modules import  Bottleneck,Res2Block,conv_bn
import torch




def Res2NetLite(base_channel = 32):

    layers = []
    stage_numbers = [4,8,4]

    layers  +=  [conv_bn(
        3, base_channel, kernel_size=3, stride=2,pad = 1
    )]
    layers  +=  [nn.MaxPool2d(
        kernel_size=3, stride=2, padding=1,
    )]

    c = 72
    input_channel = base_channel
    output_channel = c * 4
    for i,s in enumerate(stage_numbers):
        temp_layers = []
        temp_layers +=  [Bottleneck(input_channel,output_channel)]
        for j in range(1,s):
            temp_layers += [Res2Block(output_channel)]

        layers += nn.Sequential(*temp_layers)
        input_channel = output_channel
        output_channel = output_channel*2


    return layers




