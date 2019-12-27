import torch.nn as nn
import torch

import torch.nn.functional as F
import functools



CEM_FILTER = 245
anchor_number = 25

def conv_bn(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )




class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):

        g = 2
        # n, c, h, w = x.size()
        # x = x.view(n, g, c // g, h, w).permute(
        #     0, 2, 1, 3, 4).contiguous().view(n, c, h, w)

        x = x.reshape(x.shape[0], g, x.shape[1] // g, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        x_proj = x[:, :(x.shape[1] // 2), :, :]
        x = x[:, (x.shape[1] // 2):, :, :]
        return x_proj, x



class Bottleneck(nn.Module):# 32 72
    def __init__(self,inp,oup):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
        nn.Conv2d(inp , oup// 4, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup// 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(oup// 4, oup// 4, 3, 2, 1, bias=False,groups= oup// 4),
        nn.BatchNorm2d(oup// 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(oup// 4, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

        self._initialize_weights()

    def forward(self, x):
        return self.bottleneck(x)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def gropy_conv_bn(inp, oup, kernel_size , stride , pad ):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False ,groups= inp ),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class Res2Block(nn.Module):
    def __init__(self,inp,scales=4):
        super(Res2Block, self).__init__()

        self.inp = inp
        self.scale_channel = inp // scales

        self.conv1   = conv_bn(inp,self.scale_channel,1,1,0)

        self.convg_1 = gropy_conv_bn(self.scale_channel//4,self.scale_channel//4,3,1,1)
        self.convg_2 = gropy_conv_bn(self.scale_channel//4,self.scale_channel//4,3,1,1)
        self.convg_3 = gropy_conv_bn(self.scale_channel//4,self.scale_channel//4,3,1,1)

        self.conv2   = conv_bn(self.scale_channel, inp, 1, 1, 0)
        self.out_channels = inp
        self._initialize_weights()

    def forward(self, x_input):
        x = self.conv1(x_input)
        cut = self.scale_channel//4
        x1,x2,x3,x4 = x[:, :cut, :, :],x[:, cut:cut*2, :, :],x[:, cut*2:cut*3, :, :],x[:, cut*3:, :, :]
        g1 = self.convg_1(x2)
        g2_in = g1 + x3
        g2 = self.convg_2(g2_in)
        g3_in = g2 + x4
        g3 = self.convg_3(g3_in)
        x =  torch.cat((x1,g1,g2,g3),1)
        out  = self.conv2(x)
        return out +  x_input

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



