import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc_refinedet, coco_refinedet
import os

from .Snet import snet
from .modules import  conv_bn,Bottleneck,gropy_conv_bn,InvertedResidual

class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base res2netlite network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: res2netlite layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, ARM, ODM, TCB, num_classes):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # print((coco_refinedet, voc_refinedet)[num_classes == 21])
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes == 21]
        print(self.cfg)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.res2netlite = nn.ModuleList(base)
        # self.res2netlite = base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv4_3_L2Norm = L2Norm(576, 10)
        self.conv5_3_L2Norm = L2Norm(1152, 8)
        self.extras = extras


        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()

        # apply res2netlite
        for k in range(4):
            # print(self.res2netlite[k])
            x = self.res2netlite[k](x)
            if 3 == k:
                # s = self.conv4_3_L2Norm(x)
                sources.append(x)

        # apply res2netlite
        for k in range(4, len(self.res2netlite)):
            x = self.res2netlite[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            sources.append(x)


        # apply ARM and ODM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        #print([x.size() for x in sources])
        # calculate TCB features
        #print([x.size() for x in sources])
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(3-k)*3 + i](s)
            # s = self.tcb0[3 - k](s)
                #print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3-k](u)

                s += u
            # for i in range(3):
            #     s = self.tcb2[(3-k)*3 + i](s)
            # for i in range(1):
            s = self.tcb2[3 - k](s)

            p = s
            tcb_source.append(s)
        #print([x.size() for x in tcb_source])
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            # print(x.shape)
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())

        if self.phase == "test":
            #print(loc, conf)
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                             2)),                               # arm conf preds
                odm_loc.view(odm_loc.size(0), -1, 4),           # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1,
                             self.num_classes)),                # odm conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')




class Light_head(nn.Module):

    def __init__(self,inp,c,num):
        super(Light_head, self).__init__()
        self.conv1 = conv_bn(inp,256,1,1,0)
        self.conv2 = conv_bn(inp,128,1,1,0)
        # self.conv3 = gropy_conv_bn(128,128,3,1,1)
        self.conv3 = conv_bn(128,128,3,1,1)
        self.conv4 = conv_bn(128,256,1,1,0)


        self.pre  =  nn.Conv2d(256,
              c * num, kernel_size=1, padding=0)

        self._initialize_weights()
    def forward(self, x):
        x1 = F.relu( self.conv1(x))
        x2 = F.relu( self.conv2(x))
        x3 = F.relu( self.conv3(x2))
        x4 = F.relu( self.conv4(x3))

        x = x1 + x4


        return self.pre(x)


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




def arm_multibox( extra_layers, cfg ,tcb):
    arm_loc_layers = []
    arm_conf_layers = []
    res2netlite_source = [ 3, 4]
    for k, v in enumerate(res2netlite_source):

        arm_loc_layers += [ Light_head(tcb[k], cfg[k] , 4 )]
        arm_conf_layers += [Light_head(tcb[k], cfg[k] , 2) ]

    for k , extra_layer in  enumerate(extra_layers,2):
        arm_loc_layers += [  Light_head(tcb[k] , cfg[k] , 4)]
        arm_conf_layers += [Light_head(tcb[k] , cfg[k] , 2)]
        # input *= 2
    return (arm_loc_layers, arm_conf_layers)

def odm_multibox( extra_layers, cfg, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    res2netlite_source = [ 3, 4]
    for k, v in enumerate(res2netlite_source):
        odm_loc_layers += [Light_head(256 , cfg[k] , 4)]
        odm_conf_layers += [Light_head(256 , cfg[k] , num_classes)]

    for k , extra_layer in  enumerate(extra_layers,2):
        odm_loc_layers += [Light_head(256 , cfg[k] , 4)]
        odm_conf_layers += [Light_head(256 , cfg[k] , num_classes)]


    return (odm_loc_layers, odm_conf_layers)

def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [ nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]

        # feature_scale_layers += [nn.Conv2d(cfg[k], cfg[k], 3, padding=1,groups=cfg[k]),
        #                          nn.ReLU(inplace=True),
        #                          nn.Conv2d(cfg[k], 256, 1, padding=0)
        #                          ]

        feature_pred_layers += [nn.ReLU(inplace=True),
                                # nn.Conv2d(256, 256, 3, padding=1),
                                # nn.ReLU(inplace=True)
        ]

        if k != len(cfg) - 1:
            if k == len(cfg) - 2:
                # nn.Sequential(nn.Conv2d(256, 256, 1, padding=0),
                feature_upsample_layers +=  [nn.UpsamplingBilinear2d((5,5))]
            else:
                # feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
                feature_upsample_layers += [nn.UpsamplingBilinear2d(scale_factor = 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)




def build_refinedet(phase, size=320, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only RefineDet320 and RefineDet512 is supported!")
        return

    cfg = (coco_refinedet, voc_refinedet)[num_classes == 21]
    mbox = cfg["mbox"]  # number of boxes per feature map location

    tcb = [264, 528, 512, 512]


    base_ = snet(146)

    extras_ = [Bottleneck(528, 512), Bottleneck(512, 512)]

    ARM_ = arm_multibox( extras_, mbox , tcb)
    ODM_ = odm_multibox( extras_, mbox, num_classes)

    TCB_ = add_tcb(tcb)
    return RefineDet(phase, size, base_, extras_, ARM_, ODM_, TCB_, num_classes)
