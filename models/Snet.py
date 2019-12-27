from .modules import *


class SnetExtractor(nn.Module):
    cfg = {
        49: [24, 60, 120, 240, 512],
        146: [24, 132, 264, 528],
        535: [48, 248, 496, 992],
    }

    def __init__(self,  version = 146 ,model_path=None ,  **kwargs):

        super(SnetExtractor,self).__init__()
        num_layers = [4, 8, 4]
        self.model_path = model_path

        self.num_layers = num_layers
        channels = self.cfg[version]
        self.channels = channels



        self.conv1 = conv_bn(
            3, channels[0], kernel_size=3, stride=2,pad = 1
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )


        self.stage1 = self._make_layer(
            num_layers[0], channels[0], channels[1], **kwargs)
        self.stage2 = self._make_layer(
            num_layers[1], channels[1], channels[2], **kwargs)
        self.stage3 = self._make_layer(
            num_layers[2], channels[2], channels[3], **kwargs)
        if len(self.channels) == 5:
            self.conv5 = conv_bn(
                channels[3], channels[4], kernel_size=1, stride=1 ,pad=0 )


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._initialize_weights()

    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ShuffleV2Block(in_channels, out_channels, mid_channels=out_channels // 2, ksize=5, stride=2))
            else:
                layers.append(ShuffleV2Block(in_channels // 2, out_channels,
                                                    mid_channels=out_channels // 2, ksize=5, stride=1))
            in_channels = out_channels
        return nn.Sequential(*layers)




    def _initialize_weights(self):

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        if  self.model_path is not None:

            print("Loading pretrained weights from %s" % (self.model_path))
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)["state_dict"]
            else:
                state_dict = torch.load(
                    self.model_path, map_location=lambda storage, loc: storage)["state_dict"]
            keys = []
            for k, v in state_dict.items():
                keys.append(k)
            for k in keys:
                state_dict[k.replace("module.", "")] = state_dict.pop(k)

            self.load_state_dict(state_dict,strict = False)
            #
            # for para in self.conv1.parameters():
            #     para.requires_grad = False
            # print('extractor conv1 freezed')
            # for para in self.stage1.parameters():
            #     para.requires_grad = False
            # print('extractor stage1 freezed')



        else:
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

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)



        return c4 , c5


def snet(layer=146):
    snet_ = SnetExtractor(layer,"/mnt/data1/yanghuiyu/project/object_detect/RefineDet.PyTorch/weights/snet_146.tar")
    layer = []
    layer += [snet_.conv1 ,
              snet_.maxpool,
              snet_.stage1,
              snet_.stage2,
              snet_.stage3,
              ]
    return layer



