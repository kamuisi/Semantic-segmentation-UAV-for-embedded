import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                stride=stride, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
       x = self.relu(self.dw_bn(self.depthwise(x)))
       x = self.relu(self.pw_bn(self.pointwise(x)))
       return x

class FeatureFusionModule(nn.Module):
    def __init__(self, high_res_channels, low_res_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

        self.high_res_conv = nn.Sequential(
            nn.Conv2d(high_res_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.low_res_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.dwconv = nn.Sequential(
            nn.Conv2d(low_res_channels, out_channels, kernel_size=3, padding=scale_factor,
                      dilation=scale_factor, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_res_input, low_res_input):
        high = self.high_res_conv(high_res_input)
        low = F.interpolate(low_res_input, scale_factor=self.scale_factor,
                            mode='bilinear', align_corners=False)
        low = self.dwconv(low)
        low = self.low_res_conv(low)

        out = high + low
        return self.relu(out)

class Fast_SCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.mobilenet_v2(weights='DEFAULT')
        feature = list(backbone.features.children())

        self.downsample = nn.Sequential(*feature[:7])

        self.global_feature = nn.Sequential(*feature[7:])

        self.reduce_channels = nn.Conv2d(feature[-1].out_channels, 128, kernel_size=1, bias=False)

        self.fusion = FeatureFusionModule(feature[6].out_channels, 128, 128, 4)

        self.classifier = nn.Sequential(
            DepthwiseSeperableConv(128, 128),
            DepthwiseSeperableConv(128, 128),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.size()[2:]

        down = self.downsample(x)

        global_feat = self.global_feature(down)
        global_feat = self.reduce_channels(global_feat)

        fused = self.fusion(down, global_feat)

        out = self.classifier(fused)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out