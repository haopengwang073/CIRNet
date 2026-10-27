"""MobileNetV2 feature pyramid at 1/4, 1/8, 1/16 and 1/32 scale."""

import timm
import torch
import torch.nn as nn
from functools import partial
from .layers import BasicConv2d, BasicDeconv2d


class FPNLayer(nn.Module):

    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(
            chan_low,
            chan_high,
            kernel_size=4,
            stride=2,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
        )
        self.conv = BasicConv2d(
            chan_high * 2,
            chan_high * 2,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
        )

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone_name = "mobilenetv2_100"
        out_indices = (1, 2, 3, 4)
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=out_indices,
        )
        c4, c8, c16, c32 = self.backbone.feature_info.channels()
        channels = [c32, c16, c8, c4]
        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1] * 2, channels[2])
        self.fpn_layer2 = FPNLayer(channels[2] * 2, channels[3])
        self.out_conv = BasicConv2d(
            channels[3] * 2,
            channels[3] * 2,
            kernel_size=3,
            padding=1,
            norm_layer=nn.InstanceNorm2d,
        )
        self.output_channels = [
            channels[3] * 2,
            channels[2] * 2,
            channels[1] * 2,
            channels[0],
        ]

    def forward(self, images):
        c2, c3, c4, c5 = self.backbone(images)
        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p2 = self.out_conv(p2)
        return [p2, p3, p4, c5]
