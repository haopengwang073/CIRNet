"""Lightweight 2D regularization of the 1/8 geometry volume."""

import torch.nn as nn
import torch.nn.functional as F
from .layers import BasicConv2d


class GeometryAggregation(nn.Module):

    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(GeometryAggregation, self).__init__()
        self.left_att = left_att
        self.expanse_ratio = expanse_ratio
        conv0 = [
            MobileV2Residual(
                in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio
            )
            for i in range(blocks[0])
        ]
        self.conv0 = nn.Sequential(*conv0)
        self.conv1 = MobileV2Residual(
            in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio
        )
        conv2_add = [
            MobileV2Residual(
                in_channels * 2,
                in_channels * 2,
                stride=1,
                expanse_ratio=self.expanse_ratio,
            )
            for i in range(blocks[1] - 1)
        ]
        self.conv2 = nn.Sequential(*conv2_add)
        self.conv3 = MobileV2Residual(
            in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio
        )
        conv4_add = [
            MobileV2Residual(
                in_channels * 4,
                in_channels * 4,
                stride=1,
                expanse_ratio=self.expanse_ratio,
            )
            for i in range(blocks[2] - 1)
        ]
        self.conv4 = nn.Sequential(*conv4_add)
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 4,
                in_channels * 2,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 2),
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 2,
                in_channels,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.redir1 = MobileV2Residual(
            in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio
        )
        self.redir2 = MobileV2Residual(
            in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio
        )
        if self.left_att:
            self.att0 = GuidedCostVolumeExcitation(in_channels, backbone_channels[1])
            self.att2 = GuidedCostVolumeExcitation(
                in_channels * 2, backbone_channels[2]
            )
            self.att4 = GuidedCostVolumeExcitation(
                in_channels * 4, backbone_channels[3]
            )

    def forward(self, x, features_left):
        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[1])
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.left_att:
            conv2 = self.att2(conv2, features_left[2])
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.att4(conv4, features_left[3])
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return [conv6]


class MobileV2Residual(nn.Module):

    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation
        self.pwconv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride,
                pad,
                dilation=dilation,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        feat = self.pwconv(x)
        feat = self.dwconv(feat)
        feat = self.pwliner(feat)
        if self.use_res_connect:
            return x + feat
        else:
            return feat


class GuidedCostVolumeExcitation(nn.Module):

    def __init__(self, cv_chan, im_chan):
        super(GuidedCostVolumeExcitation, self).__init__()
        self.im_att = nn.Sequential(
            BasicConv2d(
                im_chan,
                im_chan,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            nn.Conv2d(im_chan, cv_chan, 1),
        )

    def forward(self, cv, im):
        im_att = self.im_att(im)
        return im_att * cv
