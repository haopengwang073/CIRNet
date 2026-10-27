"""Compact Iterative Refinement Network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    BasicConv2d,
    ContextConv2d,
    Conv2xUp,
    context_upsample,
    disparity_regression,
)
from .backbone import Backbone
from .aggregation import GeometryAggregation
from .cce import CompactCorrelationEncoding, correlation_volume
from .refinement import CoarseCorrectionUnit, FineRefinementUnit


class CIRNet(nn.Module):

    def __init__(self, iterations=4):
        super().__init__()
        self.iterations = iterations
        self.max_disp = 192
        self.left_att = True

        self.backbone = Backbone()

        self.stem2x_channel = 32
        self.stem4x_channel = 48
        self.stem8x_channel = 64
        self.stem_2 = nn.Sequential(
            BasicConv2d(
                3,
                self.stem2x_channel,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BasicConv2d(
                self.stem2x_channel,
                self.stem2x_channel,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.stem_4 = nn.Sequential(
            BasicConv2d(
                self.stem2x_channel,
                self.stem4x_channel,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BasicConv2d(
                self.stem4x_channel,
                self.stem4x_channel,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.stem_8 = nn.Sequential(
            BasicConv2d(
                self.stem4x_channel,
                self.stem8x_channel,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            BasicConv2d(
                self.stem8x_channel,
                self.stem8x_channel,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.match_feature_channel = 96
        self.conv4x = BasicConv2d(
            self.backbone.output_channels[0] + self.stem4x_channel,
            self.match_feature_channel,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.LeakyReLU,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv8x = BasicConv2d(
            self.backbone.output_channels[1] + self.stem8x_channel,
            self.match_feature_channel,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.LeakyReLU,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.desc = nn.Conv2d(
            self.match_feature_channel,
            self.match_feature_channel,
            kernel_size=1,
            padding=0,
            stride=1,
        )

        self.geometry_aggregation = GeometryAggregation(
            in_channels=self.max_disp // 8,
            left_att=self.left_att,
            blocks=[4, 8, 16],
            expanse_ratio=4,
            backbone_channels=self.backbone.output_channels,
        )

        self.hidden_dim_4x = 64
        self.hidden_dim_8x = 96
        self.corr_radius = 4
        self.cnet_4x = ContextConv2d(
            self.backbone.output_channels[0] + self.stem4x_channel,
            self.hidden_dim_4x,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.hnet_4x = nn.Sequential(
            ContextConv2d(
                self.backbone.output_channels[0] + self.stem4x_channel,
                self.hidden_dim_4x,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(self.hidden_dim_4x, self.hidden_dim_4x, 3, 1, 1, bias=False),
        )
        self.context_zqr_conv_4x = nn.Conv2d(
            self.hidden_dim_4x, self.hidden_dim_4x * 3, 3, padding=3 // 2
        )
        self.hnet_8x = nn.Sequential(
            ContextConv2d(
                self.backbone.output_channels[1] + self.stem8x_channel,
                self.hidden_dim_8x,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(self.hidden_dim_8x, self.hidden_dim_8x, 3, 1, 1, bias=False),
        )

        self.coarse_correction = CoarseCorrectionUnit()
        self.fine_refinement = FineRefinementUnit()

        self.spx_2_gru = Conv2xUp(32, 32, norm_layer=nn.BatchNorm2d)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1)
        )
        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1)
        )
        self.spx_2 = Conv2xUp(24, 32, norm_layer=nn.BatchNorm2d)
        self.spx_4 = nn.Sequential(
            BasicConv2d(
                self.match_feature_channel,
                24,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BasicConv2d(
                24,
                24,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.register_buffer(
            "mean", torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255
        )
        self.register_buffer(
            "std", torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255
        )
        self.eval()

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("iterations must be a non-negative integer")
        self._iterations = value

    def upsample_disp(self, disparity, mask_features, stem_2x):
        features = self.spx_2_gru(mask_features, stem_2x)
        weights = F.softmax(self.spx_gru(features), 1)
        return context_upsample(disparity * 4.0, weights).unsqueeze(1)

    def forward(self, left, right):
        if left.ndim != 4 or left.shape != right.shape or left.shape[1] != 3:
            raise ValueError("left and right must have equal shape [B, 3, H, W]")
        if (
            left.shape[0] < 1
            or min(left.shape[-2:]) < 32
            or any(s % 32 for s in left.shape[-2:])
        ):
            raise ValueError("image height and width must be positive multiples of 32")
        if not left.is_floating_point() or not right.is_floating_point():
            raise ValueError("images must be floating point RGB tensors in [0, 255]")
        if left.device != right.device or left.dtype != right.dtype:
            raise ValueError("left and right must have the same device and dtype")

        left = ((left - self.mean) / self.std).contiguous()
        right = ((right - self.mean) / self.std).contiguous()
        h, w = left.shape[-2:]

        # Feature extraction: shared backbone and parallel convolutional stem.
        images = torch.cat([left, right], 0)

        features = self.backbone(images)
        features_left = [torch.chunk(f, 2, dim=0)[0] for f in features]
        features_right = [torch.chunk(f, 2, dim=0)[1] for f in features]

        stem2 = self.stem_2(images)
        stem4 = self.stem_4(stem2)
        stem8 = self.stem_8(stem4)
        stem_2x, _ = torch.chunk(stem2, 2, dim=0)
        stem_4x, stem_4y = torch.chunk(stem4, 2, dim=0)
        stem_8x, stem_8y = torch.chunk(stem8, 2, dim=0)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        feat_left_4x = self.desc(self.conv4x(features_left[0]))
        feat_right_4x = self.desc(self.conv4x(features_right[0]))

        feat_left_8x = torch.cat((features_left[1], stem_8x), 1)
        feat_right_8x = torch.cat((features_right[1], stem_8y), 1)

        match_left_8x = self.desc(self.conv8x(feat_left_8x))
        match_right_8x = self.desc(self.conv8x(feat_right_8x))

        # CCE geometry branch and soft-argmax initialization (paper III-B).
        correlation = correlation_volume(
            match_left_8x, match_right_8x, self.max_disp // 8
        )
        geometry = self.geometry_aggregation(correlation, features_left)[0]
        init_disp = disparity_regression(F.softmax(geometry, dim=1), self.max_disp // 8)
        disp = F.interpolate(init_disp * 2.0, (h // 4, w // 4), mode="nearest")

        if self.iterations == 0:
            features = self.spx_2(self.spx_4(features_left[0]), stem_2x)
            weights = F.softmax(self.spx(features), 1)
            return context_upsample(disp * 4.0, weights).unsqueeze(1)

        hidden = torch.tanh(self.hnet_4x(features_left[0]))
        context = self.cnet_4x(features_left[0])
        context = list(
            self.context_zqr_conv_4x(context).split(self.hidden_dim_4x, dim=1)
        )
        coarse_hidden = torch.tanh(self.hnet_8x(feat_left_8x))
        lookup = CompactCorrelationEncoding(
            feat_left_4x, feat_right_4x, geometry.unsqueeze(1), radius=self.corr_radius
        )

        # SIR: one coarse correction, then a shared fine refinement unit.
        for step in range(self.iterations):
            evidence = lookup(disp)
            if step == 0:
                hidden, mask_features, disp_delta = self.coarse_correction(
                    hidden, coarse_hidden, context, evidence, disp
                )
            else:
                hidden, mask_features, disp_delta = self.fine_refinement(
                    hidden, context, evidence, disp
                )
            disp = disp + disp_delta
        
        return self.upsample_disp(disp, mask_features, stem_2x)
