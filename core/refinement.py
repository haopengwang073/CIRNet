"""Staged Iterative Refinement (SIR): one CCU followed by shared FRU updates."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DispHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )

    def forward(self, h, c, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + c[0])
        r = torch.sigmoid(self.convr(hx) + c[1])
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + c[2])
        h = (1 - z) * h + z * q
        return h


class MotionEncoder(nn.Module):

    def __init__(self, branch_channels, output_channels):
        super().__init__()
        cor_planes = 18
        self.convc1 = nn.Conv2d(cor_planes, branch_channels, 1, padding=0)
        self.convc2 = nn.Conv2d(branch_channels, branch_channels, 3, padding=1)
        self.convd1 = nn.Conv2d(1, branch_channels, 7, padding=3)
        self.convd2 = nn.Conv2d(branch_channels, branch_channels, 3, padding=1)
        self.conv = nn.Conv2d(2 * branch_channels, output_channels - 1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))
        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)


class CoarseCorrectionUnit(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = MotionEncoder(64, 96)
        self.local_gru = ConvGRU(64, 192)
        self.disp_head = DispHead(64, 128, 1)
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, hidden, coarse_hidden, context, correlation, disparity):
        motion = self.encoder(disparity, correlation)
        with torch.autocast(device_type=coarse_hidden.device.type, enabled=False):
            coarse = F.interpolate(
                coarse_hidden.float(),
                motion.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        coarse = coarse.to(coarse_hidden.dtype)
        hidden = self.local_gru(hidden, context, torch.cat([motion, coarse], 1))
        return hidden, self.mask_feat_4(hidden), self.disp_head(hidden)


class FineRefinementUnit(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = MotionEncoder(48, 64)
        self.local_gru = ConvGRU(64, 64)
        self.disp_head = DispHead(64, 128, 1)
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, hidden, context, correlation, disparity):
        motion = self.encoder(disparity, correlation)
        hidden = self.local_gru(hidden, context, motion)
        return hidden, self.mask_feat_4(hidden), self.disp_head(hidden)
