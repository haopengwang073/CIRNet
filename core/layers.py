import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        norm_layer=None,
        act_layer=None,
        **kwargs,
    ):
        super(BasicConv2d, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                **kwargs,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class BasicDeconv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        norm_layer=None,
        act_layer=None,
        **kwargs,
    ):
        super(BasicDeconv2d, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                **kwargs,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class ContextConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)))


class Conv2xUp(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(Conv2xUp, self).__init__()
        self.conv1 = BasicDeconv2d(
            in_channels,
            out_channels,
            norm_layer=norm_layer,
            act_layer=nn.LeakyReLU,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv2 = BasicConv2d(
            out_channels * 2,
            out_channels * 2,
            norm_layer=norm_layer,
            act_layer=nn.LeakyReLU,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        x = torch.cat((x, rem), 1)
        x = self.conv2(x)
        return x


def context_upsample(disp_low, up_weights, scale_factor=4):
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low, kernel_size=3, dilation=1, padding=1)
    disp_unfold = disp_unfold.reshape(b, -1, h, w)
    disp_unfold = F.interpolate(
        disp_unfold, (h * scale_factor, w * scale_factor), mode="nearest"
    )
    disp = (disp_unfold * up_weights).sum(1)
    return disp


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)
