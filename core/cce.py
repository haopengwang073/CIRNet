"""CCE: 1/4 local all-pairs evidence and 1/8 geometry evidence."""

import torch
import torch.nn.functional as F


def correlation_volume(left_feature, right_feature, max_disp):
    """
    Correlation volume computation

    Args:
        left_feature: Left image features [B, C, H, W]
        right_feature: Right image features [B, C, H, W]
        max_disp: Maximum disparity value

    Returns:
        cost_volume: Correlation cost volume [B, max_disp, H, W]
    """
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (
                left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]
            ).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume


def sample_correlation(image, coordinates):
    """Linear lookup along a correlation row, with zero outside its bounds."""
    width = image.shape[-1]
    x, y = coordinates.split([1, 1], dim=-1)
    grid = torch.cat([2 * x / (width - 1) - 1, y], dim=-1)
    return F.grid_sample(image, grid, align_corners=True)


class CompactCorrelationEncoding:

    def __init__(self, init_fmap1, init_fmap2, geo_volume, radius=4):
        self.radius = radius
        
        init_corr = CompactCorrelationEncoding.corr(init_fmap1, init_fmap2)
        
        B, H, W, _, W2 = init_corr.shape
        self.init_corr = init_corr.reshape(B * H * W, 1, 1, W2)
        
        Bg, Cg, Dg, Hg, Wg = geo_volume.shape
        assert Bg == B
        geo_volume_2d = geo_volume.view(Bg, Cg * Dg, Hg, Wg)
        geo_volume_2d = F.interpolate(geo_volume_2d, scale_factor=2)
        _, _, H_up, W_up = geo_volume_2d.shape
        assert H_up == H and W_up == W
        geo_volume_up = geo_volume_2d.view(Bg, Cg, Dg, H_up, W_up)
        geo_volume_up = geo_volume_up.permute(0, 3, 4, 1, 2)
        self.geo_volume = geo_volume_up.reshape(
            Bg * H_up * W_up, Cg, 1, Dg
        ).contiguous()
        
        b, c, h, w = init_fmap1.shape
        coords = (
            torch.arange(w, device=init_fmap1.device)
            .reshape(1, 1, w, 1)
            .repeat(b, h, 1, 1)
        )
        self.coords_flat = coords.reshape(b * h * w)

    def __call__(self, disp):
        r = self.radius
        
        b, _, h, w = disp.shape
        disp_flat = disp.reshape(b * h * w)
        
        samples = []
        dx = torch.linspace(-r, r, 2 * r + 1)
        dx = dx.view(1, 1, 2 * r + 1, 1).to(disp.device)
        geo_volume = self.geo_volume
        x0 = dx + disp_flat.view(b * h * w, 1, 1, 1) / 2
        y0 = torch.zeros_like(x0)
        disp_lvl = torch.cat([x0, y0], dim=-1)
        geo_volume_sampled = sample_correlation(geo_volume, disp_lvl)
        geo_volume_sampled = geo_volume_sampled.view(b, h, w, -1)
        
        init_corr = self.init_corr
        init_x0 = (
            self.coords_flat.view(b * h * w, 1, 1, 1)
            - disp_flat.view(b * h * w, 1, 1, 1)
            + dx
        )
        init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
        init_corr_sampled = sample_correlation(init_corr, init_coords_lvl)
        init_corr_sampled = init_corr_sampled.view(b, h, w, -1)
        
        samples.append(init_corr_sampled)
        samples.append(geo_volume_sampled)
        out = torch.cat(samples, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr
