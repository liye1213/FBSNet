import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError(
        "Please install mamba-ssm first: pip install mamba-ssm"
    ) from e


def flatten_hw(x: torch.Tensor):
    
    b, c, h, w = x.shape
    x_flat = x.view(b, c, h * w).transpose(1, 2)
    return x_flat, (h, w)


def unflatten_hw(x_flat: torch.Tensor, hw):
    
    b, hw_tokens, c = x_flat.shape
    h, w = hw
    return x_flat.transpose(1, 2).contiguous().view(b, c, h, w)


class BidirectionalSpatialMamba(nn.Module):
   

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

        self.mamba_forward = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.mamba_backward = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )

        self.out_proj = copy.deepcopy(self.mamba_forward.out_proj)
        self.mamba_forward.out_proj = nn.Identity()
        self.mamba_backward.out_proj = nn.Identity()

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()

        residual = x
        x_flat, hw = flatten_hw(x)
        x_norm = self.norm(x_flat)
        x_reverse = torch.flip(x_norm, dims=[1])

        y_forward = self.mamba_forward(x_norm)
        y_backward = self.mamba_backward(x_reverse)
        y = self.out_proj(y_forward + torch.flip(y_backward, dims=[1]))
        y = unflatten_hw(y, hw)
        return residual + y


class RegionalSpatialMamba(nn.Module):
  

    def __init__(
        self,
        channels: int,
        region_size: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.region_size = max(1, int(region_size))
        self.region_mamba = BidirectionalSpatialMamba(
            channels=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r = self.region_size

        if h % r != 0 or w % r != 0:
            return self.region_mamba(x)

        x_regions = (
            x.view(b, c, h // r, r, w // r, r)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1, c, r, r)
        )

        x_regions = self.region_mamba(x_regions)

        x = (
            x_regions.view(b, h // r, w // r, c, r, r)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, h, w)
        )
        return x


class GlobalSpatialMamba(nn.Module):
 

    def __init__(
        self,
        channels: int,
        pool_size: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.pool_size = max(1, int(pool_size))
        self.global_mamba = BidirectionalSpatialMamba(
            channels=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.pool_size

        if h % p == 0 and w % p == 0 and p > 1:
            pooled = F.avg_pool2d(x, kernel_size=p, stride=p)
            refined = self.global_mamba(pooled)
            refined = F.interpolate(refined, size=(h, w), mode="nearest")
            return refined

        return self.global_mamba(x)


class HSM(nn.Module):
   

    def __init__(
        self,
        channels: int,
        region_size: int = 8,
        pool_size: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.region_branch = RegionalSpatialMamba(
            channels=channels,
            region_size=region_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.global_branch = GlobalSpatialMamba(
            channels=channels,
            pool_size=pool_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.region_branch(x)
        x = self.global_branch(x)
        return residual + x



