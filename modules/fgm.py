import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierGuidanceModule(nn.Module):
    

    def __init__(self, channels: int, negative_slope: float = 0.1):
        super().__init__()
        self.amplitude_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()

        _, _, h, w = x.shape
        spectrum = torch.fft.rfft2(x, norm="backward")
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        refined_amplitude = amplitude + self.amplitude_refine(amplitude)

        real = refined_amplitude * torch.cos(phase)
        imag = refined_amplitude * torch.sin(phase)
        refined_spectrum = torch.complex(real, imag)
        refined = torch.fft.irfft2(refined_spectrum, s=(h, w), norm="backward")
        return refined


class SpatialRefinementUpsamplingBlock(nn.Module):
    

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        strip_kernel_size: int = 7,
        group_divisor: int = 4,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1, bias=False),
            nn.PixelShuffle(scale_factor),
        )

        groups = max(1, out_channels // group_divisor)

        self.vertical_branch = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, strip_kernel_size),
                padding=(0, strip_kernel_size // 2),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.horizontal_branch = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(strip_kernel_size, 1),
                padding=(strip_kernel_size // 2, 0),
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        vertical_feature = self.vertical_branch(x)
        horizontal_feature = self.horizontal_branch(x)
        x = torch.cat([vertical_feature, horizontal_feature], dim=1)
        x = self.fusion(x)
        return x


class FGRM(nn.Module):
  

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        strip_kernel_size: int = 7,
    ):
        super().__init__()
        self.fgm = FourierGuidanceModule(decoder_channels)
        self.pre_fusion = nn.Sequential(
            nn.Conv2d(decoder_channels + skip_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.srub = SpatialRefinementUpsamplingBlock(
            in_channels=decoder_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            strip_kernel_size=strip_kernel_size,
        )

    def forward(self, decoder_feature: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        guided_feature = self.fgm(decoder_feature)
        fused_feature = torch.cat([guided_feature, skip_feature], dim=1)
        fused_feature = self.pre_fusion(fused_feature)
        output = self.srub(fused_feature)
        return output



