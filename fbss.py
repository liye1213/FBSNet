import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as selective_scan_fn


class LearnableFrequencyBandDecomposition(nn.Module):
    """
    LL, LH, HL, HH.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

      
        self.low_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )
        self.high_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )

        
        self.low_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )
        self.high_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.low_h, self.high_h, self.low_v, self.high_v]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        # Eq. (2) in the paper
        ll = self.low_h(self.low_v(x))
        lh = self.high_h(self.low_v(x))
        hl = self.low_h(self.high_v(x))
        hh = self.high_h(self.high_v(x))
        return {"LL": ll, "LH": lh, "HL": hl, "HH": hh}


class SelectiveStateSpace2D(nn.Module):
    """
    2D selective scan block
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        b, c, h, w = x.shape
        l = h * w
        k = 4

        x_hwwh = torch.stack(
            [x.view(b, -1, l), torch.transpose(x, 2, 3).contiguous().view(b, -1, l)],
            dim=1,
        ).view(b, 2, -1, l)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)

        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)
        Bs = Bs.float().view(b, k, -1, l)
        Cs = Cs.float().view(b, k, -1, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, l)
        wh_y = torch.transpose(out_y[:, 1].view(b, -1, w, h), 2, 3).contiguous().view(b, -1, l)
        invwh_y = torch.transpose(inv_y[:, 1].view(b, -1, w, h), 2, 3).contiguous().view(b, -1, l)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        b, h, w, c = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, 1, 2).contiguous().view(b, h, w, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class FrequencyBandSequenceModeling(nn.Module):
    """
    Band-wise state space modeling in FBSS.
    This block takes concatenated band features and models their dependencies.
    """

    def __init__(self, channels: int, expand: int = 2, d_state: int = 16, d_conv: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ssm = SelectiveStateSpace2D(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> SS2D expects [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.ssm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FrequencyBandFusion(nn.Module):
   

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, ll, lh, hl, hh):
        x = torch.cat([ll, lh, hl, hh], dim=1)
        return self.fuse(x)


class FBSS(nn.Module):

    def __init__(
        self,
        channels: int,
        lfbd_kernel_size: int = 3,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lfbd = LearnableFrequencyBandDecomposition(channels, kernel_size=lfbd_kernel_size)
        self.band_ssm = FrequencyBandSequenceModeling(
            channels=channels,
            expand=expand,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout,
        )
        self.fusion = FrequencyBandFusion(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        bands = self.lfbd(x)
        ll = self.band_ssm(bands["LL"])
        lh = self.band_ssm(bands["LH"])
        hl = self.band_ssm(bands["HL"])
        hh = self.band_ssm(bands["HH"])

        out = self.fusion(ll, lh, hl, hh)
        return residual + out

