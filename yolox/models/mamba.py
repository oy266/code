import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import Optional, Callable
drop_prob = 0.2
DropPath.__repr__ = lambda self: f"timm.DropPath({drop_prob})"

'''
Spatio-Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_ST(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3), stride=1,
                 padding=(0, 1, 1), dilation=1, groups=1, bias=False, theta=0.6, **factory_kwargs):

        super(CDC_ST, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias, device=factory_kwargs['device'], dtype=factory_kwargs['dtype'])
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight.sum(2).sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


class SS2DTem(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv # 128
        self.expand = expand # 2
        self.d_inner = int(self.expand * self.d_model) # 256
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # 8

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
        self.conv2dxt = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_inner,
            groups=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # self.x_proj = [CDC_ST(in_channels=self.d_inner, out_channels=self.dt_rank + self.d_state * 2, kernel_size=(2, 3, 3), **factory_kwargs)
        #           for _ in range(4)]
        x_proj = [CDC_ST(in_channels=self.d_inner, out_channels=self.dt_rank + self.d_state * 2, kernel_size=(2, 3, 3), **factory_kwargs)
                  for _ in range(4)]
        self.x_proj = nn.ModuleList(x_proj)



        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N) [1024, 16]
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N) [1024]

        # self.selective_scan = selective_scan_fn
        # self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.avgpool2dC = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.maxpool2dC = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.avgpool2dR = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.maxpool2dR = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.avgpool3d = nn.AdaptiveAvgPool3d(output_size=(1, None, None))
        self.maxpool3d = nn.AdaptiveMaxPool3d(output_size=(1, None, None))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True): # 16, 256
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous() # [256, 16]
        A_log = torch.log(A)  # Keep A_log in fp32 [256, 16]
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies) # [4, 256, 16]
            if merge:
                A_log = A_log.flatten(0, 1) # [1024, 16]
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device) # [256]
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies) # [4, 256]
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def interpolate(self, xc: torch.Tensor, xt: torch.Tensor):
        B, C, H, W = xc.shape
        L = H * W
        inter_col = torch.zeros_like(xc)
        inter_row = torch.zeros_like(xc)

        xc_avgC = self.avgpool2dC(xc)
        xc_maxC = self.maxpool2dC(xc)
        xc_C = xc_avgC + xc_maxC
        xt_avgC = self.avgpool2dC(xt)
        xt_maxC = self.maxpool2dC(xt)
        xt_C = xt_avgC + xt_maxC

        xc_avgR = self.avgpool2dR(xc)
        xc_maxR = self.maxpool2dR(xc)
        xc_R = xc_avgR + xc_maxR
        xt_avgR = self.avgpool2dR(xt)
        xt_maxR = self.maxpool2dR(xt)
        xt_R = xt_avgR + xt_maxR

        inter_col[:, :, :, 0::2] = xc_C.clone()
        inter_col[:, :, :, 1::2] = xt_C.clone()
        inter_row[:, :, 0::2, :] = xc_R.clone()
        inter_row[:, :, 1::2, :] = xt_R.clone()

        x_col = torch.stack([inter_col.reshape(B, -1, L), torch.flip(inter_col, dims=[-1]).reshape(B, -1, L)], dim=1)
        inter_row = torch.transpose(inter_row, dim0=2, dim1=3).contiguous()
        x_row = torch.stack([inter_row.reshape(B, -1, L), torch.flip(inter_row, dims=[-1]).reshape(B, -1, L)], dim=1)
        x_out = torch.cat([x_row, x_col], dim=1)

        return x_out

    def forward_coretem(self, xc: torch.Tensor, xt: torch.Tensor): # [2, 256, 32, 32] [2, 256, 32, 32]
        self.selective_scan = selective_scan_fn

        B, C, H, W = xc.shape  # [2, 256, 32, 32]
        L = H * W
        K = 4  # directions
        # [2, 2, 256, 1024]

        xs = self.interpolate(xc, xt)

        # [2, 4, 256, 1024] four directions
        x_dbl = []
        x_in = torch.stack([xc, xt], dim=2)
        for i, conv in enumerate(self.x_proj):
            # conv.cuda()
            dbl_init = conv(x_in)
            dbl_init_avg = self.avgpool3d(dbl_init)
            dbl_init = self.maxpool3d(dbl_init) + dbl_init_avg
            dbl_init = dbl_init.squeeze()
            x_dbl.append(dbl_init)
        x_dbl = torch.stack(x_dbl, dim=1).reshape(B, K, -1, L)
        #  [2, 4, 40, 1024]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1) [2, 4, 40, 1024]
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) # 8, 16, 16 [2, 4, 8, 1024] [2, 4, 16, 1024]*2
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # [2, 4, 8, 1024] [4, 256, 8] -> [2, 4, 256, 1024]
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l) [2, 4, 256, 1024] -> [2, 1024, 1024]
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [2, 1024, 1024]
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [2, 4, 16, 1024]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [2, 4, 16, 1024]
        Ds = self.Ds.float().view(-1)  # (k * d) [1024]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [1024, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d) [1024]

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024]
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y # [1, 192, 1024] * 4

    def forward(self, x, xt): # , **kwargs
        B, H, W, C = x.shape  # [2, 32, 32, 128]

        xz = self.in_proj(x)  # [2, 32, 32, 512]
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [2, 32, 32, 256] * 2

        x = x.permute(0, 3, 1, 2).contiguous()  # [2, 256, 32, 32]
        x = self.act(self.conv2d(x))  # (b, d, h, w) [2, 256, 32, 32]

        xt = xt.permute(0, 3, 1, 2).contiguous()  # [2, 128, 32, 32]
        xt = self.act(self.conv2dxt(xt))  # (b, d, h, w) [2, 256, 32, 32]
        # SS2D
        y1, y2, y3, y4 = self.forward_coretem(x, xt)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # [1, 192, 1024]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2DTemWithoutDiff(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv # 128
        self.expand = expand # 2
        self.d_inner = int(self.expand * self.d_model) # 256
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # 8

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
        self.conv2dxt = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_inner,
            groups=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj


        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N) [1024, 16]
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N) [1024]

        # self.selective_scan = selective_scan_fn
        # self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.avgpool2dC = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.maxpool2dC = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.avgpool2dR = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.maxpool2dR = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.avgpool3d = nn.AdaptiveAvgPool3d(output_size=(1, None, None))
        self.maxpool3d = nn.AdaptiveMaxPool3d(output_size=(1, None, None))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True): # 16, 256
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous() # [256, 16]
        A_log = torch.log(A)  # Keep A_log in fp32 [256, 16]
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies) # [4, 256, 16]
            if merge:
                A_log = A_log.flatten(0, 1) # [1024, 16]
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device) # [256]
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies) # [4, 256]
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def interpolate(self, xc: torch.Tensor, xt: torch.Tensor):
        B, C, H, W = xc.shape
        L = H * W
        inter_col = torch.zeros_like(xc)
        inter_row = torch.zeros_like(xc)

        xc_avgC = self.avgpool2dC(xc)
        xc_maxC = self.maxpool2dC(xc)
        xc_C = xc_avgC + xc_maxC
        xt_avgC = self.avgpool2dC(xt)
        xt_maxC = self.maxpool2dC(xt)
        xt_C = xt_avgC + xt_maxC

        xc_avgR = self.avgpool2dR(xc)
        xc_maxR = self.maxpool2dR(xc)
        xc_R = xc_avgR + xc_maxR
        xt_avgR = self.avgpool2dR(xt)
        xt_maxR = self.maxpool2dR(xt)
        xt_R = xt_avgR + xt_maxR

        inter_col[:, :, :, 0::2] = xc_C.clone()
        inter_col[:, :, :, 1::2] = xt_C.clone()
        inter_row[:, :, 0::2, :] = xc_R.clone()
        inter_row[:, :, 1::2, :] = xt_R.clone()

        x_col = torch.stack([inter_col.reshape(B, -1, L), torch.flip(inter_col, dims=[-1]).reshape(B, -1, L)], dim=1)
        inter_row = torch.transpose(inter_row, dim0=2, dim1=3).contiguous()
        x_row = torch.stack([inter_row.reshape(B, -1, L), torch.flip(inter_row, dims=[-1]).reshape(B, -1, L)], dim=1)
        x_out = torch.cat([x_row, x_col], dim=1)

        return x_out

    def forward_coretem(self, xc: torch.Tensor, xt: torch.Tensor): # [2, 256, 32, 32] [2, 256, 32, 32]
        self.selective_scan = selective_scan_fn

        B, C, H, W = xc.shape  # [2, 256, 32, 32]
        L = H * W
        K = 4  # directions
        # [2, 2, 256, 1024]

        xs = self.interpolate(xc, xt)

        # [2, 4, 256, 1024] four directions
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        #  [2, 4, 40, 1024]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1) [2, 4, 40, 1024]
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) # 8, 16, 16 [2, 4, 8, 1024] [2, 4, 16, 1024]*2
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # [2, 4, 8, 1024] [4, 256, 8] -> [2, 4, 256, 1024]
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l) [2, 4, 256, 1024] -> [2, 1024, 1024]
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [2, 1024, 1024]
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [2, 4, 16, 1024]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [2, 4, 16, 1024]
        Ds = self.Ds.float().view(-1)  # (k * d) [1024]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [1024, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d) [1024]

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024]
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y # [1, 192, 1024] * 4

    def forward(self, x, xt): # , **kwargs
        B, H, W, C = x.shape  # [2, 32, 32, 128]

        xz = self.in_proj(x)  # [2, 32, 32, 512]
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [2, 32, 32, 256] * 2

        x = x.permute(0, 3, 1, 2).contiguous()  # [2, 256, 32, 32]
        x = self.act(self.conv2d(x))  # (b, d, h, w) [2, 256, 32, 32]

        xt = xt.permute(0, 3, 1, 2).contiguous()  # [2, 128, 32, 32]
        xt = self.act(self.conv2dxt(xt))  # (b, d, h, w) [2, 256, 32, 32]
        # SS2D
        y1, y2, y3, y4 = self.forward_coretem(x, xt)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # [1, 192, 1024]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2DVideoMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv  # 128
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 256
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 8

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
        self.conv2dxt = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_inner,
            groups=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=4, D, N) [1024, 16]
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=4, D, N) [1024]

        # self.selective_scan = selective_scan_fn
        # self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):  # 16, 256
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()  # [256, 16]
        A_log = torch.log(A)  # Keep A_log in fp32 [256, 16]
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # [4, 256, 16]
            if merge:
                A_log = A_log.flatten(0, 1)  # [1024, 16]
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)  # [256]
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)  # [4, 256]
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def spatial_first(self, xc: torch.Tensor, xt: torch.Tensor):
        B, C, H, W = xc.shape
        x = torch.stack([xc, xt], dim=2)
        T = 2
        x = x.reshape(B, C, T*H*W)
        x_f = torch.flip(x, dims=[2])
        x_in = torch.stack([x, x_f], dim=1)
        return x_in

    def forward_corev0(self, x: torch.Tensor, xt):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape  # [1, 192, 32, 32]
        L = H * W * 2
        K = 2  # directions
        # [1, 192, 32, 32]
        xs = self.spatial_first(x, xt)
        # [1, 4, 192, 1024] four directions
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)  # [1, 4, 38, 1024]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Ds = self.Ds.float().view(-1)  # (k * d) [768]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024] [1, 2, 192, 1024]
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, -1, L)
        return out_y[:, 0, :, 0:H*W]+out_y[:, 0, :, H*W:], inv_y[:, :, 0:H*W]+inv_y[:, :, H*W:] # [1, 192, 1024] * 4

    def forward(self, x, xt):  # , **kwargs
        B, H, W, C = x.shape  # [2, 32, 32, 128]

        xz = self.in_proj(x)  # [2, 32, 32, 512]
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [2, 32, 32, 256] * 2

        x = x.permute(0, 3, 1, 2).contiguous()  # [2, 256, 32, 32]
        x = self.act(self.conv2d(x))  # (b, d, h, w) [2, 256, 32, 32]

        xt = xt.permute(0, 3, 1, 2).contiguous()  # [2, 128, 32, 32]
        xt = self.act(self.conv2dxt(xt))  # (b, d, h, w) [2, 256, 32, 32]
        # SS2D
        y1, y2 = self.forward_corev0(x, xt)
        assert y1.dtype == torch.float32
        y = y1 + y2  # [1, 192, 1024]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2DVideoMambaWithDiff(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv  # 128
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 256
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 8

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
        self.conv2dxt = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_inner,
            groups=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_proj = [CDC_ST(in_channels=self.d_inner, out_channels=self.dt_rank + self.d_state * 2, kernel_size=(2, 3, 3), **factory_kwargs)
                  for _ in range(2)]
        self.x_proj = nn.Sequential(*x_proj)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=4, D, N) [1024, 16]
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=4, D, N) [1024]

        # self.selective_scan = selective_scan_fn
        # self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):  # 16, 256
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()  # [256, 16]
        A_log = torch.log(A)  # Keep A_log in fp32 [256, 16]
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # [4, 256, 16]
            if merge:
                A_log = A_log.flatten(0, 1)  # [1024, 16]
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)  # [256]
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)  # [4, 256]
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def spatial_first(self, xc: torch.Tensor, xt: torch.Tensor):
        B, C, H, W = xc.shape
        x = torch.stack([xc, xt], dim=2)
        T = 2
        x = x.reshape(B, C, T*H*W)
        x_f = torch.flip(x, dims=[2])
        x_in = torch.stack([x, x_f], dim=1)
        return x_in

    def forward_corev0(self, x: torch.Tensor, xt):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape  # [1, 192, 32, 32]
        L = H * W * 2
        K = 2  # directions
        # [1, 192, 32, 32]
        xs = self.spatial_first(x, xt)
        # [1, 4, 192, 1024] four directions
        # x_dbl = []
        x_in = torch.stack([x, xt], dim=2)
        x_dbl = []
        for i, conv in enumerate(self.x_proj):
            dbl_init = conv(x_in)
            x_dbl.append(dbl_init)
        x_dbl = torch.stack(x_dbl, dim=1).reshape(B, K, -1, L)
         # [1, 4, 38, 1024]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Ds = self.Ds.float().view(-1)  # (k * d) [768]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024] [1, 2, 192, 1024]
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, -1, L)
        return out_y[:, 0, :, 0:H*W]+out_y[:, 0, :, H*W:], inv_y[:, :, 0:H*W]+inv_y[:, :, H*W:] # [1, 192, 1024] * 4

    def forward(self, x, xt):  # , **kwargs
        B, H, W, C = x.shape  # [2, 32, 32, 128]

        xz = self.in_proj(x)  # [2, 32, 32, 512]
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [2, 32, 32, 256] * 2

        x = x.permute(0, 3, 1, 2).contiguous()  # [2, 256, 32, 32]
        x = self.act(self.conv2d(x))  # (b, d, h, w) [2, 256, 32, 32]

        xt = xt.permute(0, 3, 1, 2).contiguous()  # [2, 128, 32, 32]
        xt = self.act(self.conv2dxt(xt))  # (b, d, h, w) [2, 256, 32, 32]
        # SS2D
        y1, y2 = self.forward_corev0(x, xt)
        assert y1.dtype == torch.float32
        y = y1 + y2  # [1, 192, 1024]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlocktem(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2DTem(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.fc = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input):
        # x = input + self.self_attention(self.ln_1(input))
        # x = x + self.drop_path(self.fc(self.ln_2(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        xc = self.ln_1(input[0])
        xt = self.ln_1(input[1])
        x_attn = self.self_attention(xc, xt)
        x = xc + self.drop_path(x_attn)
        return [x, xt]

class VSSBlocktemWithDiffVideoMamba(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2DVideoMambaWithDiff(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.fc = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input):
        # x = input + self.self_attention(self.ln_1(input))
        # x = x + self.drop_path(self.fc(self.ln_2(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        xc = self.ln_1(input[0])
        xt = self.ln_1(input[1])
        x_attn = self.self_attention(xc, xt)
        x = xc + self.drop_path(x_attn)
        return [x, xt]

class VSSBlocktemWithDiffVivim(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VivimSS2DWithDiff(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.fc = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input):
        # x = input + self.self_attention(self.ln_1(input))
        # x = x + self.drop_path(self.fc(self.ln_2(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        xc = self.ln_1(input[0])
        xt = self.ln_1(input[1])
        x_attn = self.self_attention(xc, xt)
        x = xc + self.drop_path(x_attn)
        return [x, xt]

class VSSBlocktemWithoutDiff(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2DTem(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.fc = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input):
        # x = input + self.self_attention(self.ln_1(input))
        # x = x + self.drop_path(self.fc(self.ln_2(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        xc = self.ln_1(input[0])
        xt = self.ln_1(input[1])
        x_attn = self.self_attention(xc, xt)
        x = xc + self.drop_path(x_attn)
        return [x, xt]

class VSSBlocktemVideoMamba(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2DVideoMamba(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.fc = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input):
        # x = input + self.self_attention(self.ln_1(input))
        # x = x + self.drop_path(self.fc(self.ln_2(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        xc = self.ln_1(input[0])
        xt = self.ln_1(input[1])
        x_attn = self.self_attention(xc, xt)
        x = xc + self.drop_path(x_attn)
        return [x, xt]

class VSSBlocktemVivim(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VivimSS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        self.fc = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input):
        # x = input + self.self_attention(self.ln_1(input))
        # x = x + self.drop_path(self.fc(self.ln_2(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        xc = self.ln_1(input[0])
        xt = self.ln_1(input[1])
        x_attn = self.self_attention(xc, xt)
        x = xc + self.drop_path(x_attn)
        return [x, xt]

class VivimSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
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
        self.conv2dxt = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_inner,
            groups=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor, xt):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape  # [1, 192, 32, 32]
        L = H * W * 2
        K = 4  # directions
        # [1, 192, 32, 32]
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
        #                      dim=1).view(B, 2, -1, L)  # [1, 2, 192, 1024]
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        T = 2
        x_t = torch.stack([x, xt], dim=2)
        x_t1 = x_t.reshape(B, C, T*H*W)
        x_t2 = x_t.reshape(B, C, H*W*T)
        x1 = torch.stack([x_t1, torch.flip(x_t1, dims=[2])], dim=1)
        x2 = torch.stack([x_t2, torch.flip(x_t2, dims=[2])], dim=1)
        xs = torch.cat([x1, x2], dim=1)
        # [1, 4, 192, 1024] four directions
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)  # [1, 4, 38, 1024]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Ds = self.Ds.float().view(-1)  # (k * d) [768]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024]
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return (out_y[:, 0, :, 0:H*W] + out_y[:, 0, :, H*W:],
                inv_y[:, 0, :, 0:H*W] + inv_y[:, 0, :, H*W:],
                wh_y[:, :, 0:H*W] + wh_y[:, :, H*W:],
                invwh_y[:, :, 0:H*W] + invwh_y[:, :, H*W:])  # [1, 192, 1024] * 4

    def forward(self, x: torch.Tensor, xt,**kwargs):
        B, H, W, C = x.shape  # [1, 32, 32, 96]

        xz = self.in_proj(x)  # [1, 32, 32, 384]
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [1, 32, 32, 192] * 2

        x = x.permute(0, 3, 1, 2).contiguous()  # [1, 192, 32, 32]
        x = self.act(self.conv2d(x))  # (b, d, h, w) [1, 192, 32, 32]

        xt = xt.permute(0, 3, 1, 2).contiguous()  # [2, 128, 32, 32]
        xt = self.act(self.conv2dxt(xt))  # (b, d, h, w) [2, 256, 32, 32]
        # SS2D
        y1, y2, y3, y4 = self.forward_corev0(x, xt)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # [1, 192, 1024]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VivimSS2DWithDiff(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
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
        self.conv2dxt = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_inner,
            groups=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_proj = [CDC_ST(in_channels=self.d_inner, out_channels=self.dt_rank + self.d_state * 2, kernel_size=(2, 3, 3), **factory_kwargs)
                  for _ in range(4)]
        self.x_proj = nn.ModuleList(x_proj)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor, xt):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape  # [1, 192, 32, 32]
        L = H * W * 2
        K = 4  # directions
        # [1, 192, 32, 32]
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
        #                      dim=1).view(B, 2, -1, L)  # [1, 2, 192, 1024]
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        T = 2
        x_t = torch.stack([x, xt], dim=2)
        x_t1 = x_t.reshape(B, C, T*H*W)
        x_t2 = x_t.reshape(B, C, H*W*T)
        x1 = torch.stack([x_t1, torch.flip(x_t1, dims=[2])], dim=1)
        x2 = torch.stack([x_t2, torch.flip(x_t2, dims=[2])], dim=1)
        xs = torch.cat([x1, x2], dim=1)
        # [1, 4, 192, 1024] four directions
        x_dbl = []
        for i, conv in enumerate(self.x_proj):
            # conv.cuda()
            dbl_init = conv(x_t)
            x_dbl.append(dbl_init)
        x_dbl = torch.stack(x_dbl, dim=1).reshape(B, K, -1, L)  # [1, 4, 38, 1024]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
        Ds = self.Ds.float().view(-1)  # (k * d) [768]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024]
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return (out_y[:, 0, :, 0:H*W] + out_y[:, 0, :, H*W:],
                inv_y[:, 0, :, 0:H*W] + inv_y[:, 0, :, H*W:],
                wh_y[:, :, 0:H*W] + wh_y[:, :, H*W:],
                invwh_y[:, :, 0:H*W] + invwh_y[:, :, H*W:])  # [1, 192, 1024] * 4

    def forward(self, x: torch.Tensor, xt,**kwargs):
        B, H, W, C = x.shape  # [1, 32, 32, 96]

        xz = self.in_proj(x)  # [1, 32, 32, 384]
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [1, 32, 32, 192] * 2

        x = x.permute(0, 3, 1, 2).contiguous()  # [1, 192, 32, 32]
        x = self.act(self.conv2d(x))  # (b, d, h, w) [1, 192, 32, 32]

        xt = xt.permute(0, 3, 1, 2).contiguous()  # [2, 128, 32, 32]
        xt = self.act(self.conv2dxt(xt))  # (b, d, h, w) [2, 256, 32, 32]
        # SS2D
        y1, y2, y3, y4 = self.forward_corev0(x, xt)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # [1, 192, 1024]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
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
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape  # [1, 192, 32, 32]
        L = H * W
        K = 4  # directions
        # [1, 192, 32, 32]
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  # [1, 2, 192, 1024]
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        # [1, 4, 192, 1024] four directions
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)  # [1, 4, 38, 1024]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024] [1, 1024, 64]
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024] [1, 1024, 64]
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024] [1, 4, 16, 64]
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024] [1, 4, 16, 64]
        Ds = self.Ds.float().view(-1)  # (k * d) [768] [1024]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768, 16] [1024, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d) [1024]

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024]
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y  # [1, 192, 1024] * 4

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape  # [1, 32, 32, 96]

        xz = self.in_proj(x)  # [1, 32, 32, 384]
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [1, 32, 32, 192] * 2

        x = x.permute(0, 3, 1, 2).contiguous()  # [1, 192, 32, 32]
        x = self.act(self.conv2d(x))  # (b, d, h, w) [1, 192, 32, 32]
        # SS2D
        y1, y2, y3, y4 = self.forward_corev0(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  # [1, 192, 1024]
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class ShiftVSS(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n=1,
        **kwargs,
    ):
        super().__init__()
        module_list = [
            VSSBlocktem(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blockstem = nn.Sequential(*module_list)
        module_list = [
            VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blocks = nn.Sequential(*module_list)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)

    def temporal_shift1(self, x, xt):
        B, H, W, C = x.shape # [1, 8, 8, 128]

        out1 = torch.zeros_like(x) # [B, H, W, C]
        out1[:, :, :, :C // 2] = x[:, :, :, :C // 2] # 前半部分
        out1[:, :, :, C // 2:] = xt[:, :, :, C // 2:] # 后半部分

        return out1

    def temporal_shift2(self, x, xt):
        B, H, W, C = x.shape # [1, 8, 8, 128]

        out2 = torch.zeros_like(x)  # [B, H, W, C]
        out2[:, :, :, C // 2:] = x[:, :, :, C // 2:] # 后半部分
        out2[:, :, :, :C // 2] = xt[:, :, :, :C // 2] # 前半部分

        return out2

    def temporal_sp_shift(self, x, xt):
        xt = self.avgpool2d(xt.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).clone() # [B, H // 3, W // 3, C]
        # x[:, 1::3, 1::3, :] = xt + x[:, 1::3, 1::3, :]
        out = x.clone()
        L = x.shape[1]
        G = L // 3
        R = L - 3 * G
        # print(x.shape)
        # print(xt.shape)
        # print(R)
        if R != 0:
            out[:, 1:-R:3, 1:-R:3, :] = xt
        elif R == 0:
            out[:, 1::3, 1::3, :] = xt
        return out
    def forward(self, x, xt):
        # shift1 = self.temporal_shift1(x, xt)
        # shift2 = self.temporal_shift2(x, xt)
        # shift1_out = self.VS_blocks(shift1)
        # shift2_out = self.VS_blocks(shift2)
        [x_out_tem, xt] = self.VS_blockstem([x, xt])
        x_out = self.VS_blocks(x)

        output = x_out + x_out_tem #+ shift1_out +shift2_out

        # shift = self.temporal_sp_shift(x, xt)
        # shift_out = self.VS_blocks(shift)
        # x_out = self.VS_blocks(x)
        # output = x_out + shift_out

        return output # [1, 8, 8, 128]

class ShiftVSSAblaVideoMamba(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n=1,
        **kwargs,
    ):
        super().__init__()
        module_list = [
            VSSBlocktemVideoMamba(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blockstem = nn.Sequential(*module_list)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)

    def forward(self, x, xt):
        # shift1 = self.temporal_shift1(x, xt)
        # shift2 = self.temporal_shift2(x, xt)
        # shift1_out = self.VS_blocks(shift1)
        # shift2_out = self.VS_blocks(shift2)
        [x_out_tem, xt] = self.VS_blockstem([x, xt])


        output = x_out_tem #+ shift1_out +shift2_out

        # shift = self.temporal_sp_shift(x, xt)
        # shift_out = self.VS_blocks(shift)
        # x_out = self.VS_blocks(x)
        # output = x_out + shift_out

        return output # [1, 8, 8, 128]

class ShiftVSSAblaVivim(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n=1,
        **kwargs,
    ):
        super().__init__()
        module_list = [
            VSSBlocktemVivim(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blockstem = nn.Sequential(*module_list)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)

    def forward(self, x, xt):
        # shift1 = self.temporal_shift1(x, xt)
        # shift2 = self.temporal_shift2(x, xt)
        # shift1_out = self.VS_blocks(shift1)
        # shift2_out = self.VS_blocks(shift2)
        [x_out_tem, xt] = self.VS_blockstem([x, xt])


        output = x_out_tem #+ shift1_out +shift2_out

        # shift = self.temporal_sp_shift(x, xt)
        # shift_out = self.VS_blocks(shift)
        # x_out = self.VS_blocks(x)
        # output = x_out + shift_out

        return output # [1, 8, 8, 128]

class ShiftVSSWithoutDiff(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n=1,
        **kwargs,
    ):
        super().__init__()
        module_list = [
            VSSBlocktem(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blockstem = nn.Sequential(*module_list)
        module_list = [
            VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blocks = nn.Sequential(*module_list)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)

    def temporal_shift1(self, x, xt):
        B, H, W, C = x.shape # [1, 8, 8, 128]

        out1 = torch.zeros_like(x) # [B, H, W, C]
        out1[:, :, :, :C // 2] = x[:, :, :, :C // 2] # 前半部分
        out1[:, :, :, C // 2:] = xt[:, :, :, C // 2:] # 后半部分

        return out1

    def temporal_shift2(self, x, xt):
        B, H, W, C = x.shape # [1, 8, 8, 128]

        out2 = torch.zeros_like(x)  # [B, H, W, C]
        out2[:, :, :, C // 2:] = x[:, :, :, C // 2:] # 后半部分
        out2[:, :, :, :C // 2] = xt[:, :, :, :C // 2] # 前半部分

        return out2

    def temporal_sp_shift(self, x, xt):
        xt = self.avgpool2d(xt.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).clone() # [B, H // 3, W // 3, C]
        # x[:, 1::3, 1::3, :] = xt + x[:, 1::3, 1::3, :]
        out = x.clone()
        L = x.shape[1]
        G = L // 3
        R = L - 3 * G
        # print(x.shape)
        # print(xt.shape)
        # print(R)
        if R != 0:
            out[:, 1:-R:3, 1:-R:3, :] = xt
        elif R == 0:
            out[:, 1::3, 1::3, :] = xt
        return out
    def forward(self, x, xt):
        # shift1 = self.temporal_shift1(x, xt)
        # shift2 = self.temporal_shift2(x, xt)
        # shift1_out = self.VS_blocks(shift1)
        # shift2_out = self.VS_blocks(shift2)
        [x_out_tem, xt] = self.VS_blockstem([x, xt])
        x_out = self.VS_blocks(x)

        output = x_out + x_out_tem #+ shift1_out +shift2_out

        # shift = self.temporal_sp_shift(x, xt)
        # shift_out = self.VS_blocks(shift)
        # x_out = self.VS_blocks(x)
        # output = x_out + shift_out

        return output # [1, 8, 8, 128]

class ShiftVSSDiffWithVideoMamba(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n=1,
        **kwargs,
    ):
        super().__init__()
        module_list = [
            VSSBlocktemWithDiffVideoMamba(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blockstem = nn.Sequential(*module_list)
        module_list = [
            VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blocks = nn.Sequential(*module_list)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)


    def forward(self, x, xt):
        # shift1 = self.temporal_shift1(x, xt)
        # shift2 = self.temporal_shift2(x, xt)
        # shift1_out = self.VS_blocks(shift1)
        # shift2_out = self.VS_blocks(shift2)
        [x_out_tem, xt] = self.VS_blockstem([x, xt])
        x_out = self.VS_blocks(x)

        output = x_out + x_out_tem #+ shift1_out +shift2_out

        # shift = self.temporal_sp_shift(x, xt)
        # shift_out = self.VS_blocks(shift)
        # x_out = self.VS_blocks(x)
        # output = x_out + shift_out

        return output # [1, 8, 8, 128]

class ShiftVSSDiffWithVivim(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        n=1,
        **kwargs,
    ):
        super().__init__()
        module_list = [
            VSSBlocktemWithDiffVivim(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blockstem = nn.Sequential(*module_list)
        module_list = [
            VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path,
                     attn_drop_rate=attn_drop_rate, d_state=d_state)
        for _ in range(n)]
        self.VS_blocks = nn.Sequential(*module_list)
        self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)


    def forward(self, x, xt):
        # shift1 = self.temporal_shift1(x, xt)
        # shift2 = self.temporal_shift2(x, xt)
        # shift1_out = self.VS_blocks(shift1)
        # shift2_out = self.VS_blocks(shift2)
        [x_out_tem, xt] = self.VS_blockstem([x, xt])
        x_out = self.VS_blocks(x)

        output = x_out + x_out_tem #+ shift1_out +shift2_out

        # shift = self.temporal_sp_shift(x, xt)
        # shift_out = self.VS_blocks(shift)
        # x_out = self.VS_blocks(x)
        # output = x_out + shift_out

        return output # [1, 8, 8, 128]



if __name__ == "__main__":
    x = torch.randn(2, 32, 32, 128).cuda()
    xt = torch.randn(2, 32, 32, 128) .cuda()
    module = ShiftVSSDiffWithVivim(hidden_dim=128, drop_path=0.2, attn_drop_rate=0, d_state=16, n=2).cuda()
    out = module(x, xt)
    pass

