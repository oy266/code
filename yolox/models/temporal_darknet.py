import torch
from torch import nn
import torch.nn.functional as F
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
# from .STF import baseNet3D, STFF, ISTFF, BasicConv3d
import torch.fft as fft
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import Optional, Callable
drop_prob = 0.2
DropPath.__repr__ = lambda self: f"timm.DropPath({drop_prob})"
# def Conv3DBlockS3(in_channels, out_channels):
#     return nn.Sequential(
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 5), stride=(1, 1, 1),
#                         padding=(0, 0, 2)),
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 5, 1), stride=(1, 1, 1),
#                         padding=(0, 2, 0)),
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
#                         padding=(1, 0, 0)),
#         )
#
# def Conv3DBlockS4(in_channels, out_channels):
#     return nn.Sequential(
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 3), stride=(1, 1, 1),
#                         padding=(0, 0, 1)),
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3, 1), stride=(1, 1, 1),
#                         padding=(0, 1, 0)),
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
#                         padding=(1, 0, 0)),
#         )
#
# def Conv3DBlockS5(in_channels, out_channels):
#     return nn.Sequential(
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 3), stride=(1, 1, 1),
#                         padding=(0, 0, 1)),
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3, 1), stride=(1, 1, 1),
#                         padding=(0, 1, 0)),
#             BasicConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
#                         padding=(1, 0, 0)),
#         )
class Van_Tada(nn.Module):
    def __init__(self, c_in, ratio, kernels, bn_eps=1e-5, bn_mmt=0.1):
        super(Van_Tada, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.glocalpool = nn.AdaptiveMaxPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernels[0], 1, 1],
            padding=[kernels[0]//2, 0, 0],
        )
        self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in//ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1, 1],
            padding=[kernels[1]//2, 0, 0],
            bias=False
        )
        self.b.skip_init=True
        self.b.weight.data.zero_()

    def forward(self, x):
        g = self.glocalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(g))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x




class attentionblock(nn.Module):  # attention global BxCxHxW

    def __init__(self, in_dim):
        super(attentionblock, self).__init__()

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=3, padding=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # [2, 384, 5, 5] [2, 384, 5, 5]
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # [2, 25, 24]
        proj_key = self.key_conv(y).view(m_batchsize, -1, width * height)  # [2, 24, 25]
        energy = torch.bmm(proj_query, proj_key)  # [2, 25, 25]
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).view(m_batchsize, -1, width * height)  # [2, 384, 25]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)  # [2, 384, 5, 5]

        return out

class Fourier(nn.Module):
    def __init__(self, in_dim):
        super(Fourier, self).__init__()
        self.chanel_in = in_dim

    def forward(self, x):
        shape = x.shape

        dft = fft.fft2(x.view(shape[0], shape[1], shape[2], shape[3] * shape[4]), dim=(2, 3), norm='ortho')
        dft = torch.conj(dft) * dft
        dft = fft.ifft2(dft, dim=(2, 3), norm='ortho')
        dft = dft.view(shape[0], shape[1], shape[2], shape[3], shape[4])

        output = 0.01 * dft.abs() * x

        return output


class Fourier2(nn.Module):
    def __init__(self, in_dim):
        super(Fourier2, self).__init__()
        self.chanel_in = in_dim

    def forward(self, x):
        # shape = x.shape

        fft_space = fft.fft2(x, dim=(3, 4), norm='ortho')
        fft_tem = fft.fft(fft_space, dim=2, norm='ortho')

        output = 0.01 * fft_tem.abs() * x

        return output

# class VivimSS2D(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             # d_state="auto", # 20240109
#             d_conv=3,
#             expand=2,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             conv_bias=True,
#             bias=False,
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#
#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         self.conv2dxt = nn.Conv2d(
#             in_channels=self.d_model,
#             out_channels=self.d_inner,
#             groups=self.d_model,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         self.act = nn.SiLU()
#
#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
#         del self.x_proj
#
#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
#         del self.dt_projs
#
#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
#         self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
#
#         # self.selective_scan = selective_scan_fn
#         self.forward_core = self.forward_corev0
#
#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None
#
#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
#
#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#
#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         dt_proj.bias._no_reinit = True
#
#         return dt_proj
#
#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log
#
#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D
#
#     def forward_corev0(self, x: torch.Tensor, xt):
#         self.selective_scan = selective_scan_fn
#
#         B, C, H, W = x.shape  # [1, 192, 32, 32]
#         L = H * W * 2
#         K = 4  # directions
#         # [1, 192, 32, 32]
#         # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
#         #                      dim=1).view(B, 2, -1, L)  # [1, 2, 192, 1024]
#         # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
#         T = 2
#         x_t = torch.stack([x, xt], dim=2)
#         x_t1 = x_t.reshape(B, C, T*H*W)
#         x_t2 = x_t.reshape(B, C, H*W*T)
#         x1 = torch.stack([x_t1, torch.flip(x_t1, dims=[2])], dim=1)
#         x2 = torch.stack([x_t2, torch.flip(x_t2, dims=[2])], dim=1)
#         xs = torch.cat([x1, x2], dim=1)
#         # [1, 4, 192, 1024] four directions
#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)  # [1, 4, 38, 1024]
#         # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)
#
#         xs = xs.float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
#         dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) [1, 768, 1024]
#         Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
#         Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) [1, 4, 16, 1024]
#         Ds = self.Ds.float().view(-1)  # (k * d) [768]
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768, 16]
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
#
#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)  # [1, 768, 1024] -> [1, 4, 192, 1024]
#         assert out_y.dtype == torch.float
#
#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#
#         return (out_y[:, 0, :, 0:H*W] + out_y[:, 0, :, H*W:],
#                 inv_y[:, 0, :, 0:H*W] + inv_y[:, 0, :, H*W:],
#                 wh_y[:, :, 0:H*W] + wh_y[:, :, H*W:],
#                 invwh_y[:, :, 0:H*W] + invwh_y[:, :, H*W:])  # [1, 192, 1024] * 4
#
#     def forward(self, x: torch.Tensor, xt,**kwargs):
#         B, H, W, C = x.shape  # [1, 32, 32, 96]
#
#         xz = self.in_proj(x)  # [1, 32, 32, 384]
#         x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) [1, 32, 32, 192] * 2
#
#         x = x.permute(0, 3, 1, 2).contiguous()  # [1, 192, 32, 32]
#         x = self.act(self.conv2d(x))  # (b, d, h, w) [1, 192, 32, 32]
#
#         xt = xt.permute(0, 3, 1, 2).contiguous()  # [2, 128, 32, 32]
#         xt = self.act(self.conv2dxt(xt))  # (b, d, h, w) [2, 256, 32, 32]
#         # SS2D
#         y1, y2, y3, y4 = self.forward_corev0(x, xt)
#         assert y1.dtype == torch.float32
#         y = y1 + y2 + y3 + y4  # [1, 192, 1024]
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
#
#         y = self.out_norm(y)
#         y = y * F.silu(z)
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out

# class VSSBlocktemVivim(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 0,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         attn_drop_rate: float = 0,
#         d_state: int = 16,
#         **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = VivimSS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
#         self.drop_path = DropPath(drop_path)
#         self.ln_2 = norm_layer(hidden_dim)
#         self.fc = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)
#
#     def forward(self, input):
#         # x = input + self.self_attention(self.ln_1(input))
#         # x = x + self.drop_path(self.fc(self.ln_2(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
#         xt = self.ln_1(input[0])
#         xt = self.ln_1(input[1])
#         x_attn = self.self_attention(xc, xt)
#         x = xc + self.drop_path(x_attn)
#         return [x, xt]

# class Mamba(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 0,
#         drop_path: float = 0,
#         attn_drop_rate: float = 0,
#         d_state: int = 16,
#         n=1,
#         **kwargs,
#     ):
#         super().__init__()
#         module_list = [
#             VSSBlocktemVivim(hidden_dim=hidden_dim, drop_path=drop_path,
#                      attn_drop_rate=attn_drop_rate, d_state=d_state)
#         for _ in range(n)]
#         self.VS_blockstem = nn.Sequential(*module_list)
#         self.avgpool2d = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
#
#     def forward(self, x):
#         # shift1 = self.temporal_shift1(x, xt)
#         # shift2 = self.temporal_shift2(x, xt)
#         # shift1_out = self.VS_blocks(shift1)
#         # shift2_out = self.VS_blocks(shift2)
#         [x_out_tem, xt] = self.VS_blockstem([x])
#
#
#         output = x_out_tem #+ shift1_out +shift2_out
#
#         # shift = self.temporal_sp_shift(x, xt)
#         # shift_out = self.VS_blocks(shift)
#         # x_out = self.VS_blocks(x)
#         # output = x_out + shift_out
#
#         return output # [1, 8, 8, 128]


class TAModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, gate=False, attn_type='fourier'):
        super().__init__()
        self.gate = gate
        self.gate_control = Gate_control(channel=out_channels)
        self.attn_type = attn_type

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.init = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.maxpool = nn.AdaptiveMaxPool3d((None, 5, 5))
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporalconv = nn.Conv3d(in_channels, in_channels, (1, 1, 1))
        # nn.init.constant_(self.temporalconv.weight, 0)
        # nn.init.constant_(self.temporalconv.bias, 0)
        self.fc = nn.Conv3d(in_channels, 1, (1, 1, 1))

        if self.attn_type == 'cross_attn':
            self.attention = attentionblock(in_channels)
        if self.attn_type == 'fourier':
            self.fourier_attn = Fourier2(in_channels)
        # if self.attn_type == 'mamba':
        #     self.mamba = Mamba(in_channels=in_channels, hidden_dim=in_channels, n=1, drop_path=0.2,
        #                                 attn_drop_rate=0., d_state=16)
        # if self.attn_type == 'van_tada':
        #     self.van_tada = Van_Tada(c_in=in_channels)
        if self.attn_type == 'tada':
            self.spat_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.temp_pool = nn.AdaptiveAvgPool3d((1, None, None))
            self.conv1d_temp = nn.Conv3d(kernel_size=(1,1,1), in_channels=in_channels, out_channels=out_channels)
            self.conv1d_1 = nn.Conv3d(kernel_size=(5, 1, 1), padding=(2, 0, 0), in_channels=in_channels,
                                      out_channels=int(in_channels//2))
            self.conv1d_2 = nn.Conv3d(kernel_size=(5, 1, 1), padding=(2, 0, 0), in_channels=int(in_channels//2),
                                      out_channels=in_channels)
            self.fc_tada = nn.Conv3d(in_channels, 1, (1, 1, 1))


        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def generateweight_attn(self, xin):
            # [2, 8, 384, 10, 10]

        xin = xin.permute(0, 2, 1, 3, 4)  # x BxCxLxHxW [2, 384, 8, 10, 10]
        # xin = self.maxpool(xin)  # [2, 384, 8, 5, 5]

        prior_knowledge = self.init(xin[:, :, -1, :, :])  # [2, 384, 5, 5]

        for length in range(xin.size(2)):
            prior_knowledge = self.attention(prior_knowledge.squeeze(2),
                                                        xin[:, :, length, :, :]).unsqueeze(2)

            if length == 0:
                allxin = prior_knowledge
            else:
                allxin = torch.cat((allxin, prior_knowledge), 2)

        allxin = self.avgpool(allxin)  # x BxCxLx1x1 [2, 384, 8, 1, 1]

        calibration = self.temporalconv(allxin)

        finalweight = self.weight * (calibration + 1).unsqueeze(0).permute(1, 3, 0, 2, 4, 5)

        bias = self.bias * (self.fc(allxin) + 1).squeeze().unsqueeze(-1)

        return finalweight, bias

    def generateweight_fourier(self, xin):
        xin = xin.permute(0, 2, 1, 3, 4) # x BxCxTxHxW
        fourier_out = self.fourier_attn(xin)
        allxin = xin + fourier_out

        allxin = self.avgpool(allxin)

        calibration = self.temporalconv(allxin)
        finalweight = self.weight * (calibration + 1).unsqueeze(0).permute(1, 3, 0, 2, 4, 5)
        bias = self.bias * (self.fc(allxin) + 1).squeeze().unsqueeze(-1)

        return finalweight, bias

    def generateweight_mamba(self, xin):
        xin = xin.permute(0, 2, 1, 3, 4)  # x BxCxTxHxW
        mamba_out = self.mamba(xin) # [B, T, C] [2, 5, 128]
        mamba_out = mamba_out.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1) # [B, C, T, 1, 1] [2, 128, 5, 1, 1]

        calibration = self.temporalconv(mamba_out)
        finalweight = self.weight * (calibration + 1).unsqueeze(0).permute(1, 3, 0, 2, 4, 5)
        bias = self.bias * (self.fc(mamba_out) + 1).squeeze().unsqueeze(-1)

        return finalweight, bias

    def generateweight_tada(self, xin):
    # x BxCxTxHxW
        B, T, C, H, W = xin.shape
        xin = xin.reshape(B, C, T, H, W)
        x_s = self.spat_pool(xin)
        x_t = self.temp_pool(x_s)
        x_t = self.conv1d_temp(x_t)
        x = x_s + x_t
        x = self.conv1d_1(x)
        calibration = self.conv1d_2(x)
        finalweight = self.weight * (calibration + 1).unsqueeze(0).permute(1, 3, 0, 2, 4, 5)
        bias = self.bias * (self.fc_tada(x_s) + 1).squeeze().unsqueeze(-1)

        return finalweight, bias


    def forward(self, x):  # x B*L*C*W*H
        # [2, 8, 384, 10, 10]
        if self.attn_type == 'cross_attn':
            finalweight, finalbias = self.generateweight_attn(x)
        elif self.attn_type == 'fourier':
            finalweight, finalbias = self.generateweight_fourier(x)
        elif self.attn_type == 'mamba':
            finalweight, finalbias = self.generateweight_mamba(x)
        elif self.attn_type == 'tada':
            finalweight, finalbias = self.generateweight_tada(x)

        b, l, c_in, h, w = x.size()

        x = x.reshape(1, -1, h, w)
        finalweight = finalweight.reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)
        finalbias = finalbias.view(-1)

        if self.bias is not None:

            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)

        output = output.view(-1, self.out_channels, output.size(-2), output.size(-1))
        if self.gate:
            s = x.reshape(b, c_in, l, h, w)
            t = output.reshape(b, c_in, l, h, w)
            output = self.gate_control(s, t)
            output = output.reshape(-1, c_in, h, w)
        else:
            output = x.reshape(-1, c_in, h, w) + output
        output = self.bn(output)
        output = self.act(output)

        return output.reshape(b, l, c_in, h, w)

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

class Gate_control(nn.Module):
    def __init__(self, channel, r=16):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.w = nn.Linear(channel * 2, channel)

    def forward(self, s, t):
        bs, c, d, h, w = s.shape
        s = s.flatten(2) # bs, c, dhw
        s = s.permute(0, 2, 1) # bs, dhw, c
        t = t.flatten(2)
        t = t.permute(0, 2, 1)
        st = torch.cat((s, t), dim=2)
        weight = self.sigmoid(self.w(st))
        out = (1 - weight) * t + weight * s
        out = out.reshape(bs, c, d, h, w)
        return out

class TemCSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        num_frames=5
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.num_frames = num_frames
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        # self.TAS3 = TAModule(in_channels=base_channels * 4,
        #                      out_channels=base_channels * 4,
        #                      kernel_size=3)
        module_listTAS3 = [
            TAModule(
                in_channels=base_channels * 4, out_channels=base_channels * 4,  kernel_size=3
            )
            for _ in range(base_depth * 3)
        ]
        self.TAS3 = nn.Sequential(*module_listTAS3)

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        module_listTAS4 = [
            TAModule(
                in_channels=base_channels * 8, out_channels=base_channels * 8, kernel_size=3
            )
            for _ in range(base_depth * 3)
        ]
        self.TAS4 = nn.Sequential(*module_listTAS4)

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        module_listTAS5 = [
            TAModule(
                in_channels=base_channels * 16, out_channels=base_channels * 16, kernel_size=3
            )
            for _ in range(base_depth)
        ]
        self.TAS5 = nn.Sequential(*module_listTAS5)

    def forward(self, x):
        bs, _, T, H, W = x.shape # [2, 3, 5, 128, 128]
        x = x.permute(0, 2, 1, 3, 4) # [2, 5, 3, 128, 128]
        x = x.reshape(-1, x.size(2), x.size(3), x.size(4)) # [2, 3, 5, 128, 128] -> [10, 3, 128, 128] bs * t
        outputs = {}
        # outputs_spatial = {}
        x = self.stem(x) # [10, 32, 64, 64] H/2 W/2 32
        outputs["stem"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))#[:, :, -1, :, :]  # [2, 32, 5, 64, 64]
        x = self.dark2(x) # [10, 64, 32, 32] H/4 W/4 64
        outputs["dark2"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))#[:, :, -1, :, :]  # [2, 64, 5, 32, 32]
        x = self.dark3(x) # [10, 128, 16, 16]
        # x = x.view(bs, T, x.size(-3), x.size(-2), x.size(-1))
        # outputs_spatial["dark3"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))[:, :, -1, :, :]
        x = self.TAS3(x.view(bs, T, x.size(-3), x.size(-2), x.size(-1))) # [10, 128, 16, 16]
        outputs["dark3"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))#[:, :, -1, :, :]  # [2, 128, 5, 16, 16]
        x = self.dark4(x.flatten(0, 1)) # [10, 256, 8, 8]
        # outputs_spatial["dark4"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))[:, :, -1, :, :]
        x = self.TAS4(x.view(bs, T, x.size(-3), x.size(-2), x.size(-1))) # [10, 256, 8, 8]
        outputs["dark4"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))#[:, :, -1, :, :]  # [[2, 256, 5, 8, 8]
        x = self.dark5(x.flatten(0, 1)) # [10, 512, 4, 4]
        # outputs_spatial["dark5"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))[:, :, -1, :, :]
        x = self.TAS5(x.view(bs, T, x.size(-3), x.size(-2), x.size(-1))) # [10, 512, 4, 4]
        outputs["dark5"] = x.reshape(bs, -1, T, x.size(-2), x.size(-1))#[:, :, -1, :, :] # [2, 512, 5, 4, 4]
        return {k: v for k, v in outputs.items() if k in self.out_features}
        # [2, 64, 5, 32, 32] [2, 256, 5, 8, 8] [2, 512, 5, 4, 4]
        # [2, 64, 5, 32, 32] [2, 256, 5, 8, 8] [2, 512, 5, 4, 4]

if __name__ == '__main__':
    net = TemCSPDarknet(dep_mul=0.33, wid_mul=0.5)
    input  = torch.randn(2, 3, 5, 128, 128)
    output = net(input)
