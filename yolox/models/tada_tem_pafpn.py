#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from .temporal_darknet import TemCSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .mamba import ShiftVSS

class MambaFuse(nn.Module):
    def __init__(self, in_channels): # 128, 256, 512
        super().__init__()
        self.in_channels = in_channels
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1_1x1 = BaseConv(in_channels=in_channels[1], out_channels=in_channels[1], ksize=1, stride=1) # 512 -> 256
        self.conv2_1x1 = BaseConv(in_channels=in_channels[0], out_channels=in_channels[0], ksize=1, stride=1) # 256 -> 128

        self.vssblocks1 = ShiftVSS(hidden_dim=in_channels[1], drop_path=0.2, n=1) # 256
        self.vssblocks2 = ShiftVSS(hidden_dim=in_channels[0], drop_path=0.2, n=1) # 128

        # self.vssblocks1 = ShiftVSSAblaVivim(hidden_dim=in_channels[1], drop_path=0.2, n=1) # 256
        # self.vssblocks2 = ShiftVSSAblaVivim(hidden_dim=in_channels[0], drop_path=0.2, n=1) # 128

        # self.vssblocks1 = ShiftVSSAblaVideoMamba(hidden_dim=in_channels[1], drop_path=0.2, n=1) # 256
        # self.vssblocks2 = ShiftVSSAblaVideoMamba(hidden_dim=in_channels[0], drop_path=0.2, n=1) # 128

        # self.vssblocks1 = ShiftVSSWithoutDiff(hidden_dim=in_channels[1], drop_path=0.2, n=1) # 256
        # self.vssblocks2 = ShiftVSSWithoutDiff(hidden_dim=in_channels[0], drop_path=0.2, n=1) # 128

        # self.vssblocks1 = ShiftVSSDiffWithVideoMamba(hidden_dim=in_channels[1], drop_path=0.2, n=1) # 256
        # self.vssblocks2 = ShiftVSSDiffWithVideoMamba(hidden_dim=in_channels[0], drop_path=0.2, n=1) # 128

        # self.vssblocks1 = ShiftVSSDiffWithVivim(hidden_dim=in_channels[1], drop_path=0.2, n=1) # 256
        # self.vssblocks2 = ShiftVSSDiffWithVivim(hidden_dim=in_channels[0], drop_path=0.2, n=1) # 128

        # self.maxpool_tem = nn.MaxPool3d([3, 1, 1])
        # self.avgpool_tem = nn.AvgPool3d([3, 1, 1])

        self.maxpool_tem = nn.AdaptiveMaxPool3d((1, None, None))
        self.avgpool_tem = nn.AdaptiveAvgPool3d((1, None, None))


    def forward(self, xin):
        [x2, x1, x0] = xin # [b, 128, t, 4h, 4w] [256, 2w, 2h] [512, w, h]
        num_frames = x2.shape[2]
        cur_x2 = x2[:, :, -1, :, :]
        x2_out = []
        for i in range(num_frames-1):
            out2 = self.vssblocks2(cur_x2.permute(0, 2, 3, 1), x2[:, :, i, :, :].permute(0, 2, 3, 1))
            # [b, c, h, w] -> [b, h, w, c]
            out2 = self.conv2_1x1(out2.permute(0, 3, 1, 2)) # [b, h, w, c] -> [b, c, h, w]
            x2_out.append(out2)
        x2_out = torch.stack(x2_out, dim=2)
        x2_outmax = self.maxpool_tem(x2_out)
        x2_out = x2_outmax + self.avgpool_tem(x2_out)

        cur_x1 = x1[:, :, -1, :, :]
        x1_out = []
        for i in range(num_frames-1):
            out1 = self.vssblocks1(cur_x1.permute(0, 2, 3, 1), x1[:, :, i, :, :].permute(0, 2, 3, 1))
            out1 = self.conv1_1x1(out1.permute(0, 3, 1, 2))
            x1_out.append(out1)
        x1_out = torch.stack(x1_out, dim=2)
        x1_outmax = self.maxpool_tem(x1_out)
        x1_out = x1_outmax + self.avgpool_tem(x1_out)

        return x2_out.squeeze(2), x1_out.squeeze(2) # [128, 4h, 4w] [256, 4h, 4w]

class TemPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        freeze_backbone=False,
        fourier=False,
        pre_fourier=False
    ):
        super().__init__()
        self.backbone = TemCSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        self.freeze = freeze_backbone
        self.fourier = fourier
        self.pre_fourier = pre_fourier
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        channels = [int(ch * width) for ch in in_channels]
        self.mambafuse = MambaFuse(in_channels=channels)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat 512 -> 256

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        ) # 256 -> 128
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        # for layer in self.backbone.parameters():
        #     layer.requires_grad = False
        True_list = ['C3_p3', 'difference_model', 'cbam']
        False_list = ['backbone',]
        for layer in self.named_parameters():
            name = layer[0].split('.')[0]
            if name in False_list:
                layer[1].requires_grad = False
            else:
                layer[1].requires_grad = True
    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        out_features = self.backbone(input)
        diff_features = [out_features[f] for f in self.in_features]
        features_t = [out_features[f] for f in self.in_features]
        features = [f_t[:, :, -1, :, :] for f_t in features_t]
        [x2, x1, x0] = features
        [t2, t1] = self.mambafuse(diff_features)
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = f_out0 + t1
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)
        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = f_out1 + t2
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)
        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)
        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
