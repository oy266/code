#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Myexp as MyExp


class MYExp(MyExp):
    def __init__(self):
        super(MYExp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = None
        self.train_ann = None
        self.val_ann = None
        self.gap = 1

        self.model_name = 'tada_tem_fpn'

        self.num_classes = 1

        self.max_epoch = 200
        self.data_num_workers = 4
        self.eval_interval = 1

        self.num_frames = 5

        self.output_dir = None

        self.freeze_backbone = True
