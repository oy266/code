import matplotlib.pyplot as plt
from torchvision import transforms
import os
import torch
import numpy as np
import math
import cv2
import random
def random_num(size,end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

def feature_visualization(features, features_num=512):
    save_dir = '/home/zmj/oyyj/feature_map/features_{}/'.format(features_num)
    features = features[2]


    pass

