import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .utils import Conv, CBAM, SimAM, Bottleneck

class imgfeature_extractor(nn.Module):
    """input size: 960*640*3"""
    def __init__(self, input_height=640, input_weight=960):
        super(imgfeature_extractor, self).__init__()
        self.input_height = input_height
        self.input_weight = input_weight

        