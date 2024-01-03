import torch.nn as nn
import numpy as np
from .decoding import Decodeing
from . import hr


class Vltenet(nn.Module):
    def __init__(self, pretrained, final_kernel):
        super(Vltenet, self).__init__()

        self.base_network = hr.hrnet18(pretrained=pretrained)
        self.Decodeing = Decodeing(final_kernel, 256, 64)

    def forward(self, x):

        x = self.base_network(x)
        feature_dict = self.Decodeing(x)
        return feature_dict