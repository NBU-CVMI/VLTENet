import torch.nn as nn
import torch.nn.functional as F
import torch

# ffca
class FFCA(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False):
        super(FFCA, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c_low, c_up, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(c_up, c_up, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        x = torch.cat((x_up, x_low), 1)
        x_avg = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        x_max = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        weight = self.sigmoid(x_max + x_avg)
        x_up = x_up*weight

        return self.cat_conv(torch.cat((x_up, x_low), 1))


