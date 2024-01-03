import torch.nn as nn
import torch
from .ffca import FFCA

class Decodeing(nn.Module):
    def __init__(self, final_kernel, head_conv, channel):
        super(Decodeing, self).__init__()

        # hr 18
        self.dec_c2 = FFCA(36, 18, batch_norm=True)
        self.dec_c3 = FFCA(72, 36, batch_norm=True)
        self.dec_c4 = FFCA(144, 72, batch_norm=True)

        channel =18
        head_conv = 256
        self.fc_hm = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(head_conv, 1, kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True))
        self.fc_vec = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1,
                                     padding=final_kernel // 2, bias=True))

    def forward(self, x):

        feature_dict = {}
        p3_combine = self.dec_c4(x[-1], x[-2])
        p2_combine = self.dec_c3(p3_combine, x[-3])
        p1_combine = self.dec_c2(p2_combine, x[-4])

        feature_dict['hm'] = self.fc_hm(p1_combine)
        feature_dict['hm'] = torch.sigmoid(feature_dict['hm'])
        feature_dict['vec_ind'] = self.fc_vec(p1_combine)

        return feature_dict