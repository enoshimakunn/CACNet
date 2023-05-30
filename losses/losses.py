import torch
import torch.nn as nn
from einops import rearrange


class DownMSELoss(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, dmap, gt_density):
        #print(dmap.shape)
        gt_density = self.avgpooling(gt_density) * self.tot
        #print(gt_density.shape)
        #gt_density = gt_density 
        #dmap *= 10
        '''
        print(torch.max(gt_density))
        print(torch.mean(gt_density))
        print(torch.min(gt_density))
        print(torch.max(dmap))
        print(torch.mean(dmap))
        print(torch.min(dmap))
        '''
        b, c, h, w = dmap.size()
        assert gt_density.size() == dmap.size()
        return self.mse(dmap, gt_density)
