import collections

import torch
import torch.nn as nn
import torchvision
from models.convolution_module import ConvBlock, OutputNet
from models.transformer_module import Transformer

from .ae import AutoEncoder

#from convolution_module import ConvBlock, OutputNet

#from transformer_module import Transformer


class VGG16Trans2(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False):
        super().__init__()
        self.scale_factor = 16//dcsize
        
        self.encoder = AutoEncoder(
            img_size=384, 
            patch_size=16, 
            depth=8, 
            embed_dim=768, 
            out_chans=256
        )
        
        self.em = nn.Conv2d(256, 512, 1, 1)
        
        self.encoder.load_state_dict(torch.load(
            '/home/enoshima/workspace/dip/CHSNet/pretrained/vitBackbone.pkl',
            map_location='cuda:0'
        )['model_state_dict'])
        
        self.tran_decoder = Transformer(layers=4)
        self.tran_decoder_p2 = OutputNet(dim=512)

        # self.conv_decoder = nn.Sequential(
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        # )
        # self.conv_decoder_p2 = OutputNet(dim=512)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, sample):
        raw_x = self.em(self.encoder(x))
        bs, c, h, w = raw_x.shape

        # path-transformer
        x = raw_x.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        x = self.tran_decoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x)

        return y

if __name__ == '__main__':
    x = torch.randn((8, 3, 384, 384))
    m = VGG16Trans2(8)
    print(m(x).shape)