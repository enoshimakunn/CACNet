import collections

import torch
import torch.nn as nn
import torchvision
from convolution_module import ConvBlock, OutputNet
from model import CrossAttention
from transformer_module import Transformer

from .ae import AutoEncoder

#from models.convolution_module import ConvBlock, OutputNet
#from models.model import CrossAttention
#from models.transformer_module import Transformer



class VGG16Trans_m(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False):
        super().__init__()
        self.scale_factor = 16//dcsize
        '''
        self.encoder = nn.Sequential(
            ConvBlock(cin=3, cout=64),
            ConvBlock(cin=64, cout=64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=64, cout=128),
            ConvBlock(cin=128, cout=128),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=128, cout=256),
            ConvBlock(cin=256, cout=256),
            ConvBlock(cin=256, cout=256),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=256, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
            ConvBlock(cin=512, cout=512),
        )
        '''
        self.sample_encoder = nn.Sequential(
            ConvBlock(3, 32),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512), 
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvBlock(512, 512),
        )
        
        self.ca = CrossAttention(512, 8)

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
        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.encoder.state_dict().items())):
                temp_key = list(self.encoder.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.encoder.load_state_dict(fsd)

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
        
        raw_x = self.encoder(x)
        y = self.sample_encoder(sample)
        
        bs, c, h, w = raw_x.shape
        raw_x = raw_x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        print(raw_x.shape)
        print(y.shape)
        
        
        #z = self.ca(raw_x, y)

        # path-transformer
        raw_x
        x = raw_x.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        x = self.tran_decoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x)

        return y

if __name__ == '__main__':
    x = torch.randn((8, 3, 384, 384))
    y = torch.randn((8, 3, 32, 32))
    m = VGG16Trans_m(8)
    print(m(x, y).shape)