import collections

import torch
import torch.nn as nn
import torchvision

from .ae import AutoEncoder
from .convolution_module import ConvBlock
from .encoder_decoder import Block


class CACNet2(nn.Module):
    def __init__(
        self,
        pretrained_ae_path: str = '/home/enoshima/workspace/dip/CHSNet/pretrained/vitBackbone.pkl',
        detach_ae: bool = False,
    ):
        super().__init__()
        '''
        self.vit_encoder = AutoEncoder(
            img_size=384, 
            patch_size=16, 
            depth=8, 
            embed_dim=768, 
            out_chans=256
        )
        
        self.vit_encoder.load_state_dict(torch.load(
            pretrained_ae_path,
            map_location='cuda:0'
        )['model_state_dict'])
        if detach_ae:
            self.vit_encoder.requires_grad_(False)
        '''
        
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, 3, 4, 4)
        )
        
        self.vit_encoder = nn.Sequential(
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
        
        if not False:
            if True:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.vit_encoder.state_dict().items())):
                temp_key = list(self.vit_encoder.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.vit_encoder.load_state_dict(fsd)
        
        self.cnn_encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64, res_link=True),
            nn.AvgPool2d(2, 2),
            ConvBlock(64, 128),
            #ConvBlock(128, 128, res_link=True),
            nn.AvgPool2d(2, 2),
            ConvBlock(128, 256),
            #ConvBlock(256, 256, res_link=True),
            nn.AvgPool2d(2, 2),
            #ConvBlock(256, 256, res_link=True),
            ConvBlock(256, 512),
            nn.AvgPool2d(2, 2),
            #ConvBlock(256, 256, res_link=True),
            ConvBlock(512, 512, res_link=True),
            nn.AvgPool2d(2, 2),
            #ConvBlock(256, 256, res_link=True),
            ConvBlock(512, 512, res_link=True),
            nn.AvgPool2d(2, 2),
        )
        
        self.neck0 = nn.Sequential(
            Block(512, 2),
            Block(512, 2),
            Block(512, 2),
            Block(512, 2)
        )

        self.similarity_weight1 = nn.Linear(512, 1024)
        self.similarity_weight2 = nn.Linear(512, 1024)
        
        self.neck = nn.Sequential(
            Block(544, 4),
            Block(544, 4),
            Block(544, 4),
            Block(544, 4)
        )
        
        self.decoder = nn.Sequential(
            ConvBlock(544, 256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(256, 128),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(128, 64),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            #nn.Sigmoid()
        )
        
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
       
    def forward(self, query, sample):
        x = self.vit_encoder(query)
        y = self.cnn_encoder(sample)
        
        sz = x.shape[-1]
        #'''
        x = torch.flatten(x, -2, -1)
        y = torch.flatten(y, -2, -1)
        k = torch.cat([x, y], -1)
        k = k.unsqueeze(-1).permute(0, 2, 3, 1)
        k = self.neck0(k).squeeze(2).permute(0, 2, 1)
        x, y = k.split([sz ** 2, 1], dim=-1)
        x = x.view(x.shape[0], x.shape[1], sz, sz)
        y = y.view(y.shape[0], y.shape[1], 1, 1)
        #'''
        x1 = self.similarity_weight1(x.permute(0, 2, 3, 1))
        y1 = self.similarity_weight2(y.permute(0, 2, 3, 1))
        
        
        out = torch.bmm(x1.reshape((x1.shape[0], 24 ** 2, 1024)), y1.squeeze(2).permute(0, 2, 1))
        out = out.reshape((x1.shape[0], 24, 24, 1)).permute(0, 3, 1, 2)
        s = out
        out = out.repeat(1, 32, 1, 1)
        #out.repeat(64)
        #print(out.shape)
        
        #z = self.ca(x.permute(0, 2, 3, 1), out.permute(0, 2, 3, 1))
        #z = nn.LayerNorm(256)(z)
        x = x.permute(0, 2, 3, 1)
        out = out.permute(0, 2, 3, 1)
        z = torch.cat([x, out], dim=-1)
        z = self.neck(z)
        #print(x.shape)
        #z = torch.cat([x, out], dim=1)
        #print(z.shape)
        #return nn.UpsamplingBilinear2d(scale_factor=2)(s) + self.decoder(z.permute(0, 3, 1, 2))
        return self.decoder(z.permute(0, 3, 1, 2))

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        
    def forward(self, query, sim):
        B, H, W, _ = query.shape
        q = self.q(query).reshape(B, H * W, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        kv = self.kv(sim).reshape(B, H * W, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        q = q.reshape(1, B * self.num_heads, H * W, -1)
        k, v = kv.reshape(2, B * self.num_heads, H * W, -1).unbind(0)
        
        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

if __name__ == '__main__':
    x = torch.randn((8, 3, 384, 384))
    sample = torch.randn((8, 3, 32, 32))
    
    ae = AutoEncoder(img_size=400, patch_size=16, depth=8, embed_dim=144, out_chans=48)
    ae.load_state_dict(torch.load('/home/enoshima/workspace/dip/CHSNet/pretrained/epoch5.pkl',
        map_location='cuda:0'
    )['model_state_dict'])
    
    #print(ae(x).shape)
    #x1 = nn.Linear(48, 128)(ae(x).permute(0, 2, 3, 1))
    #print(x1.shape)
    
    n = CACNet2()
    #print(n.cnn_encoder(sample).shape)
    #n1 = nn.Linear(48, 128)(n.cnn_encoder(sample).permute(0, 2, 3, 1))
    #print(n1.shape)
    
    #print((x1 @ n1.squeeze()).shape)
    #out = torch.bmm(x1.reshape((8, 625, 128)), n1.squeeze(2).permute(0, 2, 1))
    #out = out.reshape((8, 25, 25, 1))
    #print(out.shape)

    
    #y = torch.randn((8, 96, 25, 25))
    #print(n.decoder(y).shape)
    
    
    n(x, sample)
    print(n(x, sample).shape)
    
