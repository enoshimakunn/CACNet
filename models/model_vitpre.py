import torch
import torch.nn as nn

from .ae import AutoEncoder
from .convolution_module import ConvBlock
from .encoder_decoder import Block


class CACNet_vitpre(nn.Module):
    def __init__(
        self,
        pretrained_ae_path: str = '/home/enoshima/workspace/dip/CHSNet/pretrained/epoch5.pkl',
        detach_ae: bool = False,
    ):
        super().__init__()
        
        self.vit_encoder = AutoEncoder(
            img_size=400, 
            patch_size=16, 
            depth=8, 
            embed_dim=144, 
            out_chans=48
        )
        
        self.vit_encoder.load_state_dict(torch.load(
            pretrained_ae_path,
            map_location='cuda:0'
        )['model_state_dict'])
        
        if detach_ae:
            self.vit_encoder.requires_grad_(False)
        
        ''' 
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ConvBlock(16, 16, res_link=True),
            nn.Conv2d(16, 32, 3, 2, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 32, res_link=True),
            nn.Conv2d(32, 48, 3, 2, 1), 
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True), 
            ConvBlock(48, 48, res_link=True),
            #ConvBlock(256, 256, res_link=True),
            nn.Conv2d(48, 48, 3, 2, 1), 
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ConvBlock(48, 48, res_link=True),
            #ConvBlock(256, 256, res_link=True),
            nn.AvgPool2d(2)
        )
        '''
        
        self.cnn_encoder = nn.Sequential(
            ConvBlock(3, 16),
            nn.AvgPool2d(2, 2),
            ConvBlock(16, 32),
            nn.AvgPool2d(2, 2),
            ConvBlock(32, 48),
            nn.AvgPool2d(2, 2),
            ConvBlock(48, 48),
            nn.AvgPool2d(2, 2),
            ConvBlock(48, 48),
            nn.AvgPool2d(2, 2),
        )
        
        self.similarity_weight1 = nn.Linear(48, 256)
        self.similarity_weight2 = nn.Linear(48, 256)
        
        self.ca = CrossAttention()
        self.neck = nn.Sequential(
            Block(96, 8),
            Block(96, 8),
            Block(96, 8),
            Block(96, 8)
        )
        
        self.decoder = nn.Sequential(
            ConvBlock(96, 128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(128, 64),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(64, 32),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(32, 16),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            #nn.Sigmoid()
        )
        
        #self._initialize_weights()
    
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
        x1 = self.similarity_weight1(x.permute(0, 2, 3, 1))
        y1 = self.similarity_weight2(y.permute(0, 2, 3, 1))
                
        out = torch.bmm(x1.reshape((x1.shape[0], 25 ** 2, 256)), y1.squeeze(2).permute(0, 2, 1))
        out = out.reshape((x1.shape[0], 25, 25, 1)).permute(0, 3, 1, 2)
        out = out.repeat(1, 48, 1, 1)
        #out.repeat(64)
        #print(out.shape)
        
        x = x.permute(0, 2, 3, 1)
        out = out.permute(0, 2, 3, 1)
        z = torch.cat([x, out], dim=-1)
        z = self.neck(z)
        #z = torch.cat([x, out], dim=1)
        #print(z.shape)
        return self.decoder(z.permute(0, 3, 1, 2))

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int = 48,
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
    
    n = CACNet_vitpre()
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
    