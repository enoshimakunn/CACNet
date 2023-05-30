from typing import List, Optional, Tuple, Type

import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST

from .encoder_decoder import Decoder, Encoder


class AutoEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )
        
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )
    
    def forward(self, x: torch.Tensor):
        embedding = self.encoder(x)
        return embedding
    
    def decode(self, x: torch.Tensor):
        return self.decoder(x)
    
    def reconstruct(self, x: torch.Tensor):
        x = self.encoder(x)
        #x += torch.randn_like(x)
        std = torch.exp(0.5 * x)
        eps = torch.randn_like(std)
        x = eps * std + x
        o = torch.mean(self.decoder(x), 1)
        o = o.unsqueeze(1)
        #print(o.shape)
        return o
    
if __name__ == '__main__':
    t = transforms.Compose([
        transforms.ToTensor(),
        
        transforms.CenterCrop(32),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = cv.imread('test.png')
    data = torch.unsqueeze(t(img), 0).cuda()
    #data = data.repeat(16, 1, 1, 1)
    
    model = AutoEncoder(img_size=32, patch_size=4, depth=4, embed_dim=192, out_chans=64).cuda()
    
    train_data = MNIST(train=True, transform=t, root='dataset/', download=True)

    print(train_data[0])