import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from models.ae import AutoEncoder

ae = AutoEncoder(
    img_size=384,
    patch_size=16,
    depth=8
)

ae.load_state_dict(torch.load('/home/enoshima/workspace/dip/CHSNet/pretrained/epoch2.pkl', map_location='cpu')['model_state_dict'])
print(ae)