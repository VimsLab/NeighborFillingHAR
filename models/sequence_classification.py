import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from transformer_v1 import *

class PSTTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 dim, depth, heads, dim_head, dropout1,                                 # transformer
                 mlp_dim, num_classes, dropout2):                                       # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)
        # [B, L, n, 3], [B, L, C, n]

        features = features.permute(0, 1, 3, 2)
        # print(features.shape)
        # print(xyzs.shape)
        # exit()

        output = self.transformer(xyzs, features)
        # print(output.shape)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        # print(output.shape)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        # print(output.shape)
        output = self.mlp_head(output)
        # print(output.shape)

        return output
