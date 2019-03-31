#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:46:36 2019

@author: liang
"""

# define auto encoder
import torch
from torch import nn

# note for checkpoints2,we use Tanh() activation
class Encoder(nn.Module):
  
  def __init__(self):
    super(Encoder,self).__init__()
    self.features = nn.Sequential(
                                
                                # conv1_block ->5x5x32
                                nn.Conv2d(8,12,3,bias=False),
                                nn.BatchNorm2d(12),
                                nn.Tanh(),
                                nn.InstanceNorm2d(12),
                                nn.Tanh(),
                                # conv2_block ->3x3x64
                                nn.Conv2d(12,24,3,bias=False),
                                nn.BatchNorm2d(24),
                                nn.Tanh(),
                                # conv3_block ->1x1x128
                                nn.Conv2d(24,32,3,bias=False),
                                nn.BatchNorm2d(32),
                                nn.Tanh()
                              )
    # initialization
    self.features.apply(weights_init)
    
  def forward(self,xs):
    xs = self.features(xs)
    return xs
  
  
class Decoder(nn.Module):
  
  def __init__(self):
    super(Decoder,self).__init__()
    self.features = nn.Sequential(
                                  # deconv block1 ->3x3x64
                                  nn.ConvTranspose2d(32,24,3,bias=False),
                                  nn.BatchNorm2d(24),
                                  nn.Tanh(),
                                  # deconv block2 ->5x5x32
                                  nn.ConvTranspose2d(24,12,3,bias=False),
                                  nn.BatchNorm2d(12),
                                  nn.Tanh(),
                                  # deconv block3 ->7x7x8
                                  nn.ConvTranspose2d(12,8,3,bias=False),
                                  nn.BatchNorm2d(8),
        )
    
    
  def forward(self,xs):
    xs = self.features(xs)
    return xs

 
def weights_init(m):
  
  if isinstance(m, nn.Conv2d):
    nn.init.orthogonal_(m.weight, gain=0.6)
    try:
      nn.init.constant_(m.bias, 0.01)
    except:
      pass
  elif isinstance(m,nn.ConvTranspose2d):
    nn.init.orthogonal_(m.weight, gain=0.6)
    try:
      nn.init.constant_(m.bias, 0.01)
    except:
      pass

  
if __name__ == '__main__':
  xs = torch.randn((128,8,7,7))
  encoder = Encoder()
  decoder = Decoder()
  xs_hat = decoder(encoder(xs))
  print(xs_hat.shape)