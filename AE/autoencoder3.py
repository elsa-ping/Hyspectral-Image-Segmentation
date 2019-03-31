#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:40:01 2019

@author: liang
"""

# define auto encoder
import torch
from torch import nn

class Encoder(nn.Module):
  
  def __init__(self):
    super(Encoder,self).__init__()
    self.branch1 = nn.Sequential(
                                # conv1_block ->5x5x12
                                nn.Conv2d(8,12,3,bias=True),
                                nn.BatchNorm2d(12),
                                nn.Tanh(),
                                # conv2_block ->3x3x24
                                nn.Conv2d(12,24,3,bias=True),
                                nn.BatchNorm2d(24),
                                nn.Tanh(),
                                # conv3_block ->1x1x32
                                nn.Conv2d(24,32,3,bias=True),
                                nn.BatchNorm2d(32),
                                nn.Tanh()
                              )
    self.branch2 = nn.Sequential(
                                # conv1_block ->3x3x16
                                nn.Conv2d(8,16,5,bias=True),
                                nn.BatchNorm2d(16),
                                nn.Tanh(),
                                # conv2_block ->1x1x32
                                nn.Conv2d(16,32,3,bias=True),
                                nn.BatchNorm2d(32),
                                nn.Tanh()
                                )
    
    self.branch1.apply(weights_init)
    self.branch2.apply(weights_init)

  def forward(self,xs):
    out1 = self.branch1(xs)
    out2 = self.branch2(xs)
    out = torch.cat((out1,out2),dim=1)
    return out
  
  
class Decoder(nn.Module):
  
  def __init__(self):
    super(Decoder,self).__init__()
    self.features = nn.Sequential(
                                  # deconv block1 ->3x3x64
                                  nn.ConvTranspose2d(64,32,3,bias=True),
                                  nn.BatchNorm2d(32),
                                  nn.Tanh(),
                                  # deconv block2 ->5x5x32
                                  nn.ConvTranspose2d(32,16,3,bias=True),
                                  nn.BatchNorm2d(16),
                                  nn.Tanh(),
                                  # deconv block3 ->7x7x8
                                  nn.ConvTranspose2d(16,8,3,bias=True),
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