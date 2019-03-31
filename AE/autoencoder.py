#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:46:34 2019

@author: liang
"""

# define auto encoder
import torch
from torch import nn

class Encoder(nn.Module):
  
  def __init__(self):
    super(Encoder,self).__init__()
    self.features = nn.Sequential(
                                # conv1_block ->16x16x32
                                nn.Conv2d(8,32,3,padding=1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.Tanh(),
                                nn.Conv2d(32,32,3,padding=1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.Tanh(),
                                nn.MaxPool2d(2,2),
                                # conv2_block ->8x8x64
                                nn.Conv2d(32,64,3,padding=1,bias=False),
                                nn.BatchNorm2d(64),
                                nn.Tanh(),
                                nn.Conv2d(64,64,3,padding=1,bias=False),
                                nn.BatchNorm2d(64),
                                nn.Tanh(),
                                nn.MaxPool2d(2,2),
                                # conv3_block ->1x1x128
                                nn.Conv2d(64,128,8,bias=False),
                                nn.BatchNorm2d(128),
                                nn.Tanh()
                              )
    
  def forward(self,xs):
    xs = self.features(xs)
    return xs
  
  
class Decoder(nn.Module):
  
  def __init__(self):
    super(Decoder,self).__init__()
    self.features = nn.Sequential(
                                  # deconv block1 ->8x8x64
                                  nn.ConvTranspose2d(128,64,8,bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.Tanh(),
                                  # deconv block2 ->16x16x32
                                  nn.ConvTranspose2d(64,64,4,2,padding=1,bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.Tanh(),
                                  nn.Conv2d(64,32,3,padding=1,bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.Tanh(),
                                  # deconv block3 ->32x32x8
                                  nn.ConvTranspose2d(32,32,4,2,padding=1,bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.Tanh(),
                                  nn.Conv2d(32,8,3,padding=1,bias=False),
                                  nn.BatchNorm2d(8)
        )
    
  def forward(self,xs):
    xs = self.features(xs)
    return xs
  

if __name__ == '__main__':
  xs = torch.randn((128,8,32,32))
  encoder = Encoder()
  decoder = Decoder()
  xs_hat = decoder(encoder(xs))
  print(xs_hat.shape)
