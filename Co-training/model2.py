#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:45:02 2019

@author: liang
"""

import torch
from torch import nn
import torch.nn.functional as F

class TinyNet(nn.Module):
  
  def __init__(self):
    super(TinyNet,self).__init__()
    self.features = nn.Sequential(
                                # conv1_block ->16x16x16
                                nn.Conv2d(8,16,3,padding=1,bias=False),
                                nn.BatchNorm2d(16),
                                nn.Tanh(),
                                nn.Conv2d(16,16,3,padding=1,bias=False),
                                nn.BatchNorm2d(16),
                                nn.Tanh(),
                                nn.InstanceNorm2d(16),                             
                                nn.Tanh(),
                                nn.MaxPool2d(2,2),
                                # conv2_block ->8x8x32
                                nn.Conv2d(16,32,3,padding=1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.Tanh(),
                                nn.Conv2d(32,32,3,padding=1,bias=False),
                                nn.BatchNorm2d(32),
                                nn.Tanh(),                      
                                nn.MaxPool2d(2,2),
                                # conv3_block ->1x1x32
                                nn.AdaptiveAvgPool2d((1,1)),
                              )
    self.classifier = nn.Sequential(
                                nn.Linear(32,64),
                                nn.Tanh(),
                                nn.Linear(64,4))
    
    
  def forward(self,xs):
    bs = xs.size(0)
    xs = self.features(xs)
    xs = xs.view(bs,-1)
    xs = self.classifier(xs)
    return xs
  
if __name__ == '__main__':
  tinynet = TinyNet()
  print(tinynet)
  xs = torch.randn(size=[128,8,32,32])
  out = tinynet(xs)
  print(out.size())