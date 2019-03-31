#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:43:18 2019

@author: liang
"""

import torch
from torch.nn import functional as F
from model2 import TinyNet
from dataset import TestLoader
from tqdm import tqdm
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_type='margin'

result_dir = os.path.join('./results',data_type)

if not os.path.exists(result_dir):
  os.makedirs(result_dir)


def test():
  """
  return batchx4 numpy array
  row_index,col_index,predicts,probs
  """
  # config
  batchsize=2048
  # model
  tinynet1 = TinyNet().cuda()
  tinynet2 = TinyNet().cuda()
  ckpt1 = './checkpoints4/model1-Num-4.pth'
  ckpt2 = './checkpoints4/model2-Num-4.pth'
  # load parameters
  tinynet1.load_state_dict(torch.load(ckpt1))
  tinynet2.load_state_dict(torch.load(ckpt2))
  testloader = TestLoader(data_type,batchsize)
  # evaluation
  tinynet1.eval()
  tinynet2.eval()
  
  # testing
  with torch.no_grad():
    for idx,(image,coord) in enumerate(tqdm(testloader,desc='Testing!!!')):
      image = image.cuda()
      output1 = tinynet1(image)
      output2 = tinynet2(image)
      output = 0.5*(output1+output2)
      output = F.softmax(output,dim=-1)
      # ----------------------# 
      probs,predicts = torch.max(output,dim=-1)
      probs = probs.cpu().numpy()
      predicts = predicts.cpu().numpy()
      coord = coord.numpy()
      mask = probs<0.95
      predicts[mask] = 0
      np.savez(os.path.join(result_dir,'%d.npz'%(idx+1)),coord=coord,predict=predicts,prob=probs)
      
  print('\nTest Finished!!!')


if __name__ == '__main__':
  test()  
      
  

