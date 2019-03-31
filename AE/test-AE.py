#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:33:44 2019

@author: liang
"""

import torch
from torch.nn import functional as F
from autoencoder import Encoder
from dataset import DataLoader
from tqdm import tqdm
import os
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

data_type='region'

'''
Threshold selection
1.checkpoint2    accu:41.68  threshold=0.90
2.checkpoint2_1  accu:41.71  threshold=0.70
3.checkpoint2_1  accu:41.50  threshold=0.60
4.checkpoint2_1  accu:41.69  threshold=0.80
'''

threshold1 = 0.50 # 0.9
threshold2 = 0.70 

feature_dir = './svm_features'

if not os.path.exists(feature_dir):
  os.makedirs(feature_dir)

def AE_test():
  """
  return batchx4 numpy array
  row_index,col_index,predicts,probs
  """
  
  result_dir = os.path.join('./results',data_type)

  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
  # config
  batchsize=2048
  # model
  encoder = Encoder()
  ckpt = './checkpoints2/Encoder-epoch-2-loss:0.048.pth'
  # load parameters
  encoder.load_state_dict(torch.load(ckpt))
#  testloader = DataLoader(data_type,batchsize)
  # evaluation
  encoder.cuda()
  encoder.eval()
  
  # testing
  with torch.no_grad():
    # compute cluster center
    infos = np.load('./train_infos.npy').item()
    cls1_patches = infos[1]
    cls2_patches = infos[2]
    cls3_patches = infos[3]
    bs1 = len(cls1_patches)
    bs2 = len(cls2_patches)
    bs3 = len(cls3_patches)
    # normalization
    cls1_mean,cls1_std = np.mean(cls1_patches,axis=(1),keepdims=True),np.std(cls1_patches,axis=(1),keepdims=True)
    cls2_mean,cls2_std = np.mean(cls2_patches,axis=(1),keepdims=True),np.std(cls2_patches,axis=(1),keepdims=True)
    cls3_mean,cls3_std = np.mean(cls3_patches,axis=(1),keepdims=True),np.std(cls3_patches,axis=(1),keepdims=True)
    cls1_patches = (cls1_patches-cls1_mean)/cls1_std
    cls2_patches = (cls2_patches-cls2_mean)/cls2_std
    cls3_patches = (cls3_patches-cls3_mean)/cls3_std
    
    # convert to tensor
    cls1_patches = torch.from_numpy(cls1_patches).float().cuda()
    cls2_patches = torch.from_numpy(cls2_patches).float().cuda()
    cls3_patches = torch.from_numpy(cls3_patches).float().cuda()
    # descriptors
    cls1_desc = encoder(cls1_patches).view(bs1,-1)
    cls2_desc = encoder(cls2_patches).view(bs2,-1)
    cls3_desc = encoder(cls3_patches).view(bs3,-1)

    # assert cls3_desc.size(0)==bs3 and cls3_desc.size(1)==64 # or 32
    # compute center vector 
    cls1_center_desc = F.normalize(torch.sum(cls1_desc,dim=0,keepdim=True))
    cls2_center_desc = F.normalize(torch.sum(cls2_desc,dim=0,keepdim=True))
    cls3_center_desc = F.normalize(torch.sum(cls3_desc,dim=0,keepdim=True))
    
    
    cls_center_desc = torch.cat((cls1_center_desc,
                                 cls2_center_desc,
                                 cls3_center_desc),
                                 dim=0)
    # assert cls_center_desc.size(0)==3 and cls_center_desc.size(1)==64 # or 32
    # print(cls_center_desc)
    simi_matrix = cls_center_desc.mm(cls_center_desc.t())
    print(simi_matrix)
    
    '''
    for idx,(image,coord) in enumerate(tqdm(testloader,desc='Testing!!!')):
      image = image.cuda()
      output = encoder(image)
      bs = output.size(0)
      output = output.view(bs,-1)
      # l2-normalize descriptors
      output = F.normalize(output,p=2,dim=-1)
      # compute cosine similarity matrix
      simi_matrix = output.mm(cls_center_desc.t())
      
      # get predicts
      probs,predicts = torch.max(simi_matrix,dim=-1)
      probs = probs.cpu().numpy()
      predicts = predicts.cpu().numpy()
      coord = coord.numpy()
      
      # assign label
      for i in range(bs):
        if probs[i]>threshold2:
          predicts[i] += 1
        else:
          predicts[i] = 0
      np.savez(os.path.join(result_dir,'%d.npz'%(idx+1)),coord=coord,predict=predicts,prob=probs)
    
#      max_simi,_ = torch.max(simi_matrix,dim=-1)
#      min_simi,_ = torch.min(simi_matrix,dim=-1)
#      max_simi = max_simi.cpu().numpy()
#      min_simi = min_simi.cpu().numpy()
#      coord = coord.numpy()
#      fg_coords = []
#      fg_features = np.empty(shape=(0,32),dtype=np.float32)
#      for i in range(bs):
#        if min_simi[i]>threshold1 or max_simi[i]>threshold2:
#          fg_coords.append(coord[i])
#          fg_features = np.concatenate((fg_features,output[i:i+1].cpu().numpy()),axis=0)
#      np.savez(os.path.join(feature_dir,'%s_features_%d.npz'%(data_type,idx)),coord=fg_coords,feature=fg_features)
      
  print('\nTest Finished!!!')
  '''

def extract_features_for_SVM_test():
  """
  return batchx4 numpy array
  row_index,col_index,predicts,probs
  """
  features_dir = os.path.join('./svm_features',data_type)

  if not os.path.exists(features_dir):
    os.makedirs(features_dir)
  # config
  batchsize=2048
  # model
  encoder = Encoder()
  ckpt = './checkpoints2/Encoder-epoch-1-loss:0.065.pth'
  # load parameters
  encoder.load_state_dict(torch.load(ckpt))
  testloader = DataLoader(data_type,batchsize)
  # evaluation
  encoder.cuda()
  encoder.eval()
  
  # testing
  with torch.no_grad():
    for idx,(image,coord) in enumerate(tqdm(testloader,desc='Extracting Features for SVM Testing!!!')):
      image = image.cuda()
      output = encoder(image)
      bs = output.size(0)
      output = output.view(bs,-1)
      # l2-normalize descriptors
      output = F.normalize(output,p=2,dim=-1)
      output = output.cpu().numpy()
      coord = coord.numpy()
      np.savez(os.path.join(features_dir,'%d.npz'%(idx+1)),coord=coord,feature=output)
      
  print('\nTest Finished!!!')

if __name__ == '__main__':
  # AE_test()  
  extract_features_for_SVM_test()
