#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:18:12 2019

@author: liang
"""

# extract features for svm
import torch
from torch.nn import functional as F
from autoencoder import Encoder
from dataset import RegularizerDataLoader
from tqdm import tqdm
import os
import numpy as np

features_dir = './svm_features'
if not os.path.exists(features_dir):
  os.makedirs(features_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def extract():
  """
  return batchx4 numpy array
  row_index,col_index,predicts,probs
  """
  # config
  batchsize=2048
  # model
  encoder = Encoder()
  ckpt = './checkpoints2/Encoder-epoch-1-loss:0.065.pth'
  # load parameters
  encoder.load_state_dict(torch.load(ckpt))
  testloader = RegularizerDataLoader(batchsize)
  # evaluation
  encoder.cuda()
  encoder.eval()
  
  # testing
  with torch.no_grad():
    svm_train_features = np.empty(shape=[0,128],dtype=np.float32)
    svm_train_labels = np.empty(shape=(0,),dtype=np.uint8)
    for epoch in tqdm(range(10),desc='Testing!!!'):
      for image,label in testloader:
        image = image.cuda()
        output = encoder(image)
        bs = output.size(0)
        output = output.view(bs,-1)
        # l2-normalize descriptors
        output = F.normalize(output,p=2,dim=-1)
        svm_train_features = np.concatenate((svm_train_features,output.cpu().numpy()),axis=0)
        svm_train_labels = np.concatenate((svm_train_labels,label))
   
    print(svm_train_features.shape,svm_train_labels.shape)
    np.savez('./svm_train_data.npz',features=svm_train_features,labels=svm_train_labels)   
      
  print('\nExtracting Finished!!!')
  

if __name__ == '__main__':
  extract()  