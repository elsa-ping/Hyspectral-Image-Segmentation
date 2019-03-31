#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:13:05 2019

@author: liang
"""

# get train infos
import numpy as np
import re
import os

datadir = '../dataset_1/train_1'
dataNames = os.listdir(datadir)


filtpaths = [os.path.join(datadir,name) for name in dataNames]

train_infos = {1:[],2:[],3:[]}

def extract_infos():
  for path in filtpaths:
    data = np.load(path)
    image = data['image']
    label = data['label'].tolist()
    train_infos[label].append(image)
  train_infos[1] = np.array(train_infos[1],dtype=np.uint16)
  train_infos[2] = np.array(train_infos[2],dtype=np.uint16)
  train_infos[3] = np.array(train_infos[3],dtype=np.uint16)
  print(train_infos[1].shape,train_infos[2].shape,train_infos[3].shape)
  np.save('./train_infos.npy',train_infos)
  
  
def extract_bg_infos():
  import skimage.io as io
  img = io.imread('../image.tif')
  files = ['bg_coord_1.npy']
  # adding  bg class
  train_infos = np.load('./train_infos.npy').item()
  train_infos.update({0:np.empty(shape=[0,8,32,32],dtype=np.uint16)})
  for file in files:
    coords = np.abs(np.load(file).astype(np.int16))
    for coord in coords:
      x1,y1 = coord
      col1,col2 = x1-16,x1+16
      if col1<0:
        col1 = 0
        col2 = col1+32
      if col2>=17809:
        col2 = 17809
        col1 = col2-32
      patch = np.expand_dims(img[:,col1:col2,y1-16:y1+16],axis=0)
      # print(patch.shape)
      if patch.shape==(1,8,32,32):
        train_infos[0] = np.concatenate((train_infos[0],patch),axis=0)
  print(train_infos[0].shape)
  np.save('./svm_train_infos.npy',train_infos)
   

if __name__ == '__main__':
  # extract_infos()
  extract_bg_infos()