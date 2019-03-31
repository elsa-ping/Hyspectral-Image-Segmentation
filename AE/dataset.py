#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:29:38 2019

@author: liang
"""

# load data
import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm


np.random.seed(0)

class Dataset(data.Dataset):
  
  def __init__(self,data_type='region'):
    super(Dataset,self).__init__()
    self.data_type = data_type
    self.data_names = os.path.join('../dataset_1/test',self.data_type,'%s.npz'%self.data_type)
    self.data_infos = np.load(self.data_names)
    self.images = self.data_infos['data']
    self.coords = self.data_infos['coord']
    
  def __len__(self):
    return len(self.images)
  
  
  def __getitem__(self,idx):
    image = self.images[idx]
    coord = self.coords[idx]
    # simple normalization
    # every pixel along depth
    mean,std = np.mean(image,axis=(0,),keepdims=True),np.std(image,axis=(0,),keepdims=True)
    image = (image-mean)/(std+1e-12)
    # convert to pytorch tensor
    image = torch.from_numpy(image).float()
    coord = torch.from_numpy(coord).int()
    return image,coord
  
def DataLoader(data_type='region',batchsize=512,sampling=True):
  testdataset = Dataset(data_type)
  dataloader = data.DataLoader(testdataset,
                               batch_size=batchsize,
                               num_workers=16,
                               shuffle=True,
                               pin_memory=True)
  return dataloader


class RegularizerDataset(data.Dataset):
  
  def __init__(self):
    super(RegularizerDataset,self).__init__()
    self.root_path = './svm_train_infos.npy'
    self.data_info = np.load(self.root_path).item()
    self.datas = np.empty(shape=(0,8,32,32),dtype=np.float32)
    self.labels = np.empty(shape=(0,),dtype=np.int16)
    # load data
    for key,value in tqdm(self.data_info.items(),desc='Loading Dataset!!!'):
      num = value.shape[0]
      label = [key]*num
      self.datas = np.concatenate((self.datas,value),axis=0)
      self.labels = np.concatenate((self.labels,np.array(label,dtype=np.int16)))
    
  
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self,idx):
    image = self.datas[idx]
    label = self.labels[idx]
    # normalize image
    mean,std = np.mean(image,axis=(0,),keepdims=True),np.std(image,axis=(0,),keepdims=True)
    image = (image-mean)/(std+1e-12)
    # random data augment
     # random rorate
    idx = np.random.randint(0,6)
    if idx==0:
      image = np.rot90(image,k=1,axes=(1,2))
    elif idx==2:
      image = np.rot90(image,k=2,axes=(1,2))
    elif idx==4:
      image = np.rot90(image,k=3,axes=(1,2))
    # random flip
    if np.random.uniform()>0.5:
      # random flip updown
      image = image[:,::-1,:]
    if np.random.uniform()>0.5:
      # random flip left-right
      image = image[:,:,::-1]
    # convert to tensor
    image = torch.from_numpy(image.copy())
    label = torch.from_numpy(np.array(label))
    return image,label
  
  
def RegularizerDataLoader(batchsize=256):
  dataset = RegularizerDataset()
  dataloader = data.DataLoader(dataset,
                               batch_size=batchsize,
                               shuffle=True,
                               num_workers=16,
                               pin_memory=True)
  return dataloader 
 


   
if __name__ == '__main__':
  dataloader = DataLoader(data_type='margin',batchsize=2048)
#  dataiter = iter(dataloader)
#  imgs,labs = dataiter.next()
#  print(imgs.shape,labs.shape)
  for imgs,labs in dataloader:
    print(imgs.shape,labs.shape)
  
