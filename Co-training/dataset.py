#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:00:12 2019

@author: liang
"""

# load data
import torch
from torch.utils import data
import numpy as np
import os


class TrainDataset(data.Dataset):
  
  def __init__(self,root_path):
    super(TrainDataset,self).__init__()
    self.root_path = root_path
    self.file_names = os.listdir(self.root_path)
    self.file_paths = [os.path.join(self.root_path,name) for name in self.file_names]
    
  def __len__(self):
    return len(self.file_paths)
  
  
  def __getitem__(self,idx):
    file_path = self.file_paths[idx]
    # load data
    data = np.load(file_path)
    image = data['image']
    label = data['label']
    # normalize image
    mean,std = np.mean(image,axis=(1,2),keepdims=True),np.std(image,axis=(1,2),keepdims=True)
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
    # convert torch tensor
    # copy is very important for pytorch contigous
    image = torch.from_numpy(image.copy()).float()
    label = torch.from_numpy(label).long()
    return image,label


class TestDataset(data.Dataset):
  
  def __init__(self,data_type='region'):
    super(TestDataset,self).__init__()
    self.data_type = data_type
    self.data_names = os.path.join('./dataset_1/test',self.data_type,'%s.npz'%self.data_type)
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
    mean,std = np.mean(image,axis=(1,2),keepdims=True),np.std(image,axis=(1,2),keepdims=True)
    image = (image-mean)/(std+1e-12)
    # convert to pytorch tensor
    image = torch.from_numpy(image).float()
    coord = torch.from_numpy(coord).int()
    return image,coord
  

class UnlabelDataset(data.Dataset):
  
  def __init__(self):
    super(UnlabelDataset,self).__init__()
    self.data_names = os.path.join('./dataset_1/train_unlabel','region.npz')
    self.data_infos = np.load(self.data_names)
    self.images = self.data_infos['data']
    self.coords = self.data_infos['coord']
    # random sampling
    self.nums = len(self.coords)
    
    if self.nums>50000:
      self.sampling = 50000
    else:
      self.sampling = self.nums
    self.index = np.random.choice(self.nums,self.sampling,replace=False)
    
    self.test_imgs = self.images[self.index]
    self.test_coords = self.coords[self.index]
    
    # delete have been used
    self.images = np.delete(self.images,self.index,axis=0)
    self.coords = np.delete(self.coords,self.index,axis=0)
    np.savez(self.data_names,data=self.images,coord=self.coords)
    
  def __len__(self):
    return len(self.test_imgs)
  
  def __getitem__(self,idx):
    image = self.test_imgs[idx]
    coord = self.test_coords[idx]
    # simple normalization
    # every pixel along depth
    mean,std = np.mean(image,axis=(1,2),keepdims=True),np.std(image,axis=(1,2),keepdims=True)
    image = (image-mean)/(std+1e-12)
    # convert to pytorch tensor
    image = torch.from_numpy(image).float()
    coord = torch.from_numpy(coord).int()
    return image,coord
    
    
def TrainLoader(batchsize,datapath):
  traindataset = TrainDataset(datapath)
  trainloader = data.DataLoader(traindataset,
                                batch_size=batchsize,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
  return trainloader


def TestLoader(data_type='region',batchsize=8192):
  testdataset = TestDataset(data_type)
  testloader = data.DataLoader(testdataset,
                               batch_size=batchsize,
                               num_workers=16,
                               pin_memory=True)
  return testloader


def UnlabelLoader(batchsize):
  unlabeldataset = UnlabelDataset()
  dataloader = data.DataLoader( unlabeldataset,
                                batch_size=batchsize,
                                num_workers=16,
                                pin_memory=True)
  return dataloader

if __name__ == '__main__':
  from tqdm import tqdm
  trainloader = TestLoader(batchsize=2048)
  # testloader = iter(UnlabelLoader(batchsize=2048))
  for img,coord in trainloader:
    print(img.shape,coord.shape)
    pass
    
  
    

    
    
    
