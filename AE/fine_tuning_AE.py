#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:21:16 2019

@author: liang
"""
# adding regularization to fine-tuning AE

import torch
from torch import optim,nn
from autoencoder2_2 import Encoder,Decoder
from dataset import RegularizerDataLoader
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

checkpoints = './checkpoints2_3'
if not os.path.exists(checkpoints):
  os.makedirs(checkpoints)


def train():
  # config
  epochs=100
  batchsize=128
  
  # model
  encoder = Encoder().cuda()
  decoder = Decoder().cuda()
  # load model parameters
  encoder.load_state_dict(torch.load(os.path.join(checkpoints,'Encoder-epoch-5-loss:0.046.pth')))
  decoder.load_state_dict(torch.load(os.path.join(checkpoints,'Decoder-epoch-5-loss:0.046.pth')))
  
  trainloader = RegularizerDataLoader(batchsize)
  
  params = list(encoder.parameters())+list(decoder.parameters())
  # optimizier
  optimizer = optim.SGD( params,
                         lr=0.0001,
                         momentum=0.9,
                         weight_decay=1e-4)
  
  # loss function
  loss_func = nn.MSELoss().cuda()
  best_loss = 1000
  
  for epoch in range(epochs):
    epoch_loss = []
    pbar = tqdm(trainloader)
    for image,label in pbar:
      image = image.cuda()
      encoded_vector = encoder(image)
      output = decoder(encoded_vector)
      # compute loss
      loss = loss_func(output,image)
      # regularizer loss
      encoded_vector = encoded_vector.squeeze()
      encoded_vector = nn.functional.normalize(encoded_vector,dim=-1)
      regular_loss = regularizer_loss(encoded_vector,label)
      total_loss = loss + regular_loss
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      epoch_loss.append(total_loss.item())
      fmt = 'Epoch[{:2d}]-Reconstruct-loss:{:.3f}-Regularizer-loss:{:.3f}-Total-loss:{:.3f}'.\
      format(epoch+1,loss.item(),regular_loss.item(),total_loss.item())
      pbar.set_description(fmt)
    avg_loss = sum(epoch_loss)/len(epoch_loss)
    
    if avg_loss<best_loss:
      best_loss = avg_loss
      torch.save(encoder.state_dict(),os.path.join(checkpoints,'fine-tuning-epoch-%d-loss:%.3f.pth'%(epoch+1,avg_loss)))
  print('Train Finished!!!')


def regularizer_loss(outputs,labels):
  bs = outputs.size(0)
  similarity_maxtrix = outputs.mm(outputs.t())
  
  mask = torch.ones(size=(bs,bs),dtype=torch.uint8)
  
  for i in range(bs):
    for j in range(bs):
      if labels[i]==labels[j]:
        mask[i,j] = 0
  regularizer_loss = torch.mean(torch.abs(similarity_maxtrix[mask]+1e-12))
  
  return regularizer_loss
        
if __name__ == '__main__':
  train()