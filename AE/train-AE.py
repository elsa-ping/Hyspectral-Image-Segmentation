#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:05:53 2019

@author: liang
"""

import torch
from torch import optim,nn
from autoencoder import Encoder,Decoder
from dataset import DataLoader
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

checkpoints = './checkpoints2'
if not os.path.exists(checkpoints):
  os.makedirs(checkpoints)

def train():
  # config
  epochs=20
  batchsize=256
  
  # model
  encoder = Encoder().cuda()
  decoder = Decoder().cuda()
  trainloader = DataLoader('region',batchsize)
  
  params = list(encoder.parameters())+list(decoder.parameters())
  # optimizier
  optimizer = optim.SGD( params,
                         lr=0.01,
                         momentum=0.9,
                         weight_decay=1e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1)
  # loss function
  loss_func = nn.MSELoss().cuda()
  best_loss = 1000
  
  for epoch in range(epochs):
    epoch_loss = []
    pbar = tqdm(trainloader)
    for image,coord in pbar:
      image = image.cuda()
      output = decoder(encoder(image))
      # compute loss
      loss = loss_func(output,image)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss.append(loss.item())
      fmt = 'Epoch[{:2d}]-Loss:{:.3f}'.format(epoch+1,loss.item())
      pbar.set_description(fmt)
    
    avg_loss = sum(epoch_loss)/len(epoch_loss)
    # optimizer update
    scheduler.step()
    if avg_loss<best_loss:
      best_loss = avg_loss
      torch.save(encoder.state_dict(),os.path.join(checkpoints,'Encoder-epoch-%d-loss:%.3f.pth'%(epoch+1,avg_loss)))
      torch.save(decoder.state_dict(),os.path.join(checkpoints,'Decoder-epoch-%d-loss:%.3f.pth'%(epoch+1,avg_loss)))
  print('Train Finished!!!')


if __name__ == '__main__':
  train()  
