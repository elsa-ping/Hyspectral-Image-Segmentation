#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:43:18 2019

@author: liang
"""

import torch
from torch import optim,nn
from model2 import TinyNet
from dataset import TrainLoader,UnlabelLoader
from tqdm import tqdm
import os
import numpy as np
import re


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.manual_seed(0)

checkpoints = './checkpoints4'
if not os.path.exists(checkpoints):
  os.makedirs(checkpoints)

def train():
  # config
  recurrents = 10
  epochs = 10
  batchsize = 256 
  # model
  tinynet1 = TinyNet().cuda()
  tinynet2 = TinyNet().cuda()
  # optimizier
  optimizer1 = optim.Adam(tinynet1.parameters(),
                         betas=(0.5,0.999),
                         weight_decay=1e-3)
  
  optimizer2 = optim.Adam(tinynet2.parameters(),
                         betas=(0.5,0.999),
                         weight_decay=1e-3)
  
  # loss function
  loss_func = nn.CrossEntropyLoss().cuda()
  # loading parameters
  # tinynet.load_state_dict(torch.load('./checkpoints1/Num-10.pth'))

  threshold = 0.92
  pattern = re.compile('(^\d+.+\.npz)')
  
  # train datapath
  datapath1 = './dataset_1/train/train1'
  datapath2 = './dataset_1/train/train2'
  
  for self_num in range(0,recurrents):
    trainloader1 = TrainLoader(batchsize,datapath1)
    trainloader2 = TrainLoader(batchsize,datapath2)
    print('\033[1;32m----------------Self-Training Nums:%d----------------\033[0m'%(self_num+1))
    tinynet1.train()
    tinynet2.train()
    
    for epoch in range(epochs):
      # compute epoch accuracy 
      num_corrects = 0
      num_total = 0
      epoch_loss = []
      pbar = tqdm(trainloader1)
      for image,label in pbar:
        image = image.cuda()
        label = label.cuda()
        output = tinynet1(image)
        # compute loss
        loss = loss_func(output,label)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        # ----------------------# 
        predicts = torch.argmax(output,dim=-1)
        corrects = (predicts==label).sum()
        batch_acc = corrects.float()/label.size(0)
        num_corrects += corrects
        num_total += label.size(0)
        epoch_loss.append(loss.item())
        fmt = 'Epoch[{:2d}]-Loss:{:.3f}-Batch_acc:{:.3f}'.format(epoch+1,loss.item(),batch_acc.item())
        pbar.set_description(fmt)
        
      epoch_accu = num_corrects.float()/num_total
      avg_loss = sum(epoch_loss)/len(epoch_loss)
      print('\033[1;31mNums:[%2d/%2d]-Epoch:%2d-Accu:%.3f-Loss:%.3f\033[0m'\
                  %(self_num+1,recurrents,epoch+1,epoch_accu.item(),avg_loss))
  
    # every recurrent save model
    torch.save(tinynet1.state_dict(),os.path.join(checkpoints,'model1-Num-%d.pth'%(self_num+1)))
    
    for epoch in range(epochs):
      # compute epoch accuracy 
      num_corrects = 0
      num_total = 0
      epoch_loss = []
      pbar = tqdm(trainloader2)
      for image,label in pbar:
        image = image.cuda()
        label = label.cuda()
        output = tinynet2(image)
        # compute loss
        loss = loss_func(output,label)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        # ----------------------# 
        predicts = torch.argmax(output,dim=-1)
        corrects = (predicts==label).sum()
        batch_acc = corrects.float()/label.size(0)
        num_corrects += corrects
        num_total += label.size(0)
        epoch_loss.append(loss.item())
        fmt = 'Epoch[{:2d}]-Loss:{:.3f}-Batch_acc:{:.3f}'.format(epoch+1,loss.item(),batch_acc.item())
        pbar.set_description(fmt)
        
      epoch_accu = num_corrects.float()/num_total
      avg_loss = sum(epoch_loss)/len(epoch_loss)
      print('\033[1;31mNums:[%2d/%2d]-Epoch:%2d-Accu:%.3f-Loss:%.3f\033[0m'\
                  %(self_num+1,recurrents,epoch+1,epoch_accu.item(),avg_loss))
  
    # every recurrent save model
    torch.save(tinynet2.state_dict(),os.path.join(checkpoints,'model2-Num-%d.pth'%(self_num+1)))
    
    # delete sampling
    delete_samples(datapath1,pattern)
    delete_samples(datapath2,pattern)
    
    # test for generate fake label
    tinynet1.eval()
    tinynet2.eval()
    
    with torch.no_grad():
      count1 = 0
      count2 = 0
      testloader = UnlabelLoader(2048)
      
      for idx,(image,coord) in enumerate(tqdm(testloader,desc='Generating Fake Labels')):
        image = image.cuda()
        # model1
        output1 = tinynet1(image)
        output1 = nn.functional.softmax(output1,dim=-1)
        probs1,predicts1 = torch.max(output1,dim=-1)
        
        probs1 = probs1.cpu().numpy()
        predicts1 = predicts1.cpu().numpy()
        
        mask1 = probs1>(threshold+0.05*self_num)
        imgs1 = image.cpu().numpy()[mask1]
        predicts1 = predicts1[mask1]
        
        
        # model2
        output2 = tinynet2(image)
        output2 = nn.functional.softmax(output2,dim=-1)
        probs2,predicts2 = torch.max(output2,dim=-1)
        
        probs2 = probs2.cpu().numpy()
        predicts2 = predicts2.cpu().numpy()
        
        mask2 = probs2>(threshold+0.05*self_num)
        imgs2 = image.cpu().numpy()[mask2]
        predicts2 = predicts2[mask2]

        
        for img,lab in zip(imgs1,predicts1):
          count1 += 1
          np.savez(os.path.join(datapath2,'%d_%d.npz'%(self_num,count1)),image=img,label=lab)
          
        for img,lab in zip(imgs2,predicts2):
          count2 += 1
          np.savez(os.path.join(datapath1,'%d_%d.npz'%(self_num,count2)),image=img,label=lab)
    
  print('Train Finished!!!')


def delete_samples(train_dir,pat):
  files = os.listdir(train_dir)
  select_files = []
  for file in files:
    res = re.findall(pat,file)
    if len(res)>0:
      select_files.append(file)
  select_paths = [os.path.join(train_dir,file) for file in select_files]
  for path in select_paths:
    os.remove(path)



if __name__ == '__main__':
  train()  
      
  
