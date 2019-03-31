#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:39:48 2019

@author: liang
"""

# postprocess
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

predict_dir = './predicts'
if not os.path.exists(predict_dir):
  os.makedirs(predict_dir)

class2value = {0:0,1:0,2:60,3:40}

def postprocess(num=1):
 
  filenames = os.listdir('./results/svm_results')
  filepaths = [os.path.join('./results/svm_results',name) for name in filenames]
  Predictions = np.zeros(shape=(17810,50362),dtype=np.int16)
  
  for filepath in tqdm(filepaths,desc='Generating Prediction'):
    data = np.load(filepath)
    coord = data['coord']
    predict = data['predict']
    predict_value = [class2value[label] for label in predict]
    for (idx1,idy1),value in zip(coord,predict_value):
      if idx1==-1:
        Predictions[-18:,idy1*32:(idy1+1)*32] = value
      elif idy1==-1:
        Predictions[idx1*32:(idx1+1)*32,-26:] = value
      else:
        x1,x2 = idx1*32,(idx1+1)*32
        y1,y2 = idy1*32,(idy1+1)*32
        Predictions[x1:x2,y1:y2] = value
        
  # convert to pil image
  prediction = Image.fromarray(Predictions)
  prediction.save(os.path.join(predict_dir,'%d.tif'%num))
  
  
if __name__ == '__main__':
  postprocess(num=10)
