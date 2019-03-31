#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:42:19 2019

@author: liang
"""

# build svm for classification
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.externals import joblib
import glob
from tqdm import tqdm


svm_dir = './svm'
if not os.path.exists(svm_dir):
  os.makedirs(svm_dir)

flags = 'test' # or test

def train():
  # load train data
  data = np.load('./svm_train_data.npz')
  train_feats,train_labs = data['features'],data['labels']
  # train_feats,test_feats,train_labs,test_labs = train_test_split(features,labs,test_size=0.25,random_state=33)
  print('Train shape',train_feats.shape)
  # build svm
  svm = SVC(C=100,verbose=2,gamma='auto',kernel='rbf')
  # train
  svm.fit(train_feats,train_labs)
  print('Train SVM Finished!!!')
  """
  y_predict = svm.predict(test_feats)
  print(classification_report(test_labs,y_predict))
  print(accuracy_score(test_labs,y_predict))
  """
  # save model
  joblib.dump(svm,os.path.join(svm_dir,'svm_model.pkl'))


def test():
  # load testing features
  datatype = 'margin'
  files = glob.glob(os.path.join('./svm_features',datatype,'*.npz'))
  results_dir = os.path.join('./results/svm_results')
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  # testing
  # load model
  print('Loading model from pickle!!!')
  svm = joblib.load(os.path.join(svm_dir,'svm_model.pkl'))
  print('Loading model finished!!!')
  # multi-process to predicting
  joblib.Parallel(n_jobs=32)\
  (joblib.delayed(multi_process_predict)(svm,file,idx,results_dir,datatype) for idx,file in enumerate(tqdm(files)))
  


def multi_process_predict(classifier,datapath,idx,results_dir,datatype):
  """
  # using multi-process to speed up svm predicting(can speed up 12 times)
  """
  data = np.load(datapath)
  try:
    features = data['feature']
    coords = data['coord']
    test_preds = classifier.predict(features)
    test_probs = classifier.predict_proba(features)
    mask = np.max(test_probs,axis=-1)<0.9
    test_preds[mask] = 0
    np.savez(os.path.join(results_dir,'%s_%d.npz'%(datatype,idx+1)),coord=coords,predict=test_preds)
  except:
    pass 
  
if __name__ == '__main__':
  
  if flags == 'train':
    train()
  else:
    test()