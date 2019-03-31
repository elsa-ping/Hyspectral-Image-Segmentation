from tqdm import tqdm
import os
import glob
import numpy as np

datafiles = glob.glob('./dataset_1/train/*.npz')
pbar = tqdm(datafiles)

for file in pbar:
  data = np.load(file)
  image = data['image']
  label = data['label']
  if image.shape!=(8,32,32):
    pbar.set_description('Delete file:%s'%os.path.basename(file))
#    os.remove(file)
  if type(label) != np.ndarray:
    pbar.set_description('Delete file:%s-label:%s'%(os.path.basename(file),label))
    os.remove(file)


