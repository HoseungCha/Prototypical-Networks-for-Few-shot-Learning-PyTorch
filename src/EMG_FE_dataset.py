# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os
import scipy.io as io
from tqdm import tqdm
import time
'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class EMG_dataset(data.Dataset):

    raw_folder = 'raw'
    processed_folder = 'data'
    nSub = 10
    nFE = 11
    nSes = 25
    nWin = 41



    def __init__(self, mode='train', root='..' + os.sep + 'dataset_EMG',
                 transform=None, target_transform=None, download=True, option=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(EMG_dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        dataset = {}
        dataset['x'] = []
        dataset['t'] = []
        dataset['s'] = []
        dataset['d'] = []
        # dataset['win'] = []

        device = 'cuda:0' if torch.cuda.is_available() and option.cuda else 'cpu'

        EMG_tensor_path = os.path.join(root,'data','EMG_tensor.pth')
        if os.path.isfile(EMG_tensor_path) and not os.path.getsize(EMG_tensor_path)/(1024*1024) < 279: # datasetan mb
            start_time = time.time()
            dataset = torch.load(EMG_tensor_path)
            print("Loading dataset Time... %.2fs" % (time.time()-start_time))
        else:
            for s in tqdm(range(self.nSub)):
                for t in range(self.nFE):
                    for k in range(self.nSes):
                        for l in range(self.nWin):
                            temp_path = os.path.join(root, 'data',
                                                     "Sub-%02d" % (s),
                                                     "FE-%02d" % (t),
                                                     "%04d.mat" %  (k * self.nWin + l))
                            dataset['x'].append(torch.from_numpy(io.loadmat(temp_path)['segment']).float().to(device))
                            dataset['s'].append(s)
                            dataset['t'].append(t)
                            if k < 5:
                                d = 1
                            elif k >= 5 and k < 10:
                                d = 2
                            elif k >= 10 and k < 15:
                                d = 3
                            elif k >= 15 and k < 20:
                                d = 4
                            elif k>= 20:
                                d = 5
                            dataset['d'].append(k)

            torch.save(dataset, EMG_tensor_path)
        self.x = dataset['x']
        self.t = dataset['t']
        self.s = dataset['s']
        self.d = dataset['d']

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.t[idx]

    def __len__(self):
        return len(self.all_items)



