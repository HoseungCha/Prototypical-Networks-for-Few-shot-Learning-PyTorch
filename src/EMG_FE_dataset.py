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
from utils import core

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class EMG_dataset(data.Dataset):

    # raw_folder = 'raw'
    # processed_folder = 'data'
    nSub = 10
    nFE = 11
    nSes = 25
    nWin = 41
    nFeat = 36
    nSesTest = 20
    nD = 5


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
        nTn = 1
        dataset = {}
        dataset['tr'] = {}
        dataset['tt'] = {}

        dataset['tr']['x'] = torch.empty(self.nSub, nTn * self.nWin * self.nFE, self.nFeat)
        dataset['tr']['t'] = torch.empty(self.nSub, self.nWin * self.nFE, 1)
        dataset['tr']['s'] = torch.empty(self.nSub, self.nWin * self.nFE, 1)
        dataset['tr']['d'] = torch.empty(self.nSub, self.nWin * self.nFE, 1)

        dataset['tt']['x'] = torch.empty(self.nSub, self.nSesTest * self.nWin * self.nFE, self.nFeat)
        dataset['tt']['t'] = torch.empty(self.nSub, self.nSesTest * self.nWin * self.nFE, 1)
        dataset['tt']['s'] = torch.empty(self.nSub, self.nSesTest * self.nWin * self.nFE, 1)
        dataset['tt']['d'] = torch.empty(self.nSub, self.nSesTest * self.nWin * self.nFE, 1)

        device = 'cuda:0' if torch.cuda.is_available() and option.cuda else 'cpu'



        EMG_tensor_path = os.path.join(root,'EMG_tensor_%d.pth' % nTn)
        if os.path.isfile(EMG_tensor_path) and not os.path.getsize(EMG_tensor_path)/(1024*1024) < 14: # datasetan mb
            start_time = time.time()
            dataset = torch.load(EMG_tensor_path)
            print("Loading dataset Time... %.2fs" % (time.time()-start_time))
        else:
            for i in tqdm(range(self.nSub)):
                temp_path = os.path.join(root, "feat_sub_%d_nt_%d" % (i, nTn-1))
                temp_path = temp_path + '.mat'
                loaded = io.loadmat(temp_path)
                # train data episode (query set here)
                dataset['tr']['x'][i][:][:] = (torch.from_numpy(loaded['tn'][0][0][0]).float().to(device))
                dataset['tr']['t'][i][:][:] = (torch.from_numpy(loaded['tn'][0][0][1]).float().to(device))
                dataset['tr']['s'][i][:][:] = torch.ones((self.nFE * self.nWin),1) * i
                dataset['tr']['d'][i][:][:] = torch.ones((self.nFE * self.nWin),1) * 0

                # train data episode (query set here)
                dataset['tt']['x'][i][:][:] = (torch.from_numpy(loaded['tt'][0][0][0]).float().to(device))
                dataset['tt']['t'][i][:][:] = (torch.from_numpy(loaded['tt'][0][0][1]).float().to(device))
                dataset['tt']['s'][i][:][:] =torch.ones((self.nSesTest *self.nFE * self.nWin),1) * i

                temp_dataset = torch.empty(self.nFE * self.nWin * self.nSesTest, 1)
                for kk in range(1, self.nD):
                    temp_dataset[(kk-1)* self.nD * self.nFE * self.nWin : (kk) * self.nD *self.nFE * self.nWin, :] \
                        = torch.ones(self.nD * self.nFE * self.nWin ,1) * kk
                dataset['tt']['d'][i][:][:] = temp_dataset

            torch.save(dataset, EMG_tensor_path)

        self.dataset = dataset

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[-1]
            file_path = os.path.join(self.root, self.splits_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            orig_root = os.path.join(self.root, self.raw_folder)
            print("== Unzip from " + file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()
        file_processed = os.path.join(self.root, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                shutil.move(os.path.join(orig_root, p, f), file_processed)
            os.rmdir(os.path.join(orig_root, p))
        print("Download finished.")


def find_items(root_dir, classes):
    retour = []
    rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            r = root.split(os.sep)
            lr = len(r)
            label = r[lr - 2] + os.sep + r[lr - 1]
            for rot in rots:
                if label + rot in classes and (f.endswith("mat")):
                    retour.extend([(f, label, root, rot)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes


def load_img(path, idx):
    path, rot = path.split(os.sep + 'rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)

    return x
