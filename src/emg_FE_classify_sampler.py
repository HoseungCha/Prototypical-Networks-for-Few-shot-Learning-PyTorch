# coding=utf-8
import numpy as np
import torch
from utils import core

class EMG_FE_Classify_Sampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''
    nSub = 10
    nFE = 11
    nSes = 25
    iSesTrain = range(0, 5)
    nWin = 41


    def __init__(self, option, sTest,index= None):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(EMG_FE_Classify_Sampler, self).__init__()
        # Todo: Possible query and support indexes
        # Todo: Confirm test subject
        self.index = index
        self.sTest = sTest
        self.sample_per_class = option.num_support_tr
        self.iterations = option.iterations
        self.nDomain = 5


    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.nFE
        sTest = self.sTest
        index = self.index

        for it in range(self.iterations):
            temp = np.random.permutation(core.getIdxExclude_of_inputIndex(range(self.nSub), [sTest]))[:2]
            si = temp[0]
            sj = temp[1]
            for t in range(self.nFE):
                support = get_idx_of_support(self.nFE, index, si, 0, spc)
                query = []
                for d  in range(1, self.nDomain):
                    query.append(get_idx_of_support(self.nFE, index, si, d))

                yield 




    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


def get_idx_from_std(index, s, t, d):
    a1 = torch.IntTensor(index['s']).eq(s).nonzero()
    a2 = torch.IntTensor(index['t']).eq(t).nonzero()
    a3 = torch.IntTensor(index['d']).eq(d).nonzero()
    return torch.IntTensor(np.intersect1d(np.intersect1d(a1.numpy(), a2.numpy()), a3.numpy()))

def get_idx_of_query(nFE,index, sTrain, dQuery,spc):
    query = []
    support = []
    for t in range(nFE):
        found = get_idx_from_std(index, sTrain, t, dQuery)
        query.append((found)[torch.randperm(found.__len__())[:spc]])

    return query

def get_idx_of_support(nFE,index, s, dSupport,):
    support = []
    for t in range(nFE):
        found = get_idx_from_std(index, s, t, dSupport)
        support.append((found)[torch.randperm(found.__len__())[:]])
    return support


def get_idx_of_query_and_support(nFE,index, s, dQuery, dSupport, spc):
    query = []
    support = []
    for t in range(nFE):
        found = get_idx_from_std(index, s, t, dQuery)
        query.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])
        found = get_idx_from_std(index, s, t, dSupport)
        support.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])
    return query, support


def add_data_into_batch(batch, nFE, varRange, spc, query):

    for t, c in enumerate(list(varRange)):
        s = slice(c * spc, (c + 1) * spc)
        batch[s] = query[t]
    # shuffle
    batch[(c + 1 - nFE) * spc:(c + 1) * spc] = \
        batch[(c + 1 - nFE) * spc:(c + 1) * spc][torch.randperm(spc * nFE)]

    return batch
