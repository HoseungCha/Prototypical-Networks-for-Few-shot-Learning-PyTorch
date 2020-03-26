# coding=utf-8
import numpy as np
import torch
from utils import core
from tqdm import tqdm

class Batch_Sampler_Val(object):
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
    nWin = 41


    def __init__(self, option, valSubject = 1, index= None, mode= None):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(Batch_Sampler_Val, self).__init__()
        # Todo: Possible query and support indexes
        # Todo: Confirm test subject
        self.index = index
        self.index_test_subject = option.test_subject_index
        self.sample_per_class = option.num_support_tr
        # self.iterations = option.iterations
        self.nDomain = 5
        self.mode = mode
        self.iterations = self.nDomain  # 360번
        self.valSubject = valSubject



    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        index = self.index

        sVal = self.valSubject

        # Todo: 테스트 피험자는 정해짐, 나머지 데이터로 CV하면서 vali정확도와 test정홗도가 향상되는지 확인


        dSupport = 0
        support = get_idx_of_support(self.nFE, index, sVal, dSupport, spc)
        query = []
        for dQuery in core.getIdxExclude_of_inputIndex(range(self.nDomain), [dSupport]):
            query.append(get_idx_of_query(self.nFE, index, sVal, dQuery))
            # index 합치기
            yield (np.array(support).ravel().tolist() + np.array(query).ravel().tolist())


    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


def get_idx_from_std(index, s, t, d):
    a1 = torch.IntTensor(index['s']).eq(s).nonzero()
    a2 = torch.IntTensor(index['t']).eq(t).nonzero()
    a3 = torch.IntTensor(index['d']).eq(d).nonzero()
    return (np.intersect1d(np.intersect1d(a1.numpy(), a2.numpy()), a3.numpy()))

def get_idx_of_query(nFE,index, s, dQuery):
    query = []
    for t in range(nFE):
        found = get_idx_from_std(index, s, t, dQuery)
        query.append(found)
    return query

def get_idx_of_support(nFE,index, s, dSupport,spc):
    support = []
    for t in range(nFE):
        found = get_idx_from_std(index, s, t, dSupport)
        support.append(found[:spc])
    return support


# def get_idx_of_query_and_support(nFE,index, s, dQuery, dSupport, spc):
#     query = []
#     support = []
#     for t in range(nFE):
#         found = get_idx_from_std(index, s, t, dQuery)
#         query.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])
#         found = get_idx_from_std(index, s, t, dSupport)
#         support.append(torch.IntTensor(found)[torch.randperm(found.__len__())[:spc]])
#     return query, support


# def add_data_into_batch(batch, nFE, varRange, spc, query):
#
#     for t, c in enumerate(list(varRange)):
#         s = slice(c * spc, (c + 1) * spc)
#         batch[s] = query[t]
#     # shuffle
#     batch[(c + 1 - nFE) * spc:(c + 1) * spc] = \
#         batch[(c + 1 - nFE) * spc:(c + 1) * spc][torch.randperm(spc * nFE)]
#
#     return batch
