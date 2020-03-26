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


    def __init__(self, option, mode, index= None):
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
        self.index_test_subject = option.test_subject_index
        self.sample_per_class = option.num_support_tr
        self.iterations = option.iterations
        self.nDomain = 5
        self.mode = mode


    @property
    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        index = self.index
        iterations = self.iterations


        # Todo: 다른 피험자로부터 train test 나눔
        # sVal = core.getIdxExclude_of_inputIndex(range(0, self.nSub), [self.index_test_subject])
        # sVal = np.random.permutation(sVal)[0]
        # sTrain = core.getIdxExclude_of_inputIndex(range(0, self.nSub), [self.index_test_subject, sVal])
        # sTrain = np.random.permutation(sTrain)[0]
        if self.mode == 'test':
            dQuery = 0
            support = []
            for dSupport in core.getIdxExclude_of_inputIndex(range(self.nDomain), [dQuery]):
                support.append(get_idx_of_support(self.nFE, index, dQuery, dSupport))
            query = get_idx_of_query(self.nFE, index, self.index_test_subject, dQuery, spc)

            # index 합치기
            batch = []
            batch.append[np.array(query).ravel().tolist()]
            batch.append[np.array(support).ravel().tolist()]
            yield np.array(batch).ravel().tolist()
        else:
            for sVal in core.getIdxExclude_of_inputIndex(range(0, self.nSub), [self.index_test_subject]):
                # TODO: Trianin/ sVal에서 leave-one-subject-out cross vlalidation의 instacne 구함
                if self.mode == 'train':
                    for sTrain in core.getIdxExclude_of_inputIndex(range(0, self.nSub), [self.index_test_subject, sVal]):
                        # Todo: training data
                        query = []
                        for dQuery in range(self.nDomain): # Todo: domain 랜덤으로 두개 선택 하기나누기
                            # dQuery = 1
                            if self.mode == 'train':
                                support = []
                                for dSupport in core.getIdxExclude_of_inputIndex(range(self.nDomain), [dQuery]):
                                    support.append(get_idx_of_support(self.nFE, index, sTrain, dSupport))
                                query = query + get_idx_of_query(self.nFE, index, sTrain, dQuery, spc)
                    # index 합치기
                    batch = []
                    batch.append[np.array(query).ravel().tolist()]
                    batch.append[np.array(support).ravel().tolist()]

                    yield np.array(batch).ravel().tolist()

                if self.mode =='val':
                    support = []
                    for dSupport in core.getIdxExclude_of_inputIndex(range(self.nDomain), [dQuery]):
                        support.append(get_idx_of_support(self.nFE, index, dQuery, dSupport))
                    query = query + get_idx_of_query(self.nFE, index, sTrain, dQuery, spc)

                    # index 합치기
                    batch = []
                    batch.append[np.array(query).ravel().tolist()]
                    batch.append[np.array(support).ravel().tolist()]

                    yield np.array(batch).ravel().tolist()





                #     # batch에 저장
                # batch = add_data_into_batch(batch, self.nFE, range(self.nFE), spc, query)
                # batch = add_data_into_batch(batch, self.nFE, range(self.nFE, 2 * self.nFE), spc, support)
                #
                # # Todo: validation data
                # temp = np.random.permutation(list(range(self.nDomain)))
                # dSupport = temp[0]
                # dQuery = temp[1]
                #
                # # query, support = get_idx_of_query_and_support(self.nFE, index, sVal, dQuery, dSupport, spc)
                #
                # # batch에 저장
                # batch = add_data_into_batch(batch, self.nFE, range(2 * self.nFE, 3 * self.nFE), spc, query)
                # batch = add_data_into_batch(batch, self.nFE, range(3 * self.nFE, 4* self.nFE), spc, support)
                #
                # # Todo: test data



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

def get_idx_of_query(nFE,index, sTrain, dQuery, spc):
    query = []
    for t in range(nFE):
        found = get_idx_from_std(index, sTrain, t, dQuery)
        query.append(found[torch.randperm(found.__len__())[:spc]].tolist())
    return query

def get_idx_of_support(nFE,index, sTrain, dSupport):
    support = []
    for t in range(nFE):
        found = get_idx_from_std(index, sTrain, t, dSupport)
        support.append(found[torch.randperm(found.__len__())[:]].tolist())
    return support


def add_data_into_batch(batch, nFE, varRange, spc, query):

    for t, c in enumerate(list(varRange)):
        s = slice(c * spc, (c + 1) * spc)
        batch[s] = query[t]
    # shuffle
    batch[(c + 1 - nFE) * spc:(c + 1) * spc] = \
        batch[(c + 1 - nFE) * spc:(c + 1) * spc][torch.randperm(spc * nFE)]

    return batch
